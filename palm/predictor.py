"""High-level annotation and WT/mutant comparison interface for PALM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .model import PALM, PALMOutput, load_model, occurrence_probability


def normalize_sequence(sequence: str) -> str:
    sequence = "".join(str(sequence).split()).upper()
    if not sequence:
        raise ValueError("Protein sequence is empty.")
    invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWYBXZJUO"))
    if invalid:
        raise ValueError(f"Unsupported amino-acid characters: {invalid}")
    return sequence


def mutate_sequence(wt_sequence: str, aa_pos: int, aa_to: str) -> str:
    """Return a mutant sequence; ``aa_pos`` is one-based."""
    sequence = normalize_sequence(wt_sequence)
    aa_to = normalize_sequence(aa_to)
    if len(aa_to) != 1:
        raise ValueError("aa_to must contain exactly one amino acid.")
    if aa_pos < 1 or aa_pos > len(sequence):
        raise ValueError(f"Position {aa_pos} is outside a sequence of length {len(sequence)}.")
    return sequence[: aa_pos - 1] + aa_to + sequence[aa_pos:]


def _label_information(
    checkpoint: dict[str, object], ipr_tsv: str | Path | None
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    domain_names: dict[str, str] = {}
    site_names: dict[str, str] = {}
    if ipr_tsv is not None and Path(ipr_tsv).is_file():
        table = pd.read_csv(ipr_tsv, sep="\t")
        if not {"ipr", "type", "name"}.issubset(table.columns):
            table = pd.read_csv(
                ipr_tsv, sep="\t", names=["ipr", "type", "name"], header=0
            )
        for row in table.itertuples(index=False):
            ipr = str(getattr(row, "ipr"))
            annotation_type = str(getattr(row, "type"))
            name = str(getattr(row, "name"))
            (domain_names if annotation_type == "Domain" else site_names)[ipr] = name
    domain = {
        int(index): {
            "ipr": str(ipr),
            "name": domain_names.get(str(ipr), str(ipr)),
            "type": "Domain",
        }
        for ipr, index in dict(checkpoint["ipr2domain"]).items()
    }
    site = {
        int(index): {
            "ipr": str(ipr),
            "name": site_names.get(str(ipr), str(ipr)),
            "type": "Site",
        }
        for ipr, index in dict(checkpoint["ipr2site"]).items()
    }
    return domain, site


@dataclass
class AnnotationResult:
    sequence: str
    table: pd.DataFrame
    domain_class_probabilities: np.ndarray
    site_class_probabilities: np.ndarray
    residue_embeddings: np.ndarray
    domain_labels: dict[int, dict[str, str]]
    site_labels: dict[int, dict[str, str]]

    @property
    def predicted_length(self) -> int:
        return len(self.table)

    def save(self, prefix: str | Path) -> tuple[Path, Path]:
        """Save a readable TSV plus lossless arrays in a compressed NPZ."""
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        tsv_path = prefix.with_suffix(".tsv")
        npz_path = prefix.with_suffix(".npz")
        self.table.to_csv(tsv_path, sep="\t", index=False)
        np.savez_compressed(
            npz_path,
            sequence=np.asarray(self.sequence),
            domain_class_probabilities=self.domain_class_probabilities,
            site_class_probabilities=self.site_class_probabilities,
            residue_embeddings=self.residue_embeddings,
            domain_labels=np.asarray(json.dumps(self.domain_labels)),
            site_labels=np.asarray(json.dumps(self.site_labels)),
        )
        return tsv_path, npz_path


class PALMPredictor:
    """Load the final checkpoint once and annotate one or more sequences."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        ipr_tsv: str | Path | None = None,
        device: str | torch.device = "auto",
        max_len: int | None = None,
        local_files_only: bool = False,
        amp: bool = True,
    ) -> None:
        self.model, self.checkpoint = load_model(
            checkpoint,
            device=device,
            max_len=max_len,
            local_files_only=local_files_only,
            require_epoch8=True,
        )
        self.amp = bool(amp)
        self.domain_labels, self.site_labels = _label_information(
            self.checkpoint, ipr_tsv
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _top_classes(
        self,
        probabilities: np.ndarray,
        labels: dict[int, dict[str, str]],
        top_k: int,
    ) -> list[str]:
        count = min(max(int(top_k), 1), probabilities.shape[1])
        indices = np.argpartition(probabilities, -count, axis=1)[:, -count:]
        rows: list[str] = []
        for position, selected in enumerate(indices):
            ordered = selected[np.argsort(probabilities[position, selected])[::-1]]
            rows.append(
                json.dumps(
                    [
                        {
                            **labels.get(int(index), {"ipr": str(index), "name": str(index)}),
                            "class_id": int(index),
                            "probability": float(probabilities[position, index]),
                        }
                        for index in ordered
                    ],
                    separators=(",", ":"),
                )
            )
        return rows

    def _result(
        self, sequence: str, output: PALMOutput, batch_index: int, top_k: int
    ) -> AnnotationResult:
        length = int(output.residue_mask[batch_index].sum().item())
        domain_occurrence = torch.softmax(
            output.domain_occurrence_logits[batch_index, :length].float(), dim=-1
        ).cpu().numpy()
        site_occurrence = torch.softmax(
            output.site_occurrence_logits[batch_index, :length].float(), dim=-1
        ).cpu().numpy()
        domain_class = torch.softmax(
            output.domain_class_logits[batch_index, :length].float(), dim=-1
        ).cpu().numpy()
        site_class = torch.softmax(
            output.site_class_logits[batch_index, :length].float(), dim=-1
        ).cpu().numpy()
        domain_id = domain_class.argmax(axis=1)
        site_id = site_class.argmax(axis=1)
        table = pd.DataFrame(
            {
                "position": np.arange(1, length + 1),
                "amino_acid": list(sequence[:length]),
                "domain_occurrence_probability": domain_occurrence[:, 1:].sum(axis=1),
                "domain_none_probability": domain_occurrence[:, 0],
                "domain_single_probability": domain_occurrence[:, 1],
                "domain_multiple_probability": domain_occurrence[:, 2],
                "domain_occurrence_class": domain_occurrence.argmax(axis=1),
                "domain_top_class_id": domain_id,
                "domain_top_ipr": [self.domain_labels[int(i)]["ipr"] for i in domain_id],
                "domain_top_name": [self.domain_labels[int(i)]["name"] for i in domain_id],
                "domain_top_probability": domain_class.max(axis=1),
                "site_occurrence_probability": site_occurrence[:, 1:].sum(axis=1),
                "site_none_probability": site_occurrence[:, 0],
                "site_single_probability": site_occurrence[:, 1],
                "site_multiple_probability": site_occurrence[:, 2],
                "site_occurrence_class": site_occurrence.argmax(axis=1),
                "site_top_class_id": site_id,
                "site_top_ipr": [self.site_labels[int(i)]["ipr"] for i in site_id],
                "site_top_name": [self.site_labels[int(i)]["name"] for i in site_id],
                "site_top_probability": site_class.max(axis=1),
            }
        )
        table["top_domain_classes"] = self._top_classes(
            domain_class, self.domain_labels, top_k
        )
        table["top_site_classes"] = self._top_classes(
            site_class, self.site_labels, top_k
        )
        return AnnotationResult(
            sequence=sequence,
            table=table,
            domain_class_probabilities=domain_class,
            site_class_probabilities=site_class,
            residue_embeddings=output.residue_embeddings[
                batch_index, :length
            ].float().cpu().numpy(),
            domain_labels=self.domain_labels,
            site_labels=self.site_labels,
        )

    @torch.inference_mode()
    def annotate_batch(
        self, sequences: Sequence[str], *, top_k: int = 5
    ) -> list[AnnotationResult]:
        normalized = [normalize_sequence(sequence) for sequence in sequences]
        output = self.model(normalized, amp=self.amp)
        return [
            self._result(sequence, output, index, top_k)
            for index, sequence in enumerate(normalized)
        ]

    def annotate(self, sequence: str, *, top_k: int = 5) -> AnnotationResult:
        return self.annotate_batch([sequence], top_k=top_k)[0]

    def compare_variant(
        self,
        wt_sequence: str,
        mutant_sequence: str,
        aa_pos: int,
        *,
        local_radius: int = 5,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Compare WT and mutant PALM outputs without fitting a classifier."""
        wt_sequence = normalize_sequence(wt_sequence)
        mutant_sequence = normalize_sequence(mutant_sequence)
        if len(wt_sequence) != len(mutant_sequence):
            raise ValueError("WT and mutant sequences must have equal length.")
        wt, mutant = self.annotate_batch([wt_sequence, mutant_sequence], top_k=top_k)
        if aa_pos < 1 or aa_pos > wt.predicted_length:
            raise ValueError(
                f"Position {aa_pos} is outside the model output length {wt.predicted_length}."
            )
        index = aa_pos - 1
        lower = max(0, index - int(local_radius))
        upper = min(wt.predicted_length, index + int(local_radius) + 1)
        embedding_delta = np.linalg.norm(
            wt.residue_embeddings - mutant.residue_embeddings, axis=1
        )

        def distribution_l1(columns: list[str]) -> np.ndarray:
            return np.abs(
                wt.table[columns].to_numpy(dtype=float)
                - mutant.table[columns].to_numpy(dtype=float)
            ).sum(axis=1)

        domain_occurrence_delta = distribution_l1(
            [
                "domain_none_probability",
                "domain_single_probability",
                "domain_multiple_probability",
            ]
        )
        site_occurrence_delta = distribution_l1(
            [
                "site_none_probability",
                "site_single_probability",
                "site_multiple_probability",
            ]
        )
        domain_class_delta = np.abs(
            wt.domain_class_probabilities - mutant.domain_class_probabilities
        ).sum(axis=1)
        site_class_delta = np.abs(
            wt.site_class_probabilities - mutant.site_class_probabilities
        ).sum(axis=1)
        per_residue = pd.DataFrame(
            {
                "position": np.arange(1, wt.predicted_length + 1),
                "amino_acid_wt": list(wt_sequence[: wt.predicted_length]),
                "amino_acid_mutant": list(mutant_sequence[: wt.predicted_length]),
                "embedding_l2": embedding_delta,
                "domain_occurrence_l1": domain_occurrence_delta,
                "site_occurrence_l1": site_occurrence_delta,
                "domain_class_l1": domain_class_delta,
                "site_class_l1": site_class_delta,
            }
        )

        def summarize(values: np.ndarray) -> dict[str, float]:
            return {
                "mean": float(values.mean()),
                "max": float(values.max()),
                "mutation": float(values[index]),
                "local_mean": float(values[lower:upper].mean()),
            }

        return {
            "aa_pos": int(aa_pos),
            "local_radius": int(local_radius),
            "predicted_length": wt.predicted_length,
            "embedding_l2": summarize(embedding_delta),
            "domain_occurrence_l1": summarize(domain_occurrence_delta),
            "site_occurrence_l1": summarize(site_occurrence_delta),
            "domain_class_l1": summarize(domain_class_delta),
            "site_class_l1": summarize(site_class_delta),
            "per_residue": per_residue,
            "wt": wt,
            "mutant": mutant,
        }

