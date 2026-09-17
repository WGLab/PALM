"""ClinVar-trained pathogenicity heads for the final PALM checkpoint.

These heads are supervised downstream models.  They are intentionally kept
separate from zero-shot PALM disruption scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .hub import (
    DEFAULT_ANNOTATION_REFERENCE,
    DEFAULT_PATHOGENICITY_HEAD_REFERENCES,
    resolve_artifact,
)
from .model import PALM, load_model
from .predictor import normalize_sequence


VALID_SUMMARIES = {"min", "max", "mean", "std", "mutation", "local"}


def parse_summaries(value: str | Sequence[str]) -> list[str]:
    summaries = (
        [part.strip().lower() for part in value.split(",") if part.strip()]
        if isinstance(value, str)
        else [str(part).lower() for part in value]
    )
    invalid = sorted(set(summaries) - VALID_SUMMARIES)
    if invalid or not summaries:
        raise ValueError(f"Invalid summaries {invalid}; valid={sorted(VALID_SUMMARIES)}")
    return summaries


def crop_around_position(
    sequence: str, position: int, max_len: int
) -> tuple[str, int, int]:
    """Return crop, one-based position in crop, and zero-based crop start."""
    if position < 1 or position > len(sequence):
        raise ValueError(f"Position {position} is outside sequence length {len(sequence)}.")
    if len(sequence) <= max_len:
        return sequence, position, 0
    start = max(0, min(position - 1 - max_len // 2, len(sequence) - max_len))
    return sequence[start : start + max_len], position - start, start


class PathogenicityHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        input_layernorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        if input_layernorm:
            layers.append(nn.LayerNorm(previous))
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    return (values * mask.unsqueeze(-1)).sum(dim=1) / denominator


def _masked_min(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = values.masked_fill(~mask.bool().unsqueeze(-1), torch.inf).min(dim=1).values
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = values.masked_fill(~mask.bool().unsqueeze(-1), -torch.inf).max(dim=1).values
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = _masked_mean(values, mask)
    denominator = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    variance = (((values - mean.unsqueeze(1)) ** 2) * mask.unsqueeze(-1)).sum(dim=1)
    return torch.sqrt((variance / denominator).clamp_min(0))


class PairFeatureExtractor(nn.Module):
    """Frozen PALM plus the exact feature summaries used by ClinVar heads."""

    def __init__(
        self,
        palm: PALM,
        num_domain_classes: int,
        num_site_classes: int,
        summaries: Sequence[str],
        local_radius: int,
    ) -> None:
        super().__init__()
        self.palm = palm
        self.summaries = parse_summaries(summaries)
        self.local_radius = int(local_radius)
        single = int(palm.encoder.out_dim) + 3 + int(num_domain_classes) + 3 + int(num_site_classes)
        self.sequence_feature_dim = len(self.summaries) * single
        self.feature_dim = 4 * self.sequence_feature_dim

    def _summarize(
        self, values: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        lengths = mask.sum(dim=1).long()
        indices = positions.to(values.device).long() - 1
        safe_indices = indices.clamp(min=0, max=values.shape[1] - 1)
        point = values[torch.arange(values.shape[0], device=values.device), safe_indices]
        point = point * ((indices >= 0) & (indices < lengths)).float().unsqueeze(-1)
        residue_indices = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
        local_mask = (
            (residue_indices >= (indices - self.local_radius).unsqueeze(1))
            & (residue_indices <= (indices + self.local_radius).unsqueeze(1))
        ).float() * mask
        outputs: list[torch.Tensor] = []
        for summary in self.summaries:
            outputs.append(
                {
                    "min": lambda: _masked_min(values, mask),
                    "max": lambda: _masked_max(values, mask),
                    "mean": lambda: _masked_mean(values, mask),
                    "std": lambda: _masked_std(values, mask),
                    "mutation": lambda: point,
                    "local": lambda: _masked_mean(values, local_mask),
                }[summary]()
            )
        return torch.cat(outputs, dim=-1).float()

    @torch.no_grad()
    def sequence_features(
        self, sequences: Sequence[str], positions: torch.Tensor, amp: bool
    ) -> torch.Tensor:
        output = self.palm(list(sequences), amp=amp)
        combined = torch.cat(
            [
                output.residue_embeddings,
                torch.softmax(output.domain_occurrence_logits.float(), dim=-1),
                torch.softmax(output.domain_class_logits.float(), dim=-1),
                torch.softmax(output.site_occurrence_logits.float(), dim=-1),
                torch.softmax(output.site_class_logits.float(), dim=-1),
            ],
            dim=-1,
        )
        return self._summarize(combined, output.residue_mask, positions)

    @torch.no_grad()
    def forward(
        self,
        wt_sequences: Sequence[str],
        mutant_sequences: Sequence[str],
        positions: torch.Tensor,
        amp: bool = True,
    ) -> torch.Tensor:
        batch_size = len(wt_sequences)
        both = list(wt_sequences) + list(mutant_sequences)
        features = self.sequence_features(
            both, torch.cat([positions, positions]), amp=amp
        )
        wt, mutant = features[:batch_size], features[batch_size:]
        delta = wt - mutant
        return torch.cat([wt, mutant, delta, delta.abs()], dim=-1)


class PathogenicityEnsemble:
    """Unweighted mean of independently trained PALM pathogenicity heads."""

    def __init__(
        self,
        palm_checkpoint: str | Path = DEFAULT_ANNOTATION_REFERENCE,
        head_checkpoints: Sequence[str | Path] | None = None,
        *,
        device: str | torch.device = "auto",
        local_files_only: bool = False,
        amp: bool = True,
        checkpoint_dir: str | Path | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
    ) -> None:
        if head_checkpoints is None:
            head_checkpoints = DEFAULT_PATHOGENICITY_HEAD_REFERENCES
        if not head_checkpoints:
            raise ValueError("At least one pathogenicity-head checkpoint is required.")
        resolved_heads = [
            resolve_artifact(
                path,
                checkpoint_dir=checkpoint_dir,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
            for path in head_checkpoints
        ]
        first_state = torch.load(
            resolved_heads[0],
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        first_args = dict(first_state["args"])
        self.max_len = int(first_args["max_len"])
        self.local_radius = int(first_args["local_radius"])
        self.summaries = parse_summaries(first_args["summaries"])
        hidden_dims = list(first_args["head_hidden_dims"])
        dropout = float(first_args["dropout"])
        input_layernorm = bool(first_args.get("input_layernorm", True))

        palm, palm_state = load_model(
            palm_checkpoint,
            device=device,
            max_len=self.max_len,
            local_files_only=local_files_only,
            require_epoch8=True,
            checkpoint_dir=checkpoint_dir,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
        for parameter in palm.parameters():
            parameter.requires_grad_(False)
        self.extractor = PairFeatureExtractor(
            palm,
            int(palm_state["num_domain_classes"]),
            int(palm_state["num_site_classes"]),
            self.summaries,
            self.local_radius,
        ).eval()
        self.amp = bool(amp)
        self.heads: list[PathogenicityHead] = []
        self.head_epochs: list[int] = []
        del first_state
        for path in resolved_heads:
            state = torch.load(
                Path(path), map_location="cpu", weights_only=False, mmap=True
            )
            current_args = dict(state["args"])
            configuration_keys = (
                "max_len",
                "local_radius",
                "summaries",
                "head_hidden_dims",
                "dropout",
                "input_layernorm",
            )
            mismatched = [
                key
                for key in configuration_keys
                if current_args.get(key, first_args.get(key))
                != first_args.get(key, current_args.get(key))
            ]
            if mismatched:
                raise ValueError(
                    f"Head {path} differs from the ensemble configuration for "
                    f"{', '.join(mismatched)}."
                )
            if int(state["feature_dim"]) != self.extractor.feature_dim:
                raise ValueError(
                    f"Head {path} feature dimension {state['feature_dim']} does not match "
                    f"the PALM extractor dimension {self.extractor.feature_dim}."
                )
            head = PathogenicityHead(
                self.extractor.feature_dim,
                hidden_dims,
                dropout,
                input_layernorm,
            ).to(self.device)
            head.load_state_dict(state["classifier_state_dict"])
            self.heads.append(head.eval())
            self.head_epochs.append(int(state.get("epoch", -1)))
            del state

    @property
    def device(self) -> torch.device:
        return next(self.extractor.parameters()).device

    @torch.inference_mode()
    def predict_batch(
        self,
        wt_sequences: Sequence[str],
        mutant_sequences: Sequence[str],
        positions: Sequence[int],
        *,
        threshold: float = 0.5,
    ) -> list[dict[str, object]]:
        if not (len(wt_sequences) == len(mutant_sequences) == len(positions)):
            raise ValueError("WT, mutant, and position inputs must have equal lengths.")
        wt_crops: list[str] = []
        mutant_crops: list[str] = []
        crop_positions: list[int] = []
        crop_starts: list[int] = []
        for wt, mutant, position in zip(wt_sequences, mutant_sequences, positions):
            wt = normalize_sequence(wt)
            mutant = normalize_sequence(mutant)
            if len(wt) != len(mutant):
                raise ValueError("WT and mutant sequences must have equal length.")
            wt_crop, crop_position, start = crop_around_position(wt, int(position), self.max_len)
            mutant_crop, mutant_position, mutant_start = crop_around_position(
                mutant, int(position), self.max_len
            )
            if (crop_position, start) != (mutant_position, mutant_start):
                raise RuntimeError("WT and mutant crops diverged.")
            wt_crops.append(wt_crop)
            mutant_crops.append(mutant_crop)
            crop_positions.append(crop_position)
            crop_starts.append(start)
        position_tensor = torch.tensor(crop_positions, device=self.device)
        features = self.extractor(
            wt_crops, mutant_crops, position_tensor, amp=self.amp
        )
        logits = torch.stack([head(features) for head in self.heads], dim=1).float()
        probabilities = torch.sigmoid(logits)
        ensemble = probabilities.mean(dim=1)
        rows: list[dict[str, object]] = []
        for index, probability in enumerate(ensemble.cpu().tolist()):
            row: dict[str, object] = {
                "aa_pos": int(positions[index]),
                "crop_start": int(crop_starts[index] + 1),
                "cropped_aa_pos": int(crop_positions[index]),
                "pathogenic_probability": float(probability),
                "predicted_label": int(probability >= threshold),
                "ensemble_size": len(self.heads),
                "head_probability_std": float(probabilities[index].std(unbiased=False).item()),
            }
            for head_index, head_probability in enumerate(probabilities[index].cpu().tolist(), 1):
                row[f"head_{head_index}_probability"] = float(head_probability)
            rows.append(row)
        return rows

    def predict_one(
        self,
        wt_sequence: str,
        mutant_sequence: str,
        aa_pos: int,
        *,
        threshold: float = 0.5,
    ) -> dict[str, object]:
        return self.predict_batch(
            [wt_sequence], [mutant_sequence], [aa_pos], threshold=threshold
        )[0]
