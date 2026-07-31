#!/usr/bin/env python3
"""Zero-shot pathogenicity benchmark for ClinVar-like or PubMind-like tables.

No pathogenicity labels are used to fit PALM or ESM2. Labels are used only to
calculate AUROC and average precision after all scores have been generated.
Input rows must already be mapped to a protein transcript/sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForMaskedLM, AutoTokenizer

from palm.hub import DEFAULT_ANNOTATION_REFERENCE
from palm.model import load_model
from palm.pathogenicity import crop_around_position
from palm.predictor import mutate_sequence, normalize_sequence


HEADS = (
    "domain_occurrence",
    "site_occurrence",
    "domain_class",
    "site_class",
)
SUMMARIES = ("mean", "mutation", "local", "max")


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    standard_deviation = numeric.std(ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - numeric.mean()) / standard_deviation


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.clip(left.astype(np.float64), 1e-12, 1.0)
    right = np.clip(right.astype(np.float64), 1e-12, 1.0)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * np.log(left / middle)).sum(axis=-1)
        + (right * np.log(right / middle)).sum(axis=-1)
    )


def summarize(values: np.ndarray, position: int, radius: int) -> dict[str, float]:
    index = position - 1
    lower, upper = max(0, index - radius), min(len(values), index + radius + 1)
    return {
        "mean": float(values.mean()),
        "mutation": float(values[index]),
        "local": float(values[lower:upper].mean()),
        "max": float(values.max()),
    }


def output_arrays(output, index: int) -> dict[str, np.ndarray]:
    length = int(output.residue_mask[index].sum().item())
    return {
        "embedding": output.residue_embeddings[index, :length].float().cpu().numpy(),
        "domain_occurrence": torch.softmax(
            output.domain_occurrence_logits[index, :length].float(), dim=-1
        ).cpu().numpy(),
        "site_occurrence": torch.softmax(
            output.site_occurrence_logits[index, :length].float(), dim=-1
        ).cpu().numpy(),
        "domain_class": torch.softmax(
            output.domain_class_logits[index, :length].float(), dim=-1
        ).cpu().numpy(),
        "site_class": torch.softmax(
            output.site_class_logits[index, :length].float(), dim=-1
        ).cpu().numpy(),
    }


@torch.inference_mode()
def esm_embeddings(
    model,
    tokenizer,
    sequences: Sequence[str],
    device: torch.device,
    amp: bool,
) -> list[np.ndarray]:
    tokens = tokenizer(
        list(sequences),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(model.config.max_position_embeddings),
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        hidden = model.esm(**tokens).last_hidden_state
    lengths = (tokens["attention_mask"].sum(dim=1) - 2).clamp_min(0)
    return [
        hidden[index, 1 : 1 + int(length)].float().cpu().numpy()
        for index, length in enumerate(lengths)
    ]


@torch.inference_mode()
def masked_llr(
    model,
    tokenizer,
    sequences: Sequence[str],
    positions: Sequence[int],
    amino_acids_wt: Sequence[str],
    amino_acids_mutant: Sequence[str],
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    tokens = tokenizer(
        list(sequences), return_tensors="pt", padding=True, truncation=True
    )
    input_ids = tokens["input_ids"]
    for row, position in enumerate(positions):
        token_position = int(position)
        if token_position >= input_ids.shape[1] - 1:
            raise ValueError(f"Position {position} was truncated in row {row}.")
        input_ids[row, token_position] = tokenizer.mask_token_id
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        logits = model(**tokens).logits.float()
    scores = []
    for row, (position, wt, mutant) in enumerate(
        zip(positions, amino_acids_wt, amino_acids_mutant)
    ):
        log_probabilities = torch.log_softmax(logits[row, int(position)], dim=-1)
        wt_id = tokenizer.convert_tokens_to_ids(str(wt))
        mutant_id = tokenizer.convert_tokens_to_ids(str(mutant))
        scores.append(float((log_probabilities[wt_id] - log_probabilities[mutant_id]).cpu()))
    return np.asarray(scores)


def prepare_table(args: argparse.Namespace) -> pd.DataFrame:
    table = pd.read_csv(args.input, sep="\t", low_memory=False)
    required = {args.wt_column, args.position_column, args.label_column}
    if not (args.mutant_column in table or args.aa_to_column in table):
        required.add(args.aa_to_column)
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Input table is missing columns: {sorted(missing)}")
    rows = []
    for row_index, row in table.iterrows():
        try:
            wt = normalize_sequence(row[args.wt_column])
            position = int(row[args.position_column])
            mutant = (
                normalize_sequence(row[args.mutant_column])
                if args.mutant_column in table and pd.notna(row[args.mutant_column])
                else mutate_sequence(wt, position, str(row[args.aa_to_column]))
            )
            label = int(row[args.label_column])
            if label not in (0, 1) or len(wt) != len(mutant):
                continue
            aa_wt, aa_mutant = wt[position - 1], mutant[position - 1]
            if aa_wt == aa_mutant:
                continue
            rows.append(
                {
                    "input_row": row_index,
                    "wt_sequence": wt,
                    "mutant_sequence": mutant,
                    "aa_pos": position,
                    "aa_wt": aa_wt,
                    "aa_mutant": aa_mutant,
                    "label": label,
                }
            )
        except (TypeError, ValueError, IndexError):
            continue
    if not rows:
        raise ValueError("No usable binary missense rows found.")
    return pd.DataFrame(rows)


def add_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    head_scores = []
    for head in HEADS:
        columns = [f"{head}_js_{summary}" for summary in SUMMARIES]
        score = f"{head}_disruption_z"
        output[score] = pd.concat([zscore(output[column]) for column in columns], axis=1).mean(axis=1)
        head_scores.append(score)
    output["palm_annotation_disruption_z"] = output[head_scores].mean(axis=1)
    for model in ("palm", "esm2"):
        columns = [f"{model}_embedding_l2_{summary}" for summary in SUMMARIES]
        output[f"{model}_embedding_disruption_z"] = pd.concat(
            [zscore(output[column]) for column in columns], axis=1
        ).mean(axis=1)
    output["esm2_llr_z"] = zscore(output["esm2_wt_minus_mutant_llr"])
    output["esm2_llr_plus_palm_annotation"] = (
        output["esm2_llr_z"] + output["palm_annotation_disruption_z"]
    ) / 2
    output["esm2_llr_plus_palm_embedding"] = (
        output["esm2_llr_z"] + output["palm_embedding_disruption_z"]
    ) / 2
    score_columns = [
        "esm2_wt_minus_mutant_llr",
        "esm2_embedding_disruption_z",
        "palm_embedding_disruption_z",
        "palm_annotation_disruption_z",
        *head_scores,
        "esm2_llr_plus_palm_annotation",
        "esm2_llr_plus_palm_embedding",
    ]
    return output, score_columns


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_ANNOTATION_REFERENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wt-column", default="wt_sequence")
    parser.add_argument("--mutant-column", default="mutant_sequence")
    parser.add_argument("--position-column", default="aa_pos")
    parser.add_argument("--aa-to-column", default="aa_to")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--esm-checkpoint", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--local-radius", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare_table(args)
    palm, _ = load_model(
        args.checkpoint,
        device=args.device,
        max_len=args.max_len,
        local_files_only=args.local_files_only,
    )
    device = next(palm.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(
        args.esm_checkpoint, local_files_only=args.local_files_only
    )
    esm = AutoModelForMaskedLM.from_pretrained(
        args.esm_checkpoint, local_files_only=args.local_files_only
    ).to(device).eval()
    amp = not args.no_amp
    feature_rows: list[dict[str, object]] = []
    for start in range(0, len(frame), args.batch_size):
        batch = frame.iloc[start : start + args.batch_size]
        wt_crops, mutant_crops, crop_positions = [], [], []
        for row in batch.itertuples(index=False):
            wt_crop, position, crop_start = crop_around_position(
                row.wt_sequence, row.aa_pos, args.max_len
            )
            mutant_crop, mutant_position, mutant_start = crop_around_position(
                row.mutant_sequence, row.aa_pos, args.max_len
            )
            if (position, crop_start) != (mutant_position, mutant_start):
                raise RuntimeError("WT/mutant crops diverged.")
            wt_crops.append(wt_crop)
            mutant_crops.append(mutant_crop)
            crop_positions.append(position)
        palm_output = palm(wt_crops + mutant_crops, amp=amp)
        esm_output = esm_embeddings(
            esm,
            tokenizer,
            wt_crops + mutant_crops,
            device,
            amp,
        )
        llr = masked_llr(
            esm,
            tokenizer,
            wt_crops,
            crop_positions,
            batch["aa_wt"].tolist(),
            batch["aa_mutant"].tolist(),
            device,
            amp,
        )
        size = len(batch)
        for offset, row in enumerate(batch.itertuples(index=False)):
            wt_palm = output_arrays(palm_output, offset)
            mutant_palm = output_arrays(palm_output, size + offset)
            values: dict[str, object] = {
                "input_row": row.input_row,
                "label": row.label,
                "aa_pos": row.aa_pos,
                "aa_wt": row.aa_wt,
                "aa_mutant": row.aa_mutant,
                "esm2_wt_minus_mutant_llr": llr[offset],
            }
            for head in HEADS:
                summary = summarize(
                    js_divergence(wt_palm[head], mutant_palm[head]),
                    crop_positions[offset],
                    args.local_radius,
                )
                values.update({f"{head}_js_{key}": value for key, value in summary.items()})
            for name, wt_embedding, mutant_embedding in (
                ("palm", wt_palm["embedding"], mutant_palm["embedding"]),
                ("esm2", esm_output[offset], esm_output[size + offset]),
            ):
                residue_l2 = np.linalg.norm(wt_embedding - mutant_embedding, axis=1)
                summary = summarize(residue_l2, crop_positions[offset], args.local_radius)
                values.update({f"{name}_embedding_l2_{key}": value for key, value in summary.items()})
            feature_rows.append(values)
        print(f"scored={min(start + size, len(frame)):,}/{len(frame):,}", flush=True)
    features, scores = add_scores(pd.DataFrame(feature_rows))
    features.to_csv(args.output_dir / "zero_shot_predictions.tsv", sep="\t", index=False)
    metrics = []
    for score in scores:
        usable = features[["label", score]].dropna()
        metrics.append(
            {
                "score": score,
                "n": len(usable),
                "pathogenic": int(usable["label"].sum()),
                "benign": int((usable["label"] == 0).sum()),
                "auroc": roc_auc_score(usable["label"], usable[score]),
                "average_precision": average_precision_score(usable["label"], usable[score]),
            }
        )
    metric_table = pd.DataFrame(metrics).sort_values("auroc", ascending=False)
    metric_table.to_csv(args.output_dir / "zero_shot_metrics.tsv", sep="\t", index=False)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "input": str(args.input),
                "n": len(features),
                "pathogenic": int(features["label"].sum()),
                "benign": int((features["label"] == 0).sum()),
            },
            indent=2,
        )
        + "\n"
    )
    print(metric_table.to_string(index=False))


if __name__ == "__main__":
    main()
