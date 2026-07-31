#!/usr/bin/env python3
"""Five-fold, tile-grouped FlexiNR mutational-screen benchmark.

The input is a prepared paired WT/mutant TSV. Cross-validation is performed
within each screen with ``tile_id`` as the group, preventing variants from the
same tile from appearing in both training and test folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForMaskedLM, AutoTokenizer

from palm.hub import DEFAULT_ANNOTATION_REFERENCE
from palm.model import load_model
from palm.predictor import normalize_sequence


def amino_acid_differences(wt: str, mutant: str) -> list[int]:
    return [index for index, pair in enumerate(zip(wt, mutant)) if pair[0] != pair[1]]


def centered_slice(start: int, end: int, length: int, window: int) -> slice:
    center = round((start + end - 1) / 2)
    lower = max(0, min(center - window // 2, max(0, length - window)))
    return slice(lower, min(length, lower + window))


def pad_flat(values: np.ndarray, length: int) -> np.ndarray:
    output = np.zeros((length, values.shape[1]), dtype=np.float32)
    output[: min(length, len(values))] = values[:length]
    return output.reshape(-1)


def mean_or_zero(values: np.ndarray, dimension: int) -> np.ndarray:
    return values.mean(axis=0).astype(np.float32) if len(values) else np.zeros(dimension, np.float32)


def embedding_features(
    wt_embedding: np.ndarray,
    mutant_embedding: np.ndarray,
    wt_sequence: str,
    mutant_sequence: str,
    motif: slice,
    local_window: int,
    max_changed_positions: int,
    max_motif_positions: int,
) -> dict[str, np.ndarray]:
    length = min(len(wt_embedding), len(mutant_embedding))
    delta = (mutant_embedding[:length] - wt_embedding[:length]).astype(np.float32)
    dimension = delta.shape[1]
    changed_indices = [
        index for index in amino_acid_differences(wt_sequence[:length], mutant_sequence[:length])
        if index < length
    ]
    start = max(0, min(length, motif.start or 0))
    end = max(start, min(length, motif.stop or length))
    changed = delta[changed_indices]
    motif_values = delta[start:end]
    local = delta[centered_slice(start, end, length, local_window)]
    means = [
        mean_or_zero(changed, dimension),
        mean_or_zero(motif_values, dimension),
        mean_or_zero(local, dimension),
        mean_or_zero(delta, dimension),
    ]
    return {
        "changed_mean_delta": means[0],
        "changed_flat_delta": pad_flat(changed, max_changed_positions),
        "motif_mean_delta": means[1],
        "motif_flat_delta": pad_flat(motif_values, max_motif_positions),
        "local_window_mean_delta": means[2],
        "local_window_flat_delta": pad_flat(local, local_window),
        "whole_mean_delta": means[3],
        "combined_mean_delta": np.concatenate(means).astype(np.float32),
    }


@torch.inference_mode()
def palm_outputs(model, sequences: Sequence[str], batch_size: int, amp: bool) -> dict[str, dict[str, np.ndarray]]:
    results = {}
    for start in range(0, len(sequences), batch_size):
        batch = list(sequences[start : start + batch_size])
        output = model(batch, amp=amp)
        for index, sequence in enumerate(batch):
            length = int(output.residue_mask[index].sum())
            results[sequence] = {
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
    return results


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.clip(left.astype(np.float64), 1e-12, 1)
    right = np.clip(right.astype(np.float64), 1e-12, 1)
    middle = (left + right) / 2
    return 0.5 * (
        (left * np.log(left / middle)).sum(axis=-1)
        + (right * np.log(right / middle)).sum(axis=-1)
    )


def annotation_matrices(
    frame: pd.DataFrame, outputs: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    head_names = (
        "domain_occurrence",
        "site_occurrence",
        "domain_class",
        "site_class",
    )
    rows = []
    for row in frame.itertuples(index=False):
        wt, mutant = outputs[row.wt_sequence], outputs[row.mutant_sequence]
        motif_start = int(getattr(row, "motif_relative_start", 1)) - 1
        motif_end = int(getattr(row, "motif_relative_end", len(row.wt_sequence)))
        changed = amino_acid_differences(row.wt_sequence, row.mutant_sequence)
        values = []
        for head in head_names:
            divergence = js_divergence(wt[head], mutant[head])
            motif = divergence[max(0, motif_start) : min(len(divergence), motif_end)]
            changed_values = divergence[[index for index in changed if index < len(divergence)]]
            values.extend(
                [
                    float(divergence.mean()),
                    float(divergence.max()),
                    float(motif.mean()) if len(motif) else 0.0,
                    float(changed_values.mean()) if len(changed_values) else 0.0,
                ]
            )
        rows.append(values)
    matrix = np.asarray(rows, dtype=np.float32)
    return {
        "annotation_all": matrix,
        "annotation_occurrence": matrix[:, :8],
        "annotation_classification": matrix[:, 8:],
    }


@torch.inference_mode()
def esm_embeddings(
    model, tokenizer, sequences: Sequence[str], batch_size: int, device: torch.device, amp: bool
) -> dict[str, np.ndarray]:
    results = {}
    for start in range(0, len(sequences), batch_size):
        batch = list(sequences[start : start + batch_size])
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            hidden = model(**tokens).last_hidden_state
        lengths = (tokens["attention_mask"].sum(dim=1) - 2).clamp_min(0)
        for index, sequence in enumerate(batch):
            results[sequence] = hidden[index, 1 : 1 + int(lengths[index])].float().cpu().numpy()
    return results


@torch.inference_mode()
def esm_llr_matrix(
    frame: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    amp: bool,
) -> dict[str, np.ndarray]:
    rows = []
    for row_index, row in enumerate(frame.itertuples(index=False), 1):
        differences = amino_acid_differences(row.wt_sequence, row.mutant_sequence)
        scores = []
        for position in differences:
            tokens = tokenizer(row.wt_sequence, return_tensors="pt", truncation=True, max_length=1024)
            if position + 1 >= tokens["input_ids"].shape[1] - 1:
                continue
            tokens["input_ids"][0, position + 1] = tokenizer.mask_token_id
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(**tokens).logits[0, position + 1].float()
            log_probability = torch.log_softmax(logits, dim=-1)
            wt_id = tokenizer.convert_tokens_to_ids(row.wt_sequence[position])
            mutant_id = tokenizer.convert_tokens_to_ids(row.mutant_sequence[position])
            scores.append(float((log_probability[wt_id] - log_probability[mutant_id]).cpu()))
        values = np.asarray(scores, dtype=np.float32)
        rows.append(
            [
                float(values.sum()) if len(values) else 0.0,
                float(values.mean()) if len(values) else 0.0,
                float(values.min()) if len(values) else 0.0,
                float(values.max()) if len(values) else 0.0,
            ]
        )
        if row_index % 100 == 0:
            print(f"LLR scored {row_index}/{len(frame)}", flush=True)
    matrix = np.asarray(rows, dtype=np.float32)
    return {"llr_all": matrix, "llr_sum": matrix[:, :1]}


def build_matrices(
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    local_window: int,
    max_changed: int,
    max_motif: int,
) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {}
    for row in frame.itertuples(index=False):
        motif_start = int(getattr(row, "motif_relative_start", 1)) - 1
        motif_end = int(getattr(row, "motif_relative_end", len(row.wt_sequence)))
        features = embedding_features(
            embeddings[row.wt_sequence],
            embeddings[row.mutant_sequence],
            row.wt_sequence,
            row.mutant_sequence,
            slice(motif_start, motif_end),
            local_window,
            max_changed,
            max_motif,
        )
        for name, vector in features.items():
            values.setdefault(name, []).append(vector)
    return {name: np.vstack(rows) for name, rows in values.items()}


def cross_validated_predictions(
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    folds: int,
    classification: bool,
    ridge_alpha: float,
    logistic_c: float,
) -> np.ndarray:
    predictions = np.full(len(target), np.nan)
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    for train, test in splitter.split(matrix, target, groups):
        if classification:
            if len(np.unique(target[train])) < 2:
                continue
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=logistic_c,
                    class_weight="balanced",
                    max_iter=3000,
                ),
            )
            estimator.fit(matrix[train], target[train])
            predictions[test] = estimator.predict_proba(matrix[test])[:, 1]
        else:
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
            estimator.fit(matrix[train], target[train])
            predictions[test] = estimator.predict(matrix[test])
    return predictions


def evaluate(
    frame: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    model_name: str,
    folds: int,
    ridge_alpha: float,
    logistic_c: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows, prediction_rows = [], []
    tasks = [
        ("delta_vs_wt", False),
        ("abs_delta_vs_wt", False),
        ("hit_changed", True),
        ("loss_of_hit", True),
    ]
    for screen, screen_frame in frame.groupby("screen", sort=True):
        indices = screen_frame.index.to_numpy()
        groups = screen_frame["tile_id"].astype(str).to_numpy()
        for feature_name, all_values in matrices.items():
            matrix = all_values[indices]
            for task, classification in tasks:
                if task not in screen_frame:
                    continue
                target = screen_frame[task].to_numpy(dtype=float if not classification else int)
                predictions = cross_validated_predictions(
                    matrix,
                    target,
                    groups,
                    folds,
                    classification,
                    ridge_alpha,
                    logistic_c,
                )
                valid = np.isfinite(predictions)
                metrics = {
                    "screen": screen,
                    "model": model_name,
                    "feature_set": feature_name,
                    "task": task,
                    "n": int(valid.sum()),
                    "spearman": np.nan,
                    "auroc": np.nan,
                    "average_precision": np.nan,
                }
                if classification and valid.sum() and len(np.unique(target[valid])) > 1:
                    metrics["auroc"] = roc_auc_score(target[valid], predictions[valid])
                    metrics["average_precision"] = average_precision_score(target[valid], predictions[valid])
                elif not classification and valid.sum() >= 3:
                    metrics["spearman"] = spearmanr(target[valid], predictions[valid]).statistic
                metric_rows.append(metrics)
                for local_index, global_index in enumerate(indices):
                    prediction_rows.append(
                        {
                            "row_index": int(global_index),
                            "screen": screen,
                            "tile_id": frame.loc[global_index, "tile_id"],
                            "model": model_name,
                            "feature_set": feature_name,
                            "task": task,
                            "truth": target[local_index],
                            "prediction": predictions[local_index],
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_ANNOTATION_REFERENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--esm-checkpoint", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--include-esm2", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-window", type=int, default=21)
    parser.add_argument("--max-changed-positions", type=int, default=10)
    parser.add_argument("--max-motif-positions", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input, sep="\t", low_memory=False)
    required = {"screen", "tile_id", "wt_sequence", "mutant_sequence"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    frame["wt_sequence"] = frame["wt_sequence"].map(normalize_sequence)
    frame["mutant_sequence"] = frame["mutant_sequence"].map(normalize_sequence)
    if "abs_delta_vs_wt" not in frame and "delta_vs_wt" in frame:
        frame["abs_delta_vs_wt"] = frame["delta_vs_wt"].abs()
    frame = frame.reset_index(drop=True)
    sequences = sorted(set(frame["wt_sequence"]) | set(frame["mutant_sequence"]))
    palm, _ = load_model(
        args.checkpoint,
        device=args.device,
        max_len=1024,
        local_files_only=args.local_files_only,
    )
    device = next(palm.parameters()).device
    outputs = palm_outputs(palm, sequences, args.batch_size, not args.no_amp)
    embeddings = {sequence: values["embedding"] for sequence, values in outputs.items()}
    palm_matrices = build_matrices(
        frame,
        embeddings,
        args.local_window,
        args.max_changed_positions,
        args.max_motif_positions,
    )
    all_metrics, all_predictions = list(
        evaluate(
            frame,
            palm_matrices,
            "PALM_epoch8_embedding",
            args.folds,
            args.ridge_alpha,
            args.logistic_c,
        )
    )
    metrics, predictions = evaluate(
        frame,
        annotation_matrices(frame, outputs),
        "PALM_epoch8_annotation",
        args.folds,
        args.ridge_alpha,
        args.logistic_c,
    )
    all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)
    all_predictions = pd.concat([all_predictions, predictions], ignore_index=True)
    if args.include_esm2:
        tokenizer = AutoTokenizer.from_pretrained(
            args.esm_checkpoint, local_files_only=args.local_files_only
        )
        esm = AutoModelForMaskedLM.from_pretrained(
            args.esm_checkpoint, local_files_only=args.local_files_only
        ).to(device).eval()
        embeddings = esm_embeddings(
            esm.esm, tokenizer, sequences, args.batch_size, device, not args.no_amp
        )
        esm_matrices = build_matrices(
            frame,
            embeddings,
            args.local_window,
            args.max_changed_positions,
            args.max_motif_positions,
        )
        metrics, predictions = evaluate(
            frame,
            esm_matrices,
            "ESM2_embedding",
            args.folds,
            args.ridge_alpha,
            args.logistic_c,
        )
        all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)
        all_predictions = pd.concat([all_predictions, predictions], ignore_index=True)
        metrics, predictions = evaluate(
            frame,
            esm_llr_matrix(frame, esm, tokenizer, device, not args.no_amp),
            "ESM2_LLR",
            args.folds,
            args.ridge_alpha,
            args.logistic_c,
        )
        all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)
        all_predictions = pd.concat([all_predictions, predictions], ignore_index=True)
        shared_features = sorted(set(palm_matrices) & set(esm_matrices))
        concatenated = {
            name: np.concatenate([palm_matrices[name], esm_matrices[name]], axis=1)
            for name in shared_features
        }
        metrics, predictions = evaluate(
            frame,
            concatenated,
            "PALM_epoch8_plus_ESM2_embedding",
            args.folds,
            args.ridge_alpha,
            args.logistic_c,
        )
        all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)
        all_predictions = pd.concat([all_predictions, predictions], ignore_index=True)
    all_metrics.to_csv(args.output_dir / "flexinr_metrics.tsv", sep="\t", index=False)
    all_predictions.to_csv(
        args.output_dir / "flexinr_predictions.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    print(all_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
