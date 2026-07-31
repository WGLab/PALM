#!/usr/bin/env python3
"""Evaluate functional organization of PALM embeddings with Reactome kNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoModel, AutoTokenizer

from palm.model import load_model
from palm.predictor import normalize_sequence


def majority_vote(neighbor_labels: np.ndarray) -> np.ndarray:
    predictions = np.empty(len(neighbor_labels), dtype=np.int64)
    for index, labels in enumerate(neighbor_labels):
        counts = np.bincount(labels)
        predictions[index] = np.flatnonzero(counts == counts.max())[0]
    return predictions


def knn_metrics(
    matrix: np.ndarray,
    encoded_labels: np.ndarray,
    ks: list[int],
    model_name: str,
) -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray]]]:
    scaled = np.nan_to_num(StandardScaler().fit_transform(matrix))
    maximum = min(max(ks), len(matrix) - 1)
    neighbors = NearestNeighbors(
        n_neighbors=maximum + 1, metric="cosine", algorithm="brute", n_jobs=-1
    ).fit(scaled)
    distances, indices = neighbors.kneighbors(scaled)
    distances, indices = distances[:, 1:], indices[:, 1:]
    rows = []
    per_protein = {}
    for requested_k in ks:
        k = min(requested_k, maximum)
        neighbor_labels = encoded_labels[indices[:, :k]]
        purity = (neighbor_labels == encoded_labels[:, None]).mean(axis=1)
        predictions = majority_vote(neighbor_labels)
        per_protein[requested_k] = (purity, predictions == encoded_labels)
        rows.append(
            {
                "model": model_name,
                "k": requested_k,
                "effective_k": k,
                "n_proteins": len(matrix),
                "n_labels": len(np.unique(encoded_labels)),
                "assignment_accuracy": accuracy_score(encoded_labels, predictions),
                "balanced_accuracy": balanced_accuracy_score(encoded_labels, predictions),
                "macro_f1": f1_score(encoded_labels, predictions, average="macro", zero_division=0),
                "neighbor_purity": purity.mean(),
                "mean_neighbor_cosine_distance": distances[:, :k].mean(),
            }
        )
    return pd.DataFrame(rows), per_protein


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--label-column", default="reactome_top_level")
    parser.add_argument("--id-column", default="protein_id")
    parser.add_argument("--neighbors", default="5,10,20,50")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--include-esm2", action="store_true")
    parser.add_argument("--esm-checkpoint", default="facebook/esm2_t30_150M_UR50D")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.input, sep="\t", low_memory=False)
    required = {args.sequence_column, args.label_column}
    if missing := required - set(table.columns):
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    table = table.dropna(subset=list(required)).copy()
    table[args.sequence_column] = table[args.sequence_column].map(normalize_sequence)
    table = table.drop_duplicates(args.id_column if args.id_column in table else args.sequence_column)
    table = table.reset_index(drop=True)
    model, _ = load_model(
        args.checkpoint,
        device=args.device,
        max_len=args.max_len,
        local_files_only=args.local_files_only,
    )
    embeddings = []
    for start in range(0, len(table), args.batch_size):
        sequences = table[args.sequence_column].iloc[start : start + args.batch_size].tolist()
        encoded, mask = model.encode(sequences, amp=not args.no_amp)
        average = (encoded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        embeddings.append(average.float().cpu().numpy())
        print(f"embedded={min(start + len(sequences), len(table)):,}/{len(table):,}", flush=True)
    palm_matrix = np.concatenate(embeddings)
    np.save(args.output_dir / "palm_average_embeddings.npy", palm_matrix)
    metadata_columns = [column for column in (args.id_column, args.label_column) if column in table]
    table[metadata_columns].to_csv(
        args.output_dir / "embedding_metadata.tsv", sep="\t", index=False
    )

    ks = sorted({int(value) for value in args.neighbors.split(",")})
    encoded_labels = LabelEncoder().fit_transform(table[args.label_column].astype(str))
    per_protein = table[metadata_columns].copy()
    metrics, protein_values = knn_metrics(
        palm_matrix, encoded_labels, ks, "PALM_epoch8"
    )
    for k, (purity, correct) in protein_values.items():
        per_protein[f"palm_neighbor_purity_k{k}"] = purity
        per_protein[f"palm_correct_k{k}"] = correct
    if args.include_esm2:
        tokenizer = AutoTokenizer.from_pretrained(
            args.esm_checkpoint, local_files_only=args.local_files_only
        )
        esm = AutoModel.from_pretrained(
            args.esm_checkpoint, local_files_only=args.local_files_only
        ).to(next(model.parameters()).device).eval()
        esm_rows = []
        for start in range(0, len(table), args.batch_size):
            sequences = table[args.sequence_column].iloc[start : start + args.batch_size].tolist()
            tokens = tokenizer(
                sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024
            )
            tokens = {key: value.to(next(esm.parameters()).device) for key, value in tokens.items()}
            with torch.autocast(
                device_type=next(esm.parameters()).device.type,
                enabled=not args.no_amp and next(esm.parameters()).device.type == "cuda",
            ):
                hidden = esm(**tokens).last_hidden_state
            lengths = (tokens["attention_mask"].sum(dim=1) - 2).clamp_min(0)
            for index, length in enumerate(lengths):
                esm_rows.append(hidden[index, 1 : 1 + int(length)].float().mean(dim=0).cpu().numpy())
        esm_matrix = np.vstack(esm_rows)
        np.save(args.output_dir / "esm2_average_embeddings.npy", esm_matrix)
        esm_metrics, protein_values = knn_metrics(
            esm_matrix, encoded_labels, ks, "Frozen_ESM2_150M"
        )
        metrics = pd.concat([metrics, esm_metrics], ignore_index=True)
        for k, (purity, correct) in protein_values.items():
            per_protein[f"esm2_neighbor_purity_k{k}"] = purity
            per_protein[f"esm2_correct_k{k}"] = correct
    metrics.to_csv(args.output_dir / "reactome_knn_metrics.tsv", sep="\t", index=False)
    per_protein.to_csv(args.output_dir / "reactome_knn_per_protein.tsv", sep="\t", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
