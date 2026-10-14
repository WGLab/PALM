#!/usr/bin/env python3
"""Evaluate all four PALM annotation heads on FASTA shards."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from palm.hub import DEFAULT_ANNOTATION_REFERENCE
from palm.model import load_model

# The training code lives beside this file under docs/, which is not an
# installed package; add docs/ to the path so `training` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train_soft_target import (
    FastaShardDataset,
    collate,
    load_label_maps,
    match_length,
)


def binary_counts(logits: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    truth = match_length(truth, logits.shape[1]).to(logits.device)
    valid = truth.ne(-100)
    predicted = logits.argmax(dim=-1).ne(0)
    positive = truth.ne(0)
    return np.asarray(
        [
            int((predicted & positive & valid).sum()),
            int((predicted & ~positive & valid).sum()),
            int((~predicted & positive & valid).sum()),
            int(valid.sum()),
        ],
        dtype=np.int64,
    )


def relaxed_class_counts(logits: torch.Tensor, sets: torch.Tensor) -> np.ndarray:
    sets = match_length(sets, logits.shape[1]).to(logits.device)
    valid_labels = sets.ge(0)
    annotated = valid_labels.any(dim=-1)
    predicted = logits.argmax(dim=-1).unsqueeze(-1)
    correct = ((predicted == sets) & valid_labels).any(dim=-1) & annotated
    return np.asarray([int(correct.sum()), int(annotated.sum())], dtype=np.int64)


def prf(values: np.ndarray) -> dict[str, float]:
    tp, fp, fn, residues = [int(value) for value in values]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "evaluated_residues": residues,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_ANNOTATION_REFERENCE)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-dir", type=Path)
    source.add_argument(
        "--pid-groups-dir",
        type=Path,
        help="Directory containing pid_* subdirectories of FASTA shards.",
    )
    parser.add_argument(
        "--ipr-tsv",
        type=Path,
        help="Optional InterPro table, checked for coverage of the checkpoint classes.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--keep-unannotated", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_files(model, checkpoint, domain_map, site_map, files, args, group: str) -> dict:
    if not files:
        raise FileNotFoundError(f"No .fasta files for group {group}")
    dataset = FastaShardDataset(
        files,
        domain_map,
        site_map,
        int(args.max_len or checkpoint["max_len"]),
        not args.keep_unannotated,
        False,
        1234,
        0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate,
        persistent_workers=args.workers > 0,
    )
    domain_occurrence = np.zeros(4, dtype=np.int64)
    site_occurrence = np.zeros(4, dtype=np.int64)
    domain_class = np.zeros(2, dtype=np.int64)
    site_class = np.zeros(2, dtype=np.int64)
    proteins = 0
    for index, batch in enumerate(loader, 1):
        output = model(batch["sequences"], amp=not args.no_amp)
        proteins += len(batch["sequences"])
        domain_occurrence += binary_counts(
            output.domain_occurrence_logits, batch["domain_occurrence"]
        )
        site_occurrence += binary_counts(
            output.site_occurrence_logits, batch["site_occurrence"]
        )
        domain_class += relaxed_class_counts(
            output.domain_class_logits, batch["domain_sets"]
        )
        site_class += relaxed_class_counts(
            output.site_class_logits, batch["site_sets"]
        )
        if index % 1000 == 0:
            print(f"group={group} evaluated_proteins={proteins:,}", flush=True)
    return {
        "group": group,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "evaluated_proteins": proteins,
        "domain_occurrence": prf(domain_occurrence),
        "site_occurrence": prf(site_occurrence),
        "domain_class": {
            "relaxed_accuracy": float(domain_class[0] / max(domain_class[1], 1)),
            "correct_annotated_residues": int(domain_class[0]),
            "annotated_residues": int(domain_class[1]),
        },
        "site_class": {
            "relaxed_accuracy": float(site_class[0] / max(site_class[1], 1)),
            "correct_annotated_residues": int(site_class[0]),
            "annotated_residues": int(site_class[1]),
        },
    }


@torch.inference_mode()
def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(
        args.checkpoint,
        device=args.device,
        max_len=args.max_len,
        local_files_only=args.local_files_only,
    )
    # Class indices must come from the checkpoint. Rebuilding them from the
    # InterPro TSV would make them depend on that file's row order, which
    # silently yields wrong per-class metrics whenever the order differs from
    # the one used during training.
    domain_map = {str(ipr): int(i) for ipr, i in dict(checkpoint["ipr2domain"]).items()}
    site_map = {str(ipr): int(i) for ipr, i in dict(checkpoint["ipr2site"]).items()}
    if len(domain_map) != int(checkpoint["num_domain_classes"]):
        raise ValueError("Domain class map does not match checkpoint.")
    if len(site_map) != int(checkpoint["num_site_classes"]):
        raise ValueError("Site class map does not match checkpoint.")
    if args.ipr_tsv is not None:
        tsv_domain, tsv_site = load_label_maps(args.ipr_tsv)
        absent = (set(domain_map) - set(tsv_domain)) | (set(site_map) - set(tsv_site))
        if absent:
            raise ValueError(
                f"{args.ipr_tsv} is missing {len(absent)} InterPro entries present in "
                "the checkpoint; it is not the table this model was trained with."
            )
    if args.test_dir is not None:
        targets = [("all", sorted(glob.glob(str(args.test_dir / "*.fasta"))))]
    else:
        targets = [
            (directory.name, sorted(glob.glob(str(directory / "*.fasta"))))
            for directory in sorted(args.pid_groups_dir.glob("pid_*"))
            if directory.is_dir()
        ]
        if not targets:
            raise FileNotFoundError(f"No pid_* directories under {args.pid_groups_dir}")
    results = [
        evaluate_files(model, checkpoint, domain_map, site_map, files, args, group)
        for group, files in targets
    ]
    payload = results[0] if len(results) == 1 else {"groups": results}
    (args.output_dir / "annotation_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    rows = []
    for metrics in results:
        for head in ("domain_occurrence", "site_occurrence", "domain_class", "site_class"):
            rows.append({"group": metrics["group"], "head": head, **metrics[head]})
    pd.DataFrame(rows).to_csv(
        args.output_dir / "annotation_metrics.tsv", sep="\t", index=False
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
