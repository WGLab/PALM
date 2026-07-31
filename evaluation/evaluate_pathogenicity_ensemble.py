#!/usr/bin/env python3
"""Evaluate supervised PALM pathogenicity heads on a prepared missense TSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from palm.pathogenicity import PathogenicityEnsemble
from palm.predictor import mutate_sequence


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--palm-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wt-column", default="wt_sequence")
    parser.add_argument("--mutant-column", default="mutant_sequence")
    parser.add_argument("--position-column", default="aa_pos")
    parser.add_argument("--aa-to-column", default="aa_to")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    table = pd.read_csv(args.input, sep="\t", low_memory=False)
    predictor = PathogenicityEnsemble(
        args.palm_checkpoint,
        args.head_checkpoints,
        device=args.device,
        local_files_only=args.local_files_only,
        amp=not args.no_amp,
    )
    output_rows = []
    for start in range(0, len(table), args.batch_size):
        batch = table.iloc[start : start + args.batch_size]
        wt_sequences, mutant_sequences, positions, source_indices = [], [], [], []
        for source_index, row in batch.iterrows():
            try:
                wt = str(row[args.wt_column])
                position = int(row[args.position_column])
                mutant = (
                    str(row[args.mutant_column])
                    if args.mutant_column in table and pd.notna(row[args.mutant_column])
                    else mutate_sequence(wt, position, str(row[args.aa_to_column]))
                )
                label = int(row[args.label_column])
                if label not in (0, 1):
                    continue
            except (TypeError, ValueError):
                continue
            wt_sequences.append(wt)
            mutant_sequences.append(mutant)
            positions.append(position)
            source_indices.append((source_index, label))
        if not wt_sequences:
            continue
        predictions = predictor.predict_batch(wt_sequences, mutant_sequences, positions)
        for (source_index, label), prediction in zip(source_indices, predictions):
            output_rows.append({"input_row": source_index, "label": label, **prediction})
        print(f"processed={min(start + len(batch), len(table)):,}/{len(table):,}", flush=True)
    output = pd.DataFrame(output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)
    print(
        f"n={len(output)} AUROC={roc_auc_score(output['label'], output['pathogenic_probability']):.6f} "
        f"PR-AUC={average_precision_score(output['label'], output['pathogenic_probability']):.6f}"
    )


if __name__ == "__main__":
    main()

