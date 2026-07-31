#!/usr/bin/env python3
"""Calculate zero-shot PALM disruption and optional supervised pathogenicity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palm import DEFAULT_ANNOTATION_REFERENCE, PALMPredictor, PathogenicityEnsemble, mutate_sequence


def read_fasta(path: Path) -> str:
    return "".join(
        line.strip() for line in path.read_text().splitlines() if not line.startswith(">")
    )


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--mutant-aa", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_ANNOTATION_REFERENCE)
    parser.add_argument("--head-checkpoints", nargs="*")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    wt = read_fasta(args.fasta)
    mutant = mutate_sequence(wt, args.position, args.mutant_aa)
    predictor = PALMPredictor(
        args.checkpoint,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    comparison = predictor.compare_variant(wt, mutant, args.position)
    comparison["per_residue"].to_csv(
        args.output_prefix.with_suffix(".per_residue.tsv"), sep="\t", index=False
    )
    summary = {
        key: value
        for key, value in comparison.items()
        if key not in {"per_residue", "wt", "mutant"}
    }
    if args.head_checkpoints:
        ensemble = PathogenicityEnsemble(
            args.checkpoint,
            args.head_checkpoints,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        summary["supervised_pathogenicity"] = ensemble.predict_one(
            wt, mutant, args.position
        )
    args.output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
