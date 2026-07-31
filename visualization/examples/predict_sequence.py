#!/usr/bin/env python3
"""Annotate one protein sequence with the final PALM model."""

from __future__ import annotations

import argparse
from pathlib import Path

from palm import DEFAULT_ANNOTATION_REFERENCE, PALMPredictor


def read_fasta(path: Path) -> str:
    return "".join(
        line.strip() for line in path.read_text().splitlines() if not line.startswith(">")
    )


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence")
    source.add_argument("--fasta", type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_ANNOTATION_REFERENCE)
    parser.add_argument("--ipr-tsv", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    predictor = PALMPredictor(
        args.checkpoint,
        ipr_tsv=args.ipr_tsv,
        device=args.device,
        local_files_only=args.local_files_only,
        amp=not args.no_amp,
    )
    sequence = args.sequence if args.sequence is not None else read_fasta(args.fasta)
    result = predictor.annotate(sequence, top_k=args.top_k)
    tsv, npz = result.save(args.output_prefix)
    print(f"predicted_residues={result.predicted_length} tsv={tsv} arrays={npz}")


if __name__ == "__main__":
    main()
