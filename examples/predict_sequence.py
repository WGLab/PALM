#!/usr/bin/env python3
"""Annotate one protein sequence with the final PALM model."""

from __future__ import annotations

import argparse
from pathlib import Path

from palm import DEFAULT_ANNOTATION_REFERENCE, PALMPredictor


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Return ``(identifier, sequence)`` for every record in a FASTA file."""
    records: list[tuple[str, str]] = []
    name = path.stem
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if chunks:
                records.append((name, "".join(chunks)))
                chunks = []
            name = line[1:].split()[0] if line[1:].split() else path.stem
        else:
            chunks.append(line.strip())
    if chunks:
        records.append((name, "".join(chunks)))
    if not records:
        raise ValueError(f"No sequences found in {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence")
    source.add_argument("--fasta", type=Path, help="FASTA file; all records are annotated.")
    parser.add_argument("--checkpoint", default=DEFAULT_ANNOTATION_REFERENCE)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory holding downloaded weights; overrides the PALM_CHECKPOINT_DIR "
        "environment variable. Files not found there are fetched from Hugging Face.",
    )
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
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        local_files_only=args.local_files_only,
        amp=not args.no_amp,
    )
    if args.sequence is not None:
        records = [(args.output_prefix.name, args.sequence)]
    else:
        records = read_fasta(args.fasta)

    for name, sequence in records:
        result = predictor.annotate(sequence, top_k=args.top_k)
        prefix = (
            args.output_prefix
            if len(records) == 1
            else args.output_prefix.with_name(f"{args.output_prefix.name}_{name}")
        )
        tsv, npz = result.save(prefix)
        print(f"{name} residues={result.predicted_length} tsv={tsv} arrays={npz}")


if __name__ == "__main__":
    main()
