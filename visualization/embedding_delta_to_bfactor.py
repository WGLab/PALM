#!/usr/bin/env python3
"""Write a per-residue PALM embedding difference into a PDB B-factor field."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--delta-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--position-column", default="position")
    parser.add_argument("--value-column", default="embedding_l2")
    parser.add_argument("--pdb-offset", type=int, default=0, help="PALM position = PDB residue number + offset")
    args = parser.parse_args()

    delta = pd.read_csv(args.delta_tsv, sep="\t")
    values = dict(zip(delta[args.position_column].astype(int), delta[args.value_column].astype(float)))
    lines = []
    matched = set()
    for line in args.pdb.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66 and line[21].strip() == args.chain:
            try:
                pdb_position = int(line[22:26])
                value = values[pdb_position + args.pdb_offset]
            except (ValueError, KeyError):
                pass
            else:
                matched.add(pdb_position + args.pdb_offset)
                line = line[:60] + f"{value:6.2f}" + line[66:]
        lines.append(line)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(lines))
    print(f"wrote={args.output} matched_residues={len(matched)}")


if __name__ == "__main__":
    main()

