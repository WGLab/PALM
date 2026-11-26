#!/usr/bin/env python3
"""Regenerate palm/interpro_classes.tsv from a released checkpoint.

The table ships with the package so that `PALMPredictor` can name classes
without being handed an InterPro file. One row per output class of each head,
with the model's own 0-based index, so the indices are authoritative rather
than re-derived from a vocabulary file whose row order might differ.

The six-head checkpoint has three class heads; the four-head one had two, with
Repeat and PTM sitting unused inside the site vocabulary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

HEAD_KEYS = {"domain": "ipr2domain", "site": "ipr2site", "repeat": "ipr2repeat"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--ipr-tsv", type=Path, required=True,
                    help="InterPro entry table with ipr, type and name columns.")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parents[2] / "palm/interpro_classes.tsv")
    args = ap.parse_args()

    entries = pd.read_csv(args.ipr_tsv, sep="\t")
    types = dict(zip(entries["ipr"].astype(str), entries["type"].astype(str)))
    names = dict(zip(entries["ipr"].astype(str), entries["name"].astype(str)))

    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False, mmap=True)
    rows = []
    for head, key in HEAD_KEYS.items():
        if key not in checkpoint:
            raise SystemExit(f"Checkpoint has no {key}; it is not a six-head model.")
        for ipr, index in checkpoint[key].items():
            ipr = str(ipr)
            rows.append({
                "head": head,
                "class_index": int(index),
                "interpro_id": ipr,
                "interpro_type": types.get(ipr, "Unknown"),
                "name": names.get(ipr, ipr),
            })
    table = pd.DataFrame(rows).sort_values(["head", "class_index"])
    table.to_csv(args.output, sep="\t", index=False)

    print(table.groupby("head").size().to_string())
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
