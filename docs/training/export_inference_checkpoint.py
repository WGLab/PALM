#!/usr/bin/env python3
"""Remove optimizer/scaler state from a PALM checkpoint for distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Refusing to overwrite the training checkpoint in place.")
    checkpoint = torch.load(
        args.input, map_location="cpu", weights_only=False, mmap=True
    )
    if int(checkpoint.get("epoch", -1)) != 8:
        raise ValueError(f"Expected epoch 8, found {checkpoint.get('epoch')}.")
    if checkpoint.get("specific_class_loss") != "uniform_soft_target_cross_entropy":
        raise ValueError("Input is not the final uniform soft-target checkpoint.")
    excluded = {"optimizer_state_dict", "scaler_state_dict"}
    inference_checkpoint = {
        key: value for key, value in checkpoint.items() if key not in excluded
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_checkpoint, args.output)
    print(
        f"wrote={args.output} bytes={args.output.stat().st_size} "
        f"removed={','.join(sorted(excluded & set(checkpoint)))}"
    )


if __name__ == "__main__":
    main()

