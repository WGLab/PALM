#!/usr/bin/env python3
"""Repackage the training-run pathogenicity heads into publishable checkpoints.

The fold checkpoints written by the cross-validation run carry only weights and
metrics. The released loader (`palm.pathogenicity.PathogenicityEnsemble`) needs
each file to be self-contained: it reads an ``args`` block to rebuild the head
and to check that every fold in an ensemble shares one feature configuration.

The configuration below is not guessed. It is the default set of
`finetune_clinvar_palm6_heads_pathogenicity.py`, which the fold array ran
without overrides, and it is verified against the weights themselves:
LayerNorm(329,968) -> Linear(329,968, 256) -> Linear(256, 64) -> Linear(64, 1),
and 329,968 = 4 blocks x 4 summaries x (1,280 + 3 + 17,951 + 3 + 991 + 3 + 392).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

HEAD_ARGS = {
    "max_len": 1024,
    "local_radius": 5,
    "summaries": "mean,max,mutation,local",
    "head_hidden_dims": [256, 64],
    "dropout": 0.35,
    "input_layernorm": True,
}
EXPECTED_FEATURE_DIM = 329_968
EXPECTED_BASE = (
    "overlap_aware_multi_task/checkpoints/overlap_aware_repeat_150m_27658822/epoch_3.pt"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold-dir", type=Path, required=True,
                    help="Directory holding fold_1/ ... fold_5/")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    epochs = []
    for fold in range(1, args.n_folds + 1):
        source = args.fold_dir / f"fold_{fold}" / "best_pathogenicity_head.pt"
        state = torch.load(source, map_location="cpu", weights_only=False)

        feature_dim = int(state["feature_dim"])
        if feature_dim != EXPECTED_FEATURE_DIM:
            raise SystemExit(f"fold {fold}: feature_dim {feature_dim:,}, "
                             f"expected {EXPECTED_FEATURE_DIM:,}")
        if EXPECTED_BASE not in str(state.get("base_soft_target_ckpt", "")):
            raise SystemExit(f"fold {fold}: base checkpoint is "
                             f"{state.get('base_soft_target_ckpt')!r}, not the epoch-3 model")
        weights = state["classifier_state_dict"]
        if tuple(weights["net.0.weight"].shape) != (EXPECTED_FEATURE_DIM,):
            raise SystemExit(f"fold {fold}: input LayerNorm does not match feature_dim")
        if tuple(weights["net.1.weight"].shape) != (HEAD_ARGS["head_hidden_dims"][0],
                                                    EXPECTED_FEATURE_DIM):
            raise SystemExit(f"fold {fold}: first Linear does not match head_hidden_dims")
        if tuple(weights["net.5.weight"].shape) != tuple(HEAD_ARGS["head_hidden_dims"][::-1]):
            raise SystemExit(f"fold {fold}: second Linear does not match head_hidden_dims")

        published = {
            "args": dict(HEAD_ARGS),
            "classifier_state_dict": weights,
            "epoch": int(state["epoch"]),
            "feature_dim": feature_dim,
            "base_soft_target_ckpt": state["base_soft_target_ckpt"],
            "inner_validation_metrics": state.get("inner_validation_metrics"),
            "split": "gene/protein-disjoint five-fold ClinVar",
        }
        destination = args.outdir / f"gene_protein_disjoint_fold_{fold}.pt"
        torch.save(published, destination)
        epochs.append(int(state["epoch"]))
        print(f"fold {fold}: epoch {state['epoch']}, "
              f"{destination.stat().st_size / 1e6:.1f} MB -> {destination.name}")

    print(f"\nselected epochs: {epochs}")


if __name__ == "__main__":
    main()
