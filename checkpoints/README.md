# Checkpoints

Model weights are intentionally not committed to ordinary Git history.

Place the final soft-target annotation checkpoint here as:

```text
checkpoints/palm_soft_target_epoch8.pt
```

Optional ClinVar-trained head checkpoints can be placed under:

```text
checkpoints/pathogenicity_random_split/
```

The five fold-selected random-split heads are combined by an unweighted
arithmetic mean of their pathogenicity probabilities. They are supervised
downstream models and are not part of PALM's zero-shot annotation-disruption
score.

The selected epochs for folds 1-5 are 3, 4, 4, 3, and 4, respectively, using
inner-validation AUROC within the common 15-epoch selection budget.

The SHA-256 checksum of the original epoch-8 training checkpoint, before
optimizer/scaler removal, is:

```text
919fec8f2b9f0d0716f9a00a628fc44b6a4ca5d3b0d2dc26f4b746eb1f9b80d7
```

The exported inference checkpoint has a different checksum; publish that
checksum alongside the final weight download.
