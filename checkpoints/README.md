# Checkpoints

Model weights are intentionally not committed to GitHub. They are hosted in the
[Wangwpi/PALM Hugging Face model repository](https://huggingface.co/Wangwpi/PALM).
By default, PALM downloads them using references such as:

```text
hf://Wangwpi/PALM::palm_soft_target_epoch8.pt
```

This directory contains no model weights. It can optionally be used as a local,
gitignored cache for offline use.

The local fallback layout is:

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
