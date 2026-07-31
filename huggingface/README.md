---
tags:
  - protein-language-model
  - protein-annotation
  - bioinformatics
  - esm2
  - peft
  - pytorch
library_name: transformers
pipeline_tag: token-classification
---

# PALM soft-target epoch 8

This repository contains the final PALM soft-target overlap-aware checkpoint
and optional ClinVar-trained pathogenicity heads.

## Files

```text
palm_soft_target_epoch8.pt
pathogenicity/random_split_fold_1.pt
pathogenicity/random_split_fold_2.pt
pathogenicity/random_split_fold_3.pt
pathogenicity/random_split_fold_4.pt
pathogenicity/random_split_fold_5.pt
checkpoint_manifest.json
```

The annotation checkpoint predicts four residue-level outputs:

- domain occurrence: none, one, or multiple;
- site occurrence: none, one, or multiple;
- InterPro domain classes; and
- InterPro site classes.

Overlapping domain/site labels use uniform soft targets over all true classes.
The final model uses ESM2-150M, LoRA on the last four ESM layers, learned
all-layer pooling, and a bidirectional LSTM residue encoder.

## Use

Install the code from the PALM GitHub repository, then use the Hugging Face
artifact reference directly:

```python
from palm import PALMPredictor

predictor = PALMPredictor(
    "hf://Wangwpi/PALM::palm_soft_target_epoch8.pt",
    device="cuda",
)
annotation = predictor.annotate("MSEQUENCE...")
```

The checkpoint is downloaded from the
[`Wangwpi/PALM`](https://huggingface.co/Wangwpi/PALM) model repository and
cached by `huggingface_hub`.

## Pathogenicity heads

The five random-split heads were selected at epochs 3, 4, 4, 3, and 4 using
inner-validation AUROC within a common 15-epoch budget. Their final prediction
is the unweighted arithmetic mean of the five pathogenicity probabilities.

These are supervised ClinVar-trained models. Their outputs must not be called
zero-shot. PALM annotation/embedding WT-mutant disruption can be evaluated
separately without loading these heads.

## Limitations

- The annotation and pathogenicity models are research models, not clinical
  diagnostic tools.
- Variant results depend on the protein transcript supplied to the model.
- Long proteins are limited by the ESM2 context window; variant workflows crop
  around the amino-acid change.
- Cohort filters and transcript aggregation must be reported with benchmark
  results.

## Citation

Citation information will be added when the PALM manuscript is available.
