---
license: mit
base_model: facebook/esm2_t30_150M_UR50D
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
interpro_classes.tsv
checkpoint_manifest.json
```

`interpro_classes.tsv` maps every output class to its InterPro entry:

```text
head    class_index  interpro_id  interpro_type  name
domain  0            IPR000001    Domain         Kringle
site    0            IPR000035    Conserved_site Alkylbase DNA glycosidase, conserved site
```

`class_index` is 0-based and per head: domain indices run 0-17950, and site
indices restart at 0 and run to 1382. The checkpoint itself carries the same
accession-to-index maps as `ipr2domain` and `ipr2site`; this file adds the
human-readable names.

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

## License

Released under the MIT license, the same terms as the GitHub repository
(`https://github.com/WGLab/PALM`).

PALM is a fine-tune of `facebook/esm2_t30_150M_UR50D`, which is itself MIT
licensed. The annotation model was trained on UniRef50 sequences labelled with
InterPro domain and site entries, and the optional pathogenicity heads on
ClinVar variant classifications; anyone redistributing derived data should
check the terms of those sources as well.

## Citation

Citation information will be added when the PALM manuscript is available.
