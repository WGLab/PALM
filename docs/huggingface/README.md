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

# PALM (six-head, epoch 3)

This repository contains the final PALM annotation checkpoint and the
ClinVar-trained pathogenicity heads that go with it.

## Files

```text
palm_post_ep3_six_head.pt
pathogenicity/gene_protein_disjoint_fold_1.pt
pathogenicity/gene_protein_disjoint_fold_2.pt
pathogenicity/gene_protein_disjoint_fold_3.pt
pathogenicity/gene_protein_disjoint_fold_4.pt
pathogenicity/gene_protein_disjoint_fold_5.pt
checkpoint_manifest.json
```

The annotation checkpoint predicts six residue-level outputs:

- domain occurrence: none, one, or multiple;
- site occurrence: none, one, or multiple;
- repeat occurrence: none, one, or multiple;
- InterPro domain classes (17,951);
- InterPro site classes (991); and
- InterPro repeat classes (392).

Overlapping labels of the same kind use uniform soft targets over all true
classes. The model is ESM2-150M with LoRA on the last four ESM layers, learned
all-layer pooling, and a bidirectional LSTM residue encoder.

## Replaces the four-head release

The previous release published `palm_soft_target_epoch8.pt` and five
`pathogenicity/random_split_fold_*.pt` heads. Those files have been removed and
are not interchangeable with these:

- The old model had **four** heads over a 1,383-class site vocabulary. In that
  vocabulary 409 classes (392 Repeat and 17 PTM) had **no training examples** and
  could never be predicted, because a data filter had removed them. Repeat is now
  a trained head of its own and PTM is trained inside the site head, so the site
  vocabulary is 991 real classes.
- Pathogenicity features are correspondingly wider: 329,968 dimensions against
  the old 329,920. An old head cannot be loaded against this checkpoint, and a
  new head cannot be loaded against the old one.

Code from before this release will not load these files. Update the `palm`
package to the matching version.

## Use

Install the code from the PALM GitHub repository, then use the Hugging Face
artifact reference directly:

```python
from palm import PALMPredictor

predictor = PALMPredictor(
    "hf://Wangwpi/PALM::palm_post_ep3_six_head.pt",
    device="cuda",
)
annotation = predictor.annotate("MSEQUENCE...")
```

The returned table carries `domain_*`, `site_*` and `repeat_*` columns, each with
occurrence probabilities and the top class call.

## Pathogenicity heads

Five heads were trained on ClinVar with **gene/protein-disjoint** five-fold
cross-validation: all variants of a gene, and of any protein sequence shared
between genes, stay inside one fold, so a head cannot score well by recognising
a protein it was trained on. PALM itself is frozen; only the heads are trained.

Each fold selected its own epoch by inner-validation AUROC within a five-epoch
budget, choosing epochs 1, 1, 1, 1 and 2 — the heads converge almost
immediately. The prediction is the unweighted arithmetic mean of the five
probabilities.

Pooled out-of-fold performance over 151,823 ClinVar variants: AUROC 0.9229,
PR-AUC 0.8669.

```python
from palm import PathogenicityEnsemble

ensemble = PathogenicityEnsemble(
    "hf://Wangwpi/PALM::palm_post_ep3_six_head.pt",
    [f"hf://Wangwpi/PALM::pathogenicity/gene_protein_disjoint_fold_{i}.pt"
     for i in range(1, 6)],
    device="cuda",
)
result = ensemble.predict_one(wt_sequence, mutant_sequence, aa_pos)
```

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
- The repeat head is informative for annotation but weak as a standalone variant
  signal; it contributes little to the ClinVar pathogenicity ranking on its own.

## Citation

Citation information will be added when the PALM manuscript is available.
