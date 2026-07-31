# PALM

PALM is an overlap-aware residue-level protein annotation model built from
ESM2-150M, LoRA adapters, a bidirectional LSTM, and four prediction heads. This
repository contains the final soft-target training implementation, inference
API, benchmark entry points, and visualization utilities used with the epoch-8
checkpoint.

## Model

PALM predicts four outputs at every residue:

1. domain occurrence: no domain, one domain, or multiple overlapping domains;
2. site occurrence: no site, one site, or multiple overlapping sites;
3. InterPro domain-class probabilities; and
4. InterPro site-class probabilities.

For a residue annotated with several domain or site classes, the classification
target is uniform over every true class. This avoids selecting an arbitrary
top-1 label at overlapping residues. The occurrence heads remain three-class
models during training. For occurrence precision, recall, and F1, classes 1 and
2 are combined as annotation present and class 0 is annotation absent.

The released model is the soft-target epoch-8 checkpoint. Earlier PALM
multiclass, relaxed-overlap, and epoch-4 checkpoints are deliberately excluded
from this directory.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <PALM_REPOSITORY_URL>
cd PALM
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first online run downloads `facebook/esm2_t30_150M_UR50D` from Hugging
Face. In an offline environment, cache that model first and add
`--local-files-only` to the commands below.

## Model weights

The final checkpoint is approximately 2.1 GB and should be distributed with
Git LFS, a GitHub Release, Hugging Face, or Zenodo rather than committed to
ordinary Git history. Download it to:

```text
checkpoints/palm_soft_target_epoch8.pt
```

The loader verifies both of the final-model metadata fields and requires epoch
8:

```text
label_mode = sparse_set_softmax_with_instance_heads
specific_class_loss = uniform_soft_target_cross_entropy
epoch = 8
```

The checkpoint contains the InterPro-to-class-index maps. An optional InterPro
TSV supplies human-readable names and must have `ipr`, `type`, and `name` as its
first three columns.

Before uploading the original training checkpoint, remove optimizer and AMP
scaler state to reduce the distribution artifact:

```bash
python training/export_inference_checkpoint.py \
  --input runs/palm_soft_target/epoch_8.pt \
  --output checkpoints/palm_soft_target_epoch8.pt
```

## Annotate a protein

Command line:

```bash
python examples/predict_sequence.py \
  --fasta protein.fasta \
  --checkpoint checkpoints/palm_soft_target_epoch8.pt \
  --ipr-tsv data/ipr_domain_sites_entry.tsv \
  --output-prefix results/protein
```

This writes:

- `results/protein.tsv`: per-residue occurrence calls, occurrence
  probabilities, and top class predictions;
- `results/protein.npz`: full domain/site class probability matrices and PALM
  residue embeddings.

Python API:

```python
from palm import PALMPredictor

predictor = PALMPredictor(
    "checkpoints/palm_soft_target_epoch8.pt",
    ipr_tsv="data/ipr_domain_sites_entry.tsv",
    device="cuda",
)
annotation = predictor.annotate("MKT...", top_k=5)
print(annotation.table.head())
annotation.save("results/example")
```

The readable TSV stores only the requested top `k` classes per residue. The NPZ
always stores every class probability and is the source for stacked plots and
downstream analyses.

## Score a missense variant

PALM's zero-shot output is a WT-versus-mutant disruption measurement; it is not
a finetuned pathogenicity probability.

```bash
python examples/predict_variant.py \
  --fasta PTEN.fasta \
  --position 124 \
  --mutant-aa S \
  --checkpoint checkpoints/palm_soft_target_epoch8.pt \
  --output-prefix results/PTEN_C124S
```

The output includes per-residue PALM embedding L2 differences and L1 changes in
all four prediction-head distributions. Summary values are reported for the
whole sequence, mutation position, local window, and maximum disruption.

### Supervised pathogenicity heads

ClinVar-finetuned heads are optional downstream models. Supply one or more head
checkpoints to `predict_variant.py`, or load them directly:

```python
from palm import PathogenicityEnsemble, mutate_sequence

model = PathogenicityEnsemble(
    "checkpoints/palm_soft_target_epoch8.pt",
    [
        "checkpoints/pathogenicity_random_split/fold_1.pt",
        "checkpoints/pathogenicity_random_split/fold_2.pt",
        "checkpoints/pathogenicity_random_split/fold_3.pt",
        "checkpoints/pathogenicity_random_split/fold_4.pt",
        "checkpoints/pathogenicity_random_split/fold_5.pt",
    ],
)
mutant = mutate_sequence(wt_sequence, 124, "S")
prediction = model.predict_one(wt_sequence, mutant, 124)
```

The ensemble score is the unweighted arithmetic mean of the five fold-selected
head probabilities. It must not be described as zero-shot because the heads
were trained using ClinVar labels.

## Visualization

Create the four-track annotation plot and a separate legend:

```bash
python visualization/plot_annotation.py \
  --tsv results/protein.tsv \
  --npz results/protein.npz \
  --output results/protein_annotation.png
```

The classification tracks stack class probabilities from 0 to 1. Classes whose
maximum probability is below the display threshold are grouped as `Other
classes`; no top-1 class is forced for visualization.

To place WT-mutant PALM embedding differences into a PDB B-factor column:

```bash
python visualization/embedding_delta_to_bfactor.py \
  --pdb structure.pdb \
  --delta-tsv results/PTEN_C124S.per_residue.tsv \
  --chain A \
  --output results/PTEN_C124S_embedding_delta.pdb
```

Use `--pdb-offset` when PDB residue numbering and canonical sequence numbering
differ. Validate the sequence-to-structure mapping before interpreting the
colored structure.

## Evaluation and benchmarks

Every evaluation command accepts explicit input and output paths. No private
cluster path, result table, or unpublished dataset is embedded in the code.

### Held-out residue annotation

```bash
python evaluation/evaluate_annotation.py \
  --checkpoint checkpoints/palm_soft_target_epoch8.pt \
  --test-dir data/uniref50/test \
  --ipr-tsv data/ipr_domain_sites_entry.tsv \
  --output-dir results/annotation_test
```

The output reports domain occurrence, site occurrence, domain classification,
and site classification separately. Class accuracy is relaxed: a prediction is
correct if it belongs to the complete true label set at that residue.

For PID-stratified FASTA subdirectories named `pid_*`, replace `--test-dir`
with `--pid-groups-dir data/uniref50/test_pid_groups`. The model is loaded once
and the same four-head metrics are written for every PID bin.

### Zero-shot pathogenicity

```bash
python evaluation/zero_shot_pathogenicity.py \
  --input data/variants_with_sequences.tsv \
  --checkpoint checkpoints/palm_soft_target_epoch8.pt \
  --output-dir results/zero_shot
```

Required default input columns are:

| Column | Meaning |
|---|---|
| `wt_sequence` | transcript-specific WT protein sequence |
| `mutant_sequence` | mutant protein sequence; optional if `aa_to` is present |
| `aa_pos` | one-based amino-acid position |
| `aa_to` | mutant amino acid when the mutant sequence is absent |
| `label` | binary benchmark label, 0 benign and 1 pathogenic |

The script evaluates ESM2 masked-marginal LLR, frozen ESM2 embedding
disruption, PALM embedding disruption, PALM annotation disruption, individual
PALM heads, and equal-z combined scores. For multi-transcript genomic variants,
prepare one row per selected transcript or aggregate transcript scores in a
predeclared protocol. Transcript selection/aggregation is part of dataset
preprocessing and must be reported; it is not learned by PALM.

### Supervised pathogenicity

```bash
python evaluation/evaluate_pathogenicity_ensemble.py \
  --input data/variants_with_sequences.tsv \
  --palm-checkpoint checkpoints/palm_soft_target_epoch8.pt \
  --head-checkpoints checkpoints/pathogenicity_random_split/fold_{1,2,3,4,5}.pt \
  --output results/pathogenicity_predictions.tsv
```

### External ANNOVAR/dbNSFP predictors

After joining PALM/ESM scores and external predictions by variant identifier,
run complete-case and pairwise-matched comparisons with:

```bash
python evaluation/compare_external_predictors.py \
  --input data/joined_variant_scores.tsv \
  --label-column label \
  --numeric-columns esm2_llr,palm_embedding,esm2_llr_plus_palm_embedding \
  --categorical-columns AlphaMissense_pred,MetaRNN_pred \
  --reference-score esm2_llr_plus_palm_embedding \
  --output-dir results/external_predictors
```

The strict table uses one common complete-case cohort. The pairwise table uses
each external tool's own matched cohort and recomputes the reference score on
that same subset.

### Reactome embedding organization

```bash
python evaluation/reactome_knn.py \
  --input data/reactome_proteins.tsv \
  --checkpoint checkpoints/palm_soft_target_epoch8.pt \
  --include-esm2 \
  --output-dir results/reactome
```

The default columns are `protein_id`, `sequence`, and `reactome_top_level`.
Embeddings are standardized, and leave-one-out cosine kNN is evaluated at
`k=5,10,20,50` for assignment accuracy and neighbor purity.

### FlexiNR mutational screen

```bash
python evaluation/flexinr_screen.py \
  --input data/flexinr_paired_variants.tsv \
  --checkpoint checkpoints/palm_soft_target_epoch8.pt \
  --include-esm2 \
  --output-dir results/flexinr
```

Required columns are `screen`, `tile_id`, `wt_sequence`, and
`mutant_sequence`. Supported outcome columns are `delta_vs_wt`,
`abs_delta_vs_wt`, `hit_changed`, and `loss_of_hit`. If available,
`motif_relative_start` and `motif_relative_end` define the one-based inclusive
motif interval. The benchmark uses five-fold GroupKFold by `tile_id` within
each screen.
It evaluates PALM annotation-disruption scalars, ESM2 LLR, PALM and ESM2
embedding-difference feature sets, and matched PALM+ESM2 concatenated features.

## Train PALM

The final training implementation is
`training/train_soft_target.py`. Training and validation directories contain
FASTA shards with headers of the form:

```text
>protein_id InterPro=IPR000001:10-40;IPR000002:35-60
MSEQUENCE...
```

Intervals are one-based and inclusive. Overlapping intervals are retained as
sets. The InterPro label table is tab-separated with this schema:

```text
ipr    type    name
IPR... Domain  ...
IPR... Site    ...
```

Run on one GPU:

```bash
python training/train_soft_target.py \
  --train-dir data/uniref50/train \
  --val-dir data/uniref50/val \
  --ipr-tsv data/ipr_domain_sites_entry.tsv \
  --output-dir runs/palm_soft_target \
  --epochs 8
```

Run distributed training:

```bash
torchrun --standalone --nproc-per-node=2 training/train_soft_target.py \
  --train-dir data/uniref50/train \
  --val-dir data/uniref50/val \
  --ipr-tsv data/ipr_domain_sites_entry.tsv \
  --output-dir runs/palm_soft_target \
  --epochs 8
```

A configurable Slurm template is provided in
`training/train_soft_target.sbatch`. Its defaults reproduce the final setup:
ESM2-150M, LoRA rank 8 and alpha 16, LoRA on the last four ESM layers, uniform
soft targets, equal loss weights, learning rate `1e-4`, and eight epochs.

## Repository layout

```text
palm/           model, checkpoint loader, inference, pathogenicity ensemble
training/       final training code, Slurm template, checkpoint export utility
evaluation/     annotation, zero-shot, supervised, Reactome, and FlexiNR code
visualization/  annotation plots and PDB B-factor export
examples/       minimal sequence and missense-variant examples
checkpoints/    weight placement instructions; weights are gitignored
tests/          lightweight tests that do not download model weights
```

## Reproducibility notes

- Keep zero-shot disruption scores separate from supervised pathogenicity-head
  probabilities.
- State the exact transcript mapping and aggregation protocol for genomic
  variants with multiple protein consequences.
- Report cohort filters, especially PubMind confidence thresholds, with every
  benchmark value.
- The ESM2 architecture limits one forward pass to approximately 1,024
  residues. The benchmark scripts crop long WT/mutant pairs around the mutation.
  Whole-protein annotation follows the checkpoint-compatible truncation
  behavior.
- Do not compare metrics across cohorts or transcript-mapping pipelines as if
  they differed only by model checkpoint.

## Citation and license

Add the paper citation, model-weight DOI/URL, author list, and the selected
software/model license before making the repository public.
