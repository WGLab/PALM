# PALM

**PALM** (Protein Annotation with a Language Model) is a residue-level protein
annotation model built on ESM2-150M with LoRA adapters, a bidirectional LSTM,
and four prediction heads.

Given a protein sequence, PALM predicts for **every residue**:

- whether the residue lies in a domain (none / one / multiple overlapping domains);
- whether the residue lies in a functional site (none / one / multiple overlapping sites);
- the InterPro **domain** class probabilities;
- the InterPro **site** class probabilities.

PALM is overlap-aware: residues covered by several annotations keep the full
label set instead of being forced into a single top-1 class. It also produces
residue embeddings, which can be compared between a wild-type and a mutant
sequence to measure how much a variant disrupts the predicted annotation.

**PALM-pathogenicity** is an optional set of five ClinVar-trained heads on top
of that comparison; their mean gives a supervised missense pathogenicity score.

## Installation

PALM runs in a conda environment defined by [`environment.yml`](environment.yml),
which pins a known-working set of versions (Python 3.10.15, PyTorch 2.10 built
for CUDA 12.8, and the Hugging Face stack).

```bash
git clone https://github.com/WGLab/PALM.git
cd PALM

conda env create -f environment.yml
conda activate palm
pip install -e . --no-deps
```

The last step makes `palm` importable from anywhere in the environment.
`environment.yml` has already installed the dependencies at their pinned
versions, so `--no-deps` keeps pip from resolving and replacing them.

`torch` is installed from the PyTorch CUDA 12.8 wheel index. If your driver
supports a different CUDA version, change the `--extra-index-url` and the
`+cu128` suffix on the torch pin to match; for a CPU-only machine, use the
plain `torch==2.10.0` with no extra index. Verify the GPU is visible before a
first run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`environment.yml` lists only what PALM itself imports: torch, transformers,
peft, huggingface-hub, numpy, and pandas. The extra packages used by the
development code under `docs/` (scipy, scikit-learn, matplotlib) are commented
out at the bottom of the file; uncomment them, or run
`pip install -e ".[docs]"`, if you need them.

### Model weights

Weights are not stored in this repository. They live in the
[Wangwpi/PALM](https://huggingface.co/Wangwpi/PALM) Hugging Face model
repository:

| File | What it is |
|---|---|
| `palm_soft_target_epoch8.pt` | the PALM annotation model |
| `pathogenicity/random_split_fold_{1..5}.pt` | the five PALM-pathogenicity heads |

You do not have to download anything by hand — PALM fetches what it needs on
first use and caches it. To download everything up front (recommended for
offline or cluster use):

```bash
conda activate palm
hf download Wangwpi/PALM --local-dir checkpoints
```

(`hf` comes with the `huggingface-hub` package already in the environment; on
older versions the command is `huggingface-cli download`.)

Then point PALM at that directory, either per run with `--checkpoint-dir` /
`checkpoint_dir=`, or once for the whole shell:

```bash
export PALM_CHECKPOINT_DIR=/path/to/checkpoints
```

Files found there are used directly; anything missing still falls back to the
Hugging Face download. A single checkpoint file can also be given by path.

PALM additionally downloads its base model, `facebook/esm2_t30_150M_UR50D`,
from Hugging Face on first use. Add `--local-files-only` (or
`local_files_only=True`) to run fully offline once everything is cached.

## Usage

### Annotate a sequence

```python
from palm import PALMPredictor

predictor = PALMPredictor(device="cuda")

result = predictor.annotate("MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEY")
print(result.table.head())
result.save("results/my_protein")     # writes .tsv and .npz
```

`result.table` is a pandas DataFrame with one row per residue:
`domain_occurrence_probability`, `site_occurrence_probability`, the top domain
and site InterPro classes with their probabilities, and the top-`k` classes per
residue. `result.save(...)` also writes an `.npz` holding the full class
probability matrices and the PALM residue embeddings.

### Annotate a list of sequences

```python
sequences = [
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEY",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR",
]

for sequence, result in zip(sequences, predictor.annotate_batch(sequences, top_k=5)):
    print(len(sequence), result.table["domain_occurrence_probability"].mean())
```

### Choosing where the weights come from

```python
# a directory of downloaded weights (or set PALM_CHECKPOINT_DIR)
predictor = PALMPredictor(checkpoint_dir="checkpoints")

# a single checkpoint file
predictor = PALMPredictor("checkpoints/palm_soft_target_epoch8.pt")

# explicit Hugging Face reference (the default)
predictor = PALMPredictor("hf://Wangwpi/PALM::palm_soft_target_epoch8.pt")
```

### Command line

```bash
# one sequence, or every record in a multi-record FASTA
python examples/predict_sequence.py --fasta protein.fasta --output-prefix results/protein
```

Useful options: `--sequence` instead of `--fasta`, `--checkpoint-dir`,
`--checkpoint`, `--top-k`, `--device`, `--local-files-only`, and `--ipr-tsv`
(an InterPro TSV with `ipr`, `type`, `name` columns, to print readable
annotation names instead of bare IPR accessions).

### Pathogenicity prediction

```python
from palm import PathogenicityEnsemble, mutate_sequence

model = PathogenicityEnsemble(device="cuda")   # loads the five default heads
# or: PathogenicityEnsemble(checkpoint_dir="checkpoints", device="cuda")

mutant = mutate_sequence(wt_sequence, 124, "S")
prediction = model.predict_one(wt_sequence, mutant, 124)
print(prediction)   # ensemble probability = unweighted mean of the five heads
```

```bash
python examples/predict_variant.py \
  --fasta PTEN.fasta --position 124 --mutant-aa S \
  --pathogenicity --output-prefix results/PTEN_C124S
```

Without `--pathogenicity`, the same command reports only PALM's zero-shot
wild-type-vs-mutant disruption (embedding and annotation-distribution changes).
That is a disruption measurement, not a calibrated pathogenicity probability;
only the supervised heads give the latter, and they are trained on ClinVar
labels.

## Repository layout

```text
palm/        the model: loader, annotation predictor, pathogenicity ensemble
examples/    ready-to-run scripts for annotation and variant scoring
tests/       lightweight tests that need no model weights
docs/        training, evaluation, visualization, and weight-upload code
```

Everything needed to *run* PALM is in `palm/` and `examples/`. The code used to
train the model, reproduce the benchmarks, plot annotation tracks, and publish
weights is kept in [`docs/`](docs/) — see [`docs/README.md`](docs/README.md).

## Reference

Citation information will be added when the PALM manuscript is available.

- Model weights: https://huggingface.co/Wangwpi/PALM
- Base protein language model: ESM2-150M (`facebook/esm2_t30_150M_UR50D`),
  Lin et al., *Science* 2023.
- Annotation labels: [InterPro](https://www.ebi.ac.uk/interpro/).
