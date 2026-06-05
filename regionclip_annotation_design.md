# Protein Domain/Site Annotation — Architecture Brainstorm & RegionCLIP Design

Design notes for annotating InterPro **domains** and **sites** on protein sequences,
borrowing architectures from image/language modeling. Captures (1) a survey of
transferable paradigms and (2) a detailed RegionCLIP-style design with the
inference/decoding mechanics worked out.

Context: current model is ESM2 (+LoRA) → BiLSTM encoder → multi-hot heads with
per-residue BCE/ASL over ~17,951 domain classes + ~1,383 site classes. Overlay
(a residue can carry several annotations, incl. a domain *and* a site) must be
preserved. The dominant pain point is **"None" dominance** — dense per-residue
multi-hot prediction manufactures ~19k×L negatives per protein.

---

## 1. Reframing the task

A protein is a **1-D image / token sequence**. Then:

- a **domain** = a *segment/span* with a class (contiguous, can nest/overlap) →
  like temporal action detection or 1-D object detection
- a **site** = a *point/keypoint* with a class (sparse) → like keypoint detection
- "annotate all of them" = **set prediction**: an unknown number of
  `(class, span)` instances, with overlap allowed

Key insight: None-dominance is partly an **architecture artifact**, not only a
loss artifact. Choosing a paradigm that predicts *instances* (not dense
per-residue negatives) removes the flood at the source; reweighting losses
(ASL/focal) only tames it.

---

## 2. Transferable paradigms (survey)

### 2a. CLIP → region-level (RegionCLIP / GLIP / LSeg / OpenSeg)
Vanilla CLIP aligns a *whole* image with a *whole* caption (global) — pooling one
protein vector against a bag of annotation tokens tells you "which domains are
present" but **loses localization**. The fix is the open-vocabulary
detection/segmentation line: encode each *residue or span* into a feature, encode
each *label* (InterPro name/abstract) with a **text encoder**, and predict by
**dot-product similarity**.

Why it fits this data:
- Label "head" is no longer a 19k one-hot matrix — it is **embeddings of InterPro
  descriptions**. Related IPRs sit near each other, so rare classes borrow signal
  from neighbors → directly attacks rare-class collapse.
- **Zero/few-shot** to IPRs barely seen in training (most of 19k are rare).
- Naturally multi-label/overlay (independent similarities), no fixed length.

### 2b. Set prediction (DETR / Pix2Seq / Mask2Former) — strongest structural fit
Matches the raw data format: InterPro literally gives `(IPR : start–end)` instances.

| Paradigm | Maps to | Why it fits |
|---|---|---|
| **DETR** (1-D / temporal-action-detection, e.g. TadTR) | N learned queries, each emits (class, start, end) via Hungarian matching + "no-object" class | predicts instances → nesting/overlap free; **dense negatives gone → None-dominance structurally removed**; variable count |
| **Pix2Seq** (detection-as-language-modeling) | decoder *generates* tokens: `IPR_id, start, end, …` | exactly the "annotation as tokens" idea but **keeps position** via span tokens; reuses a pretrained-LM decoder |
| **Mask2Former** | set of (class, soft-mask-over-residues) pairs | unifies semantic+instance; residue can join several masks (overlay); query-classification handles big label space |

With set prediction the loss is over ~(real annotations + "no-object"), **not**
19k×L slots — a cleaner cure than ASL (which only reweights the flood).

### 2c. Other transfers
- **Hierarchical / coarse-to-fine** (uses InterPro tree): predict type first
  (domain vs site vs family), then fine class. Cuts None-dominance; stacks on
  any option. Hyperbolic/Poincaré label embeddings for tree geometry.
- **Extreme/retrieval classification**: per-residue embedding + label-embedding
  table, predict by nearest label (DSSM-style). Scales to huge label sets, few-shot.
- **CRF / boundary head**: domains are contiguous — linear-chain CRF or explicit
  start/end boundary prediction enforces contiguity plain per-residue heads ignore.
- **Two-stage R-CNN style**: a cheap "is anything annotated here" region proposer
  (generalize the existing site-presence head), then classify only proposals.
- **Promptable / SAM-style**: residue prompt → domain span + class (interactive curation).
- **Masked-label pretraining** (BERT/MAE on the annotation channel): mask spans of
  annotations and reconstruct — self-supervised warmup for sparse labels.

### 2d. Fits the existing stack
ESM2+LoRA stays the **backbone** producing per-residue features; only the **head**
changes:
- minimal change / big leverage: **region-text alignment head** (replace the 19k
  linear head with residue→embed-space + InterPro label embeddings + dot product).
- biggest structural win: **DETR or Pix2Seq decoder** on ESM features → set of
  `(IPR, start, end)`.

Suggested experiment order: (1) region-text head, (2) Pix2Seq-style span
generation, (3) layer hierarchy + contiguity/CRF onto whichever is chosen.

---

## 3. RegionCLIP design — decoding "which domain sits in a region"

Core shift: the model **does not output class scores directly** — it outputs a
*vector per location* in the same low-dim space as the labels, and "which domain"
is answered by **nearest-label lookup**. The label dimensionality drops from
18k+1k one-hot to a shared `d`-dim space (e.g. 512–1000).

### 3a. The shared space (shapes)
- Backbone (ESM + encoder) → per-residue features `F ∈ R^{L×h}`.
- Projection `g: R^h → R^d` → residue embeddings `E = g(F) ∈ R^{L×d}`, L2-normalized.
- Text-encode each InterPro entry's name/abstract once → **label bank**
  `T ∈ R^{C×d}` (C ≈ 19k), L2-normalized, **precomputed and frozen**.
- Similarity: `S = (E @ Tᵀ) / τ ∈ R^{L×C}` (cosine + temperature, CLIP-style).

A residue's "meaning" is a single `d`-dim point; the 19k classes re-enter only at
the dot product, and only as a frozen matmul.

### 3b. Decode: which domain sits in region `[a,b]`
**Pool the region's embedding, then look up its nearest label:**

```
region embedding:   e_R = normalize( mean_{i in [a,b]} E[i] )      # one d-vector
scores over labels: s_R = e_R @ Tᵀ                                  # length C
domain(s) here:     argmax_c s_R                 (single)
                    OR  {c : sigmoid(s_R/τ) ≥ θ} (overlay)
```

The region is identified as the InterPro entry whose text embedding is closest to
the region's pooled embedding. For overlay, do not argmax — take **all labels
above a threshold** (independent per-label, since each `T[c]` is scored separately).

Where do regions `[a,b]` come from? Two designs:

**A — Dense first, then group (segmentation style).**
1. `P = sigmoid(S/τ)` → independent per-(residue, label) probability.
2. Per-residue present set = `{c : P[i,c] ≥ θ_c}`. Overlay falls out — multiple `c` can pass.
3. **Run-length encode** each label's positive residues into contiguous spans →
   `(IPR_c, start, end)` instances. Nested/overlapping domains emerge as separate per-label runs.
4. (Optional) re-score each extracted span with the pooled-embedding lookup above
   for a crisp class + confidence.

Closest to the current model (just swap the 19k linear head for the `E·Tᵀ` lookup).

**B — Spans first, then classify (RegionCLIP-faithful, detection style).**
1. A lightweight proposal/boundary head (or DETR-style queries) emits candidate spans.
2. Pool `E` over each span → `e_R`, dot with `T` → class by nearest label (+ "background").
3. Output = set of `(span, class)`. Crisp boundaries by construction; overlap free.

### 3c. The overlay subtlety (must handle)
A single residue vector `E[i]` can only be simultaneously *close* to label vectors
that are themselves close. A residue that is **both a domain and a site** has two
labels far apart in text-embedding space — one `d`-vector can't sit near both. Fixes:

1. **Separate spaces per annotation type** (already split domain vs site): two
   projections `E_dom = g_dom(F)`, `E_site = g_site(F)`, and two banks `T_dom`,
   `T_site`. Decode each independently → domain∩site overlay is automatic, each
   bank smaller/cleaner. **(Recommended — matches the data.)**
2. **Independent sigmoids, never softmax** over `S` — keeps each label's score
   independent so co-occurrence doesn't compete.
3. **Multi-slot residues** — predict `K` vectors per residue, each matched to one
   label (bridge toward set prediction); only if same-type co-occurrence (two
   overlapping domains at one residue) is common.

### 3d. Practical notes
- **Training**: BCE/ASL or InfoNCE on `S`, but **sample negatives** — never need
  all 19k columns per step; positives + a few hundred sampled negatives. Inference
  precomputes `T` once.
- **Frozen vs tuned labels**: start with `T` frozen (efficiency + zero-shot to rare
  IPRs); optionally add a small trainable projection on top of `T`.
- **Temperature `τ`**: learnable scalar like CLIP — important for calibrated thresholds.
- **Sharp boundaries**: dense+RLE can be ragged; add a CRF or boundary loss, or use Design B.
- **Rare-class win**: because `T` is derived from InterPro text, a rare IPR sits
  near its relatives, so the region lookup can land on it with few/zero training
  examples — what dense one-hot heads cannot do.

### 3e. End-to-end summary
> residue → `d`-vector → (pool over region) → dot with frozen InterPro text bank →
> nearest label(s) = the domain(s) in that region, with the span coming from
> RLE (Design A) or the proposal (Design B).

---

## 4. Possible next step
Prototype the region-text head in a new `_regionclip` copy alongside the ASL
variant: `E_dom`/`E_site` projection heads, build `T_dom`/`T_site` from the
InterPro TSV descriptions, temperature-scaled similarity, and the decode
(RLE + region-pool lookup). Keep ESM2+LoRA as the frozen/LoRA backbone.
