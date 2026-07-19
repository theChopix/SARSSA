# 🏷️ Neuron Labeling (`src/plugins/neuron_labeling`)

> **What this is:** the fourth stage of a SARSSA pipeline — it gives
> every SAE neuron a **human-readable label**. After SAE training the
> neurons are anonymous directions; a labeling plugin produces a
> **neuron → label-string mapping** (ideally with a confidence
> score), which is what the inspection/steering dropdowns show and
> what the labeling-evaluation stage analyses.
>
> **Who should read this:** anyone choosing between the shipped
> labeling methods or interpreting label confidence, and contributors
> adding a labeling method (§6). The plugin contract in general lives
> in [`../README.md`](../README.md); this doc is the labeling
> specifics.

---

## 📑 Table of Contents

1. [🗺️ Big picture: the category contract](#-1-big-picture-the-category-contract)
2. [🏷️ Today's plugins: the tag-based family](#-2-todays-plugins-the-tag-based-family)
3. [🔬 The two tag-based methods in detail](#-3-the-two-tag-based-methods-in-detail)
4. [📥 Inputs · 📤 Outputs](#-4-inputs--outputs)
5. [🎛️ `run()` parameters](#-5-run-parameters)
6. [🛠️ Adding your own labeling method](#-6-adding-your-own-labeling-method)
7. [⚠️ Operational notes & gotchas](#-7-operational-notes--gotchas)
8. [➡️ Where to go next](#-8-where-to-go-next)

---

## 🗺️ 1. Big picture: the category contract

Neuron labeling is the **fourth stage** of a pipeline — the last
`one_time` step. Conceptually the category's job is small and sharply
defined:

```
neuron id  ──▶  label string (+ confidence in [-1, 1], when the method has one)
```

*How* a plugin arrives at that mapping is deliberately open. Anything
that names neurons fits the category: methods built on the dataset's
**tags**, methods that feed top-activating items' **descriptions to
an LLM**, methods that compose several tags into an **aggregated
label**, … The shipped plugins (§2) are all tag-based today, but the
category — and everything downstream — only depends on the mapping
itself.

The mapping is what the rest of SARSSA runs on: the `inspection` and
`steering` dropdowns render it (label text, `conf` value, colour
tint), and the `labeling_evaluation` stage analyses the label set as
a whole.

---

## 🏷️ 2. Today's plugins: the tag-based family

Both current plugins label neurons **with tags from the dataset**
(`neuron → dataset tag`), and share one recipe:

```
1. run every item through CFM-encode → SAE-encode        (shared)
       → item_acts: (num_items × num_neurons) activation matrix
2. pick, per neuron, the tag that fits it best           (per method)
3. score each label with a confidence                    (shared)
```

They differ **only in step 2** — the rule that picks the tag:
**`tag_correlation`** takes the tag whose *presence correlates* most
with the neuron's activation; **`tf_idf`** takes the tag that is most
*distinctive* for the neuron under a TF-IDF weighting (§3).

The shared pieces live next to the plugins:

- **Item activations** (`compute_sae_item_activations`,
  `utils/torch/evaluation.py`): each item is fed through the pipeline
  as a **one-hot interaction row** ("a user who interacted with
  exactly this one item"). Column `n` of the resulting `item_acts`
  matrix is neuron `n`'s fingerprint across the catalogue. All-zero
  columns are **dead neurons** — no signal, so both plugins leave
  them **unlabelled** (`None`) rather than assigning a spurious tag.
- **Confidence** (`_confidence.py`; underscore = not a plugin): the
  **point-biserial correlation** `r ∈ [-1, 1]` between the neuron's
  continuous activation and the *binary presence* of its assigned tag
  across items (point-biserial = Pearson with one binary variable).
  `r ≈ 1` means the neuron fires reliably exactly on items carrying
  the tag; `r ≈ 0` means the label is decorative. The mean over
  labelled neurons is logged as the **`mean_confidence`** metric —
  the run's aggregate label-quality number.

---

## 🔬 3. The two tag-based methods in detail

**`tag_correlation` — label = the most-correlated tag.** Computes the
full (tags × neurons) correlation matrix and assigns each neuron the
argmax tag. It therefore **directly optimises the confidence metric**
— by construction no other tag choice can score higher per neuron.
Tags applied to fewer than `min_support` items are excluded
(correlation on a handful of items is noise).

**`tf_idf` — label = the most-distinctive tag.** Builds a tag–neuron
association matrix (tag–item probabilities × activations) and runs
**TF-IDF** over it, so a tag scores highly for a neuron when it is
strong *there* and rare *elsewhere* — globally frequent tags get
discounted by the IDF term. The `orientation` parameter chooses which
entity plays the *document* role:

- **`tag_as_document`** (default) — term frequency normalised per
  tag: "which neuron is characteristic *of this tag*";
- **`neuron_as_document`** — normalised per neuron: "which tag is
  characteristic *of this neuron*".

TF-IDF optimises distinctiveness, **not** correlation — so its
confidences (still point-biserial, §2) can be lower or even negative
than `tag_correlation`'s.

---

## 📥 4. Inputs · 📤 Outputs

**Inputs** (both current plugins) — requires all three prior steps:

| From | What |
|---|---|
| `dataset_loading` | `items.npy`, `tag_ids.json`, `tag_item_matrix.npz` |
| `training_cfm` | the CFM checkpoint (frozen encoder) |
| `training_sae` | the SAE checkpoint |

**Outputs**:

| Artifact / param | Content |
|---|---|
| `neuron_labels.json` | **the category's main product** — neuron id → `{"label", "confidence"}` (both `None` for dead neurons) |
| `top_tag_per_neuron.json` | neuron id → tag name (the label without the score) |
| `top_neuron_per_tag.json` | the reverse view: tag → the neuron that best represents it |
| `item_acts.npz` | the (items × neurons) activation matrix, stored sparse (TopK codes are ~94% zeros) |
| `tag_item_prob.npz` | *tf_idf only* — the normalised tag–item matrix it scored with |
| params `num_tags`, `num_neurons`; metric `mean_confidence` | run summary |

**Who consumes this:** `inspection` and `steering` read
`neuron_labels.json` for their neuron dropdowns and `item_acts.npz`
to rank items; the `labeling_evaluation` plugins (dendrogram,
embedding maps) analyse the label set as a whole.

---

## 🎛️ 5. `run()` parameters

| Plugin | Param | Default | Meaning |
|---|---|---|---|
| both | `batch_size` | 1024 | items per forward pass when computing activations — speed/memory only, results identical |
| both | `seed` | 42 | reproducibility of the activation pass |
| `tag_correlation` | `min_support` | 5 | minimum items a tag must apply to before it may label a neuron |
| `tf_idf` | `orientation` | `tag_as_document` | which entity is the TF-IDF document (§3) |

---

## 🛠️ 6. Adding your own labeling method

A labeling plugin is a good first plugin: no training loop, pure
matrix (or API) work. The method is open — §1's examples (an LLM
naming neurons from their top items' descriptions, aggregated
multi-tag labels, …) all fit. What matters is keeping the **output
contract** downstream stages rely on:

- `neuron_labels.json` in the neuron → `{"label", "confidence"}`
  shape (confidence may be `None` if your method has no natural
  score, but the dropdown tinting then has nothing to show);
- `item_acts.npz` if inspection-style item ranking should work;
- for tag-based methods, the `top_*_per_*.json` pair the evaluation
  plugins visualise.

Reuse `compute_sae_item_activations` for the activation matrix and —
if your labels are tag-based — `_confidence.py` for scoring.

---

## ⚠️ 7. Operational notes & gotchas

- **🔴 The current plugins require tags.** Both declare
  `tag_ids.json` / `tag_item_matrix.npz` as required inputs; a
  dataset without tags fails here with `MissingContextError` (the
  gotcha called out in the
  [dataset-loading README](../dataset_loading/README.md) §7). A
  future non-tag-based plugin would lift this for its own runs.
- **Runs on CPU by design.** The one-hot activation pass builds
  `num_items`-wide batches, which OOMs small GPUs, so the device is
  pinned to CPU. It scales with catalogue size but is a
  minutes-scale, not hours-scale, computation.
- **Confidence can be negative.** `-1` is a perfectly *anti*-tracking
  label. Expect this mainly from `tf_idf`, whose choice rule doesn't
  maximise correlation (§3).

---

## ➡️ 8. Where to go next

- **The SAE whose neurons get named here:**
  [`../training_sae/README.md`](../training_sae/README.md)
- **The stages that consume the labels:** `labeling_evaluation/`,
  `inspection/`, `steering/` (per-category READMEs still to come)
- **The plugin contract** (discovery, `io_spec`, `run()` params):
  [`../README.md`](../README.md)
