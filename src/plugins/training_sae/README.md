# 🕸️ Training SAE (`src/plugins/training_sae`)

> **What this is:** the third stage of a SARSSA pipeline — it trains
> the **sparse autoencoder (SAE)** that re-expresses the CFM's dense
> user embeddings as wide, sparse vectors whose individual neurons
> tend to correspond to human-interpretable concepts. This is the
> model that neuron labeling names and steering manipulates.
>
> **Who should read this:** anyone choosing between the three SAE
> variants or tuning their training (especially the augmentation /
> contrastive options), and contributors adding an SAE variant (§7).
> The plugin contract in general lives in
> [`../README.md`](../README.md); this doc is the SAE-training
> specifics.

---

## 📑 Table of Contents

1. [🗺️ Big picture & module organization](#-1-big-picture--module-organization)
2. [🧮 The SAE model family](#-2-the-sae-model-family)
3. [🎯 The loss](#-3-the-loss)
4. [🔁 The training loop](#-4-the-training-loop)
5. [📥 Inputs · 📤 Outputs](#-5-inputs--outputs)
6. [🎛️ `run()` parameters](#-6-run-parameters)
7. [🛠️ Adding your own SAE variant](#-7-adding-your-own-sae-variant)
8. [⚠️ Operational notes & gotchas](#-8-operational-notes--gotchas)
9. [➡️ Where to go next](#-9-where-to-go-next)

---

## 🗺️ 1. Big picture & module organization

SAE training is the **third stage** of a pipeline: it loads the
trained CFM, encodes users into dense embeddings, and trains an
autoencoder whose hidden layer is much **wider** than the embedding
(default 8192 vs 1024 — the *expansion ratio*) but **sparse** — only
a handful of neurons fire per user. Reconstruction quality keeps the
recommendations intact; sparsity is what makes single neurons
interpretable.

Three plugins ship, differing **only in how sparsity is enforced**
(§2): `basic_sae_trainer`, `topk_sae_trainer`,
`batch_topk_sae_trainer`.

The code is split so that the *models* are shared and the *plugins*
stay thin wrappers around one `train()` function each:

```
src/plugins/training_sae/                     utils/torch/models/sae_model/
├── basic_sae_trainer/      ── uses ──▶       ├── sae_model.py     ← SAE base class:
├── topk_sae_trainer/                         │     encode/decode, standardization,
└── batch_topk_sae_trainer/                   │     ALL loss terms, dead-neuron counter
      each: io_spec + run() + train()         ├── basic_sae/       ← subclass: L1 sparsity
                                              ├── topk_sae/        ← subclass: hard top-k
      evaluation: utils/torch/evaluation.py   └── batch_topk_sae/  ← subclass: batch top-k
      (evaluate_sparse_encoder)                     (each ~60 lines: post_process_embedding
                                                     + total_loss + get_config)
```

The base class owns everything hard — the architecture, every loss
term, the dead-neuron bookkeeping. A variant subclass only answers
two questions: *how is the raw code sparsified?*
(`post_process_embedding`) and *how are the loss terms combined?*
(`total_loss`). The three trainer plugins are near-copies of each
other; `topk` and `batch_topk` differ only in which model class they
build, `basic` additionally drops the contrastive plumbing.

---

## 🧮 2. The SAE model family

**Shared architecture** (`sae_model.py :: SAE`) — an untied one-layer
autoencoder over dense embeddings `x` of dim `input_dim`:

- **Standardize**: each embedding is standardized per-sample
  (subtract its mean, divide by its std); the decoder output is
  de-standardized with the same statistics, so the SAE models the
  *shape* of the embedding, not its offset/scale.
- **Encode**: `e_pre = ReLU((x − b_dec) @ W_enc + b_enc)` — the
  non-negative *pre-codes*; then the variant's
  `post_process_embedding` sparsifies them into the codes `e`.
- **Decode**: `x̂ = e @ W_dec + b_dec`. Decoder rows are kept
  **unit-norm** (re-normalized + gradient projected every step) so
  each neuron is a fixed *direction* in embedding space and its
  activation is the amount of it — the property steering relies on.
- **Dead-neuron counter**: after each training batch, neurons that
  fired nowhere in the batch have a per-neuron counter incremented
  (reset on any activity); a neuron is *dead* after
  `n_batches_to_dead` silent batches. Feeds the auxiliary loss (§3).

**The three sparsity mechanisms:**

| Variant | `post_process_embedding` | Sparsity control |
|---|---|---|
| **BasicSAE** | identity — codes stay as ReLU left them | soft: the **L1 penalty** (`l1_coef`) pushes activations to zero |
| **TopKSAE** | keep each user's **top `k`** pre-codes, zero the rest | hard: exactly `k` active neurons per user |
| **BatchTopKSAE** | keep the top **`k × batch_size`** values across the *whole batch* | hard on average: per-user count flexes around `k` (heavy users take more of the budget). A running **threshold** (mean of the per-batch minimum kept value) is learned during training and used to gate activations at inference, where there is no batch to rank against. |

---

## 🎯 3. The loss

All terms are computed in the base class for every variant
(`_compute_loss_dict`); the variant's `total_loss` combines them:

```
Loss = reconstruction_coef · Rec  +  l1_coef · L1  +  auxiliary_coef · Aux  +  contrastive_coef · Con
```

In the TopK/BatchTopK trainers
`reconstruction_coef = 1 − (auxiliary_coef + contrastive_coef)` (the
trainer rejects `aux + con ≥ 1`), so enabling the optional terms
trades weight *away* from reconstruction explicitly. The Basic
trainer keeps reconstruction at full weight and simply adds the
auxiliary term.

| Term | What it is |
|---|---|
| **Rec** | reconstruction error on the standardized embedding — selectable per run: `Cosine` (1 − cosine similarity; direction only) or `L2` (mean squared error). |
| **L1** | mean `Σ|e|` over the batch — the classic sparsity penalty. Primary control for BasicSAE; for the top-k variants it merely shrinks the surviving activations. |
| **Aux** — auxiliary (dead-neuron) loss | takes the currently-dead neurons, lets their top `topk_aux` *pre-codes* try to reconstruct what the live code left unexplained (the residual), and penalizes that residual reconstruction with MSE. Gradient flows only through dead neurons — a nudge that gives them a useful direction again instead of letting capacity rot. |
| **Con** — contrastive loss | symmetric **InfoNCE** between two views of the same user (§4): codes are L2-normalized, all-pairs cosine similarities divided by `temperature`, cross-entropy against "my own second view is the match". Pulls augmented views of a user to the *same* neurons, favouring robust features. The low temperature matters: codes are non-negative, so cosines live in [0, 1] — too flat for a softmax until sharpened. |
| `L0` | mean active-neuron count — **logged as a diagnostic only**, never part of `Loss`. |

---

## 🔁 4. The training loop

Each trainer's `train()` is the same early-stopping loop as CFM
training (see [`../training_cfm/README.md`](../training_cfm/README.md)
§3) with one extra moving part: **where the training inputs come
from**. That depends on augmentation:

- **Fast path** (`sample_interactions` off, `contrastive_coef` 0):
  user embeddings are precomputed once with the frozen CFM and epochs
  iterate over those — no base-model work per epoch.
- **On-the-fly path** (either option on): epochs iterate over the
  *interaction* rows and encode per batch, because the views must be
  freshly sampled each epoch:
  - **anchor augmentation** (`sample_interactions`): keep ~80% of
    each user's interactions (random dropout), then encode — the SAE
    never sees exactly the same embedding twice;
  - **contrastive positive** (`contrastive_coef > 0`): an
    *independently* dropped-out copy of the same rows, encoded as the
    second view for the InfoNCE term.

The rest in brief: validation runs every `evaluate_every` epochs —
all loss terms on precomputed val embeddings (sample-weighted means)
plus `evaluate_sparse_encoder`, which reports reconstruction
**CosineSim**, sparsity **L0** and **DeadNeurons**, and the
recommendation metrics **R@20 / NDCG@20 with their degradation vs the
base model** (the SAE's real quality bar: how much recommendation
quality the sparse detour costs). Early stopping tracks the total
validation loss, with patience counted in *evaluations* (not epochs);
the best model is restored and evaluated once on test. Cancellation
is checked every batch.

---

## 📥 5. Inputs · 📤 Outputs

**Inputs** — requires `dataset_loading` **and** `training_cfm`:

| From | What | Becomes |
|---|---|---|
| `dataset_loading` | `train/valid/test_csr.npz` + dataset params | data + provenance |
| `training_cfm` | the model checkpoint (loaded via `load_base_model`) + `model`, `factors`, `users`, `items` params | the frozen encoder; `factors` sizes the SAE input |

`load_context()` cross-checks that the dataset's user/item counts
match the base model's — loading a CFM trained on a *different*
dataset prefix fails fast instead of training nonsense.

**Outputs**:

| Kind | Content |
|---|---|
| artifact | `config.json` + `model.pt` via the `"model"` saver — same persistence/registry mechanics as the CFM (see [`../training_cfm/README.md`](../training_cfm/README.md) §6), just with `@register_sae_model` / `load_sae_model` |
| params | `model` (variant name), `expansion_ratio`, dataset provenance, and the full `base_*` provenance of the CFM it was trained on |
| metrics | `loss/<term>/train` and `loss/<term>/valid` curves for every term in §3, `CosineSim`/`L0`/`DeadNeurons`/`R20`/`NDCG20`(+`_Degradation`)`/valid` and `/test` |

**Who consumes the model:** `neuron_labeling` (activations per item),
`inspection` (top-activating items per neuron), `steering` (amplify a
neuron, decode back).

---

## 🎛️ 6. `run()` parameters

All three trainers share this surface; `top_k` is absent in Basic,
and the Contrastive group is TopK/BatchTopK-only. Grouped as in the
UI (note the nested groups under *Training Loop → Loss*):

| Group | Param | Default | Meaning |
|---|---|---|---|
| Architecture | `embedding_dim` | 8192 | SAE width (dictionary size); with ELSA's 1024 factors → 8× expansion |
| | `top_k` | 32 | active neurons per user (TopK) / per-batch budget (BatchTopK) |
| | `normalize` | off | L2-normalize the codes |
| Training Loop | `epochs` / `batch_size` / `early_stop` / `seed` | 250 / 512 / 50 / 42 | as in CFM training; `early_stop` counts evaluations |
| | `sample_interactions` | off | anchor augmentation (§4) |
| ⤷ Loss | `reconstruction_loss` | Cosine | `Cosine` or `L2` |
| | `l1_coef` | 3e-4 | L1 penalty strength |
| ⤷⤷ Dead Neurons Auxiliary | `auxiliary_coef` | 0.0 | weight of the aux loss (0 = off) |
| | `topk_aux` | 512 | dead neurons revived per step |
| | `n_batches_to_dead` | 5 | silent batches before "dead" |
| ⤷⤷ Contrastive | `contrastive_coef` | 0.0 | weight of InfoNCE (0 = off) |
| | `temperature` | 0.2 | softmax sharpening (§3) |
| Optimizer | `lr` / `beta1` / `beta2` | 3e-4 / 0.9 / 0.99 | Adam |
| Evaluation | `evaluate_every` | 10 | validation cadence (epochs) |
| | `target_ratio` | 0.2 | ranking-eval holdout, metrics-only |

---

## 🛠️ 7. Adding your own SAE variant

Usually **no new trainer is needed** — a new sparsity mechanism is a
new model subclass:

1. `utils/torch/models/sae_model/<name>/<name>.py` — subclass `SAE`,
   implement `post_process_embedding`, `total_loss`, `get_config`;
   decorate with `@register_sae_model("<Type>")` and add the import
   to `model_loader.py` (registration is an import side effect — see
   the CFM README §6 box).
2. Copy the closest trainer plugin folder, swap the model class and
   names, and adjust the exposed params. If your variant needs no
   contrastive term, start from `basic_sae_trainer` (it's the one
   without that plumbing).

Downstream stages only touch the checkpoint through `load_sae_model`
and `encode`/`decode`, so they work unchanged.

---

## ⚠️ 8. Operational notes & gotchas

- **This is the expensive stage without a GPU.** The default 250
  epochs over 8192-wide codes is hours on CPU; the augmented /
  contrastive path adds a full base-model encode per batch on top.
  On CPU-only hosts, prefer loading a pre-trained pipeline and lower
  `epochs` when you do train.
- **`aux + con ≥ 1` is rejected** at launch — the optional terms take
  their weight out of reconstruction (§3), and there must be some
  left.
- **BatchTopK's inference threshold is learned during training** and
  saved in the checkpoint; a run cancelled very early has seen few
  batches and its threshold (hence inference sparsity) is unreliable.

---

## ➡️ 9. Where to go next

- **The CFM whose embeddings this stage decomposes:**
  [`../training_cfm/README.md`](../training_cfm/README.md)
- **The labeling stage that names the trained neurons:**
  [`../neuron_labeling/README.md`](../neuron_labeling/README.md)
- **The plugin contract** (discovery, `io_spec`, `run()` params):
  [`../README.md`](../README.md)
