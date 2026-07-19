# 🧠 Training CFM (`src/plugins/training_cfm`)

> **What this is:** the second stage of a SARSSA pipeline — it trains
> the **collaborative-filtering model (CFM)**, the base recommender
> whose dense user-embedding space everything downstream lives in: the
> SAE decomposes it, neuron labeling explains it, steering nudges
> vectors inside it.
>
> **Who should read this:** anyone tuning the base recommender's
> hyperparameters, anyone interpreting its training metrics in MLflow,
> and contributors adding an alternative CFM architecture (§7). The
> plugin contract in general lives in [`../README.md`](../README.md);
> this doc is the CFM-training specifics.

---

## 📑 Table of Contents

1. [🗺️ Big picture](#-1-big-picture)
2. [🧮 The ELSA model](#-2-the-elsa-model)
3. [🔁 The training loop](#-3-the-training-loop)
4. [📥 Inputs · 📤 Outputs](#-4-inputs--outputs)
5. [🎛️ `run()` parameters](#-5-run-parameters)
6. [💾 Model persistence & the model registry](#-6-model-persistence--the-model-registry)
7. [🛠️ Adding your own CFM trainer](#-7-adding-your-own-cfm-trainer)
8. [⚠️ Operational notes & gotchas](#-8-operational-notes--gotchas)
9. [➡️ Where to go next](#-9-where-to-go-next)

---

## 🗺️ 1. Big picture

CFM training is the **second stage** of a pipeline: it turns the `dataset_loading` matrices into a
**trained base recommender** that maps a user's interaction row to a
**dense embedding**. That embedding space is what the rest of the
pipeline works in — `training_sae` re-expresses it sparsely,
`neuron_labeling` names the sparse directions, `steering` shifts
vectors along them.

One plugin ships today: **`elsa_trainer`**. The stage is deliberately
thin — the model lives in `utils/torch/models/base_model/elsa/`,
generic training utilities in `utils/torch/`, and the plugin wires
them together:

```
dataset_loading artifacts          elsa_trainer plugin
  train/valid/test_csr.npz   ──▶     train() loop            ──▶  MLflow:
                                       │ ELSA model                 config.json + model.pt
                                       │ evaluate_dense_encoder     loss/R20/NDCG20 metrics
                                       ▼
                             utils/torch/{models,evaluation,runtime}
```

---

## 🧮 2. The ELSA model

**ELSA** — *Scalable Linear Shallow Autoencoder*
([RecSys '22 paper](https://dl.acm.org/doi/abs/10.1145/3523227.3551482))
— is about the simplest architecture that works well on implicit
feedback, which is exactly why it's a good interpretability substrate.
Implementation: `utils/torch/models/base_model/elsa/elsa.py`.

The entire model is **one parameter matrix** `A` of shape
`(num_items, factors)` whose rows are kept **L2-normalized** — each
item is a point on the unit sphere.

- **encode**: `e = x @ A` — a user's dense embedding is the *sum of
  the item embeddings they interacted with*.
- **decode**: `ŝ = e @ Aᵀ` — score every item by dot product with the
  user embedding.
- **forward**: `ReLU(x A Aᵀ − x)` — reconstruct the interaction row,
  subtract the input (so the model is *not* rewarded for trivially
  predicting what the user already has), clip negatives.

**Loss** — `normalized_mse_loss`: both the prediction and the target
row are L2-normalized, then it's the squared distance summed over
items, averaged over the batch:

```
L = mean_over_batch( ‖ ŷ/‖ŷ‖ − y/‖y‖ ‖² )
```

Since both vectors are unit-length this equals `2 − 2·cos(ŷ, y)` —
i.e. **cosine-similarity loss**: only the *direction* of the
reconstruction matters, not its magnitude.

**Recommendation** (`recommend()`): score `= decode(encode(x)) − x`,
min-max normalized per user to [0, 1], already-interacted items
masked to 0, then top-k. This same method later serves the steering
and inspection stages.

---

## 🔁 3. The training loop

`elsa_trainer.py :: train()` is a standard early-stopping loop with a
few deliberate choices worth knowing:

| Aspect | What happens |
|---|---|
| **Batching** | `utils.data_loading.DataLoader` slices the CSR matrix, densifies each slice, and moves it to the device. Training shuffles per epoch (seeded RNG); validation doesn't. |
| **Train metric** | Per-epoch mean of the batch losses → `loss/train`. |
| **Validation, part 1 — ranking** | `evaluate_dense_encoder` hides `target_ratio` of each val user's interactions, predicts them from the rest, and logs `R20/valid` + `NDCG20/valid`. **Logged only — never drives model selection.** The split uses a fixed seed, so it is identical every epoch (curves are comparable). |
| **Validation, part 2 — reconstruction** | The model's own loss on the **full** val matrix (input = target), aggregated as a sample-weighted mean over users → `loss/valid`. **This is the model-selection signal.** |
| **Early stopping** | If `loss/valid` hasn't improved for `early_stop` consecutive epochs, stop; the best model + optimizer (deep-copied at their best epoch) are restored. `early_stop=0` disables it and runs all epochs. |
| **Test** | Once, on the restored best model: `R20/test`, `NDCG20/test`. |
| **Progress** | Epoch summaries go to the UI as toasts (`PluginNotifier`); early stop and the final test result too. |
| **Cancellation** | The cancellation token is checked **every training batch**, so *Cancel now* interrupts within seconds even mid-epoch. |

Model selection on **reconstruction loss** rather than NDCG is
deliberate: the ranking metrics on a small val split are noisy, and
the loss is what the optimizer actually minimises — the ranking
metrics are there so you can *see* whether loss improvements still
translate into recommendation quality.

---

## 📥 4. Inputs · 📤 Outputs

Declared in `io_spec` (see [`../README.md`](../README.md) for how the
engine materialises these):

**Inputs** — requires a `dataset_loading` step in the context:

| From | Artifact / param | Becomes |
|---|---|---|
| `dataset_loading` | `train_csr.npz`, `valid_csr.npz`, `test_csr.npz` | the three user×item CSR splits |
| `dataset_loading` | `num_users`, `num_items`, `min_user/item_interactions`, `dataset_name`, `val_ratio`, `test_ratio` | model sizing (`num_items` → input dim) + provenance params |

**Outputs**:

| Kind | Name | Content |
|---|---|---|
| artifact | `config.json` + `model.pt` | the trained model, written by the `"model"` saver (§6) at the artifact root |
| params | `model`, `dataset`, `users`, `items`, `val_ratio`, `test_ratio`, `min_user_interactions`, `min_item_interactions` | model name (`ELSA`) + dataset provenance re-logged so a CFM run is self-describing |
| metrics | `loss/train`, `loss/valid`, `R20/valid`, `NDCG20/valid` (per epoch); `R20/test`, `NDCG20/test` (final) | browsable as curves in the MLflow UI (*Model metrics* tab) |

**Who consumes the model:** every later stage. `training_sae` loads
it (input spec `("training_cfm", "", "base_model", "base_model")`) to
produce the embeddings the SAE trains on; inspection and steering
load it to encode users and score items.

---

## 🎛️ 5. `run()` parameters

Grouped in the UI exactly as declared in `param_groups`:

| Group | Param | Default | Meaning |
|---|---|---|---|
| Architecture | `factors` | 1024 | embedding dimensionality of `A` — the width of the space the SAE will later decompose |
| Training loop | `epochs` | 25 | max epochs (early stopping may end sooner) |
| | `batch_size` | 512 | users per gradient update |
| | `early_stop` | 10 | patience in epochs on `loss/valid`; 0 disables |
| | `seed` | 42 | seeds torch/numpy/python (`set_seed`) for reproducible weights + shuffling |
| Optimizer | `lr` | 3e-4 | Adam step size |
| | `beta1`, `beta2` | 0.9, 0.99 | Adam moment decays |
| Evaluation | `target_ratio` | 0.2 | fraction of interactions hidden as ranking targets in val/test scoring — affects the logged metrics only, **not** model selection |

---

## 💾 6. Model persistence & the model registry

The trained model is declared as
`OutputArtifactSpec("trained_model", "", "model")` — the special
**`"model"` saver** (at most one per plugin, `filename` must stay
`""`). It writes two files at the artifact root:

- **`config.json`** — from `model.get_config()`: the `model_type`
  (`"elsa"`) plus the constructor kwargs
  (`{"input_dim": …, "embedding_dim": …}`) needed to rebuild the
  architecture.
- **`model.pt`** — the `state_dict` weights.

Loading is the mirror image (`utils/torch/models/model_loader.py`):
`load_base_model(path)` reads `config.json`, looks the type up in the
**model registry** (`model_registry.py`, populated by the
`@register_base_model("elsa")` decorator), instantiates the class
from the config kwargs, and loads the weights. No pickled classes —
checkpoints stay portable across code versions as long as the
constructor signature holds.

> ⚠️ Registration happens as an *import side effect*: the decorator
> only runs when the model's module is imported, and `model_loader.py`
> imports every known model module at its top for exactly this reason.
> A new model that isn't imported there will fail to load with
> `Unknown base model type` even though its decorator looks right.

---

## 🛠️ 7. Adding your own CFM trainer

Two pieces: a **model class** and a **trainer plugin**.

1. **Model** — `utils/torch/models/base_model/<name>/<name>.py`,
   subclassing `BaseModel` and implementing the four-method
   interface: `encode(x)`, `decode(e)`,
   `recommend(batch, k, mask_interactions, mask)`, `get_config()`.
   Decorate it with `@register_base_model("<type>")` **and add the
   import to `model_loader.py`** (see the box in §6). Downstream
   expectations: `encode` must map an interaction row to a dense
   embedding (the SAE trains on its output), `decode` + `recommend`
   must score items (steering fuses your model with the SAE and calls
   them).
2. **Plugin** — `src/plugins/training_cfm/<name>_trainer/` (folder
   name = module name, class literally named `Plugin` — discovery
   rules in [`../README.md`](../README.md)). Model
   `elsa_trainer.py`: declare the same dataset inputs, expose your
   hyperparameters as `Annotated` `run()` params, and output the
   checkpoint via a single `"model"` saver plus a `model` output
   param carrying the model name. Reuse `train()`'s shape — or at
   least its habits: check the cancellation token per batch, send
   epoch notifications, log `loss/…` metrics per epoch.

Anything downstream will work unchanged: the SAE trainers and the
steering/inspection stages only ever touch the checkpoint through
`load_base_model` and the four-method interface.

---

## ⚠️ 8. Operational notes & gotchas

- **Device is auto-detected** (`set_device`): CUDA → MPS → CPU. There
  is no device parameter — on a GPU-less host you silently train on
  CPU. ELSA on CPU is workable (minutes on MovieLens-sized data,
  longer on bigger catalogues); it is the later SAE stage that hurts
  more without a GPU.
- **`target_ratio` is evaluation-only.** Changing it changes the
  logged R@20/NDCG@20, not what gets trained or selected — don't tune
  it expecting a different model.

---

## ➡️ 9. Where to go next

- **The SAE that decomposes this model's embeddings:**
  [`../training_sae/README.md`](../training_sae/README.md)
- **The plugin contract** (discovery, `io_spec`, `run()` params):
  [`../README.md`](../README.md)
- **The dataset stage that feeds this one:**
  [`../dataset_loading/README.md`](../dataset_loading/README.md)
