# 🧰 SARSSA Shared Utilities (`src/utils`)

> **What this is:** the shared, provider-agnostic infrastructure that
> both the backend (`src/app`) and the plugins (`src/plugins`) build
> on — MLflow access, the ML model framework, pluggable LLM/embedder
> providers, batching, logging, and the UI notifier. It is **not**
> domain logic; it is the toolbox the rest of the codebase calls.
>
> **Who should read this:** contributors touching model code, the
> MLflow layer, or the LLM/embedder providers. `utils/` is also a
> **secondary extension point** — the *primary* one is plugins (see
> [`../plugins/README.md`](../plugins/README.md)), but adding a new
> LLM/embedder provider or a new torch model happens **here**.

---

## 📑 Table of Contents

1. [🗺️ Orientation](#-1-orientation)
2. [📦 `data_loading/`](#-2-data_loading)
3. [🔤 `embedder/`](#-3-embedder)
4. [💬 `llm/`](#-4-llm)
5. [🔥 `torch/` — the ML model framework](#-5-torch--the-ml-model-framework)
6. [🗃️ `mlflow_manager.py`](#-6-mlflow_managerpy)
7. [🪵 `plugin_logger.py` & `plugin_notifier.py`](#-7-plugin_loggerpy--plugin_notifierpy)
8. [⚠️ Gotchas](#-8-gotchas)
9. [➡️ Where to go next](#-9-where-to-go-next)

---

## 🗺️ 1. Orientation

`utils/` has no single entry point — it is a set of independent
toolboxes. Quick map of what lives here and who calls it:

| Module | Purpose | Main consumers |
|--------|---------|----------------|
| `data_loading/` | Mini-batch iterator over interaction matrices | trainers, `torch/evaluation.py` |
| `embedder/` | Pluggable **text-embedding** providers | `labeling_evaluation` plugins |
| `llm/` | Pluggable **chat-LLM** providers | *(none yet — intended mainly for `labeling_evaluation`)* |
| `torch/` | The whole ML model framework (base / SAE / steered models, training metrics) | `training_cfm`, `training_sae`, `neuron_labeling`, `steering`, plugin loaders |
| `mlflow_manager.py` | The artifact/param **read** API for MLflow | `BasePlugin` loaders, backend `routes_items` / `item_enrichment` |
| `plugin_logger.py` | Shared stdout logger | ~everything (16+ call sites) |
| `plugin_notifier.py` | Plugin → UI message channel | pipeline worker, plugins |

> **🔑 Credentials.** The embedder and LLM providers shipped today are
> **OpenAI only**, and they read `OPENAI_API_KEY` from the environment
> (loaded from a repo-root `.env` via `python-dotenv`). Copy the
> committed template and fill in your key — `cp .env.sample .env` —
> then set `OPENAI_API_KEY`. **Any feature that uses an embedder or
> LLM requires it** — today the `labeling_evaluation` embedding
> plugins; without it those calls fail at request time. (`.env` is
> gitignored; [`.env.sample`](../../.env.sample) is the tracked
> template and the single source of truth for required env vars.)

---

## 📦 2. `data_loading/`

One file, `data_loader.py` → **`DataLoader`**: a tiny iterator that
yields fixed-size batches of a SciPy CSR matrix (or ndarray) as torch
tensors on a device, with optional shuffling. Densifies each CSR batch
(`.toarray()`) lazily, one batch at a time, so a huge sparse matrix
never has to be densified whole.

**Where it's used:** `training_cfm/elsa_trainer`,
`training_sae/sae_trainer`, and `torch/evaluation.py` — i.e. the
training/eval inner loop.

> ⚠️ This is **not** the dataset-loading *plugin*. It's a training-time
> batcher; the name overlap is purely incidental (see the note in
> [`../plugins/dataset_loading/README.md`](../plugins/dataset_loading/README.md)).

---

## 🔤 3. `embedder/`

Text-embedding behind a provider abstraction:

| File | Role |
|------|------|
| `embedder.py` | `EmbeddingLLM` — the ABC: `generate_embedding(text)` and `generate_embeddings(texts)` |
| `registry.py` | `_PROVIDERS` (name → class), `create_embedder(provider, model)` factory, `known_providers()` |
| `openai_embedder/` | `OpenAIEmbeddingLLM` — wraps `langchain_openai.OpenAIEmbeddings`; reads `OPENAI_API_KEY` |

Callers never instantiate a provider directly — they call
`create_embedder("openai", "text-embedding-3-small")` and get back an
`EmbeddingLLM`. The provider name maps through `_PROVIDERS`.

**Where it's used:** the `labeling_evaluation` category — its
`_embedding_cache.py` calls `create_embedder`, feeding the
embedding-map and nearest-label-distance plugins (single + compare)
that judge label quality in embedding space.

**Extensible.** This is deliberately a registry behind an interface —
a non-OpenAI embedding provider can be added here without touching any
caller. The concrete how-to is documented within the module itself;
this README only flags that this is an extension point.

---

## 💬 4. `llm/`

Structurally a mirror of `embedder/`, for chat completions:

| File | Role |
|------|------|
| `llm.py` | `ChatLLM` — the ABC: `generate_response(prompt)` |
| `registry.py` | `create_chat_llm(provider, model, temperature, max_tokens)`, `known_providers()` |
| `openai_llm/` | `OpenAIChatLLM` — wraps `langchain_openai.ChatOpenAI`; reads `OPENAI_API_KEY` |

Unlike the embedder registry there is **no caching layer** — chat
generation is non-deterministic and prompts rarely repeat, so caching
would mostly waste memory.

**Where it's used:** **nowhere in production yet.** `llm/` is fully
wired and unit-tested, but no shipped plugin or backend component
currently calls `create_chat_llm`. Treat it as a ready-to-use,
extensible module that simply has no consumer at the moment —
accurate to the code as it stands.

**Extensible.** Same shape as the embedder extension point (add a
provider without touching callers); the concrete how-to lives in the
module itself.

---

## 🔥 5. `torch/` — the ML model framework

The largest and most important part of `utils/`. Everything the
pipeline trains, loads, evaluates, or steers lives here.

### `runtime.py`
`set_seed(seed)` (seeds torch / numpy / random / CUDA / MPS) and
`set_device()` (CUDA → MPS → CPU). Called by every trainer and by
`neuron_labeling` to make runs reproducible and device-aware.

### `models/model_registry.py` — the registry
**Decorator-based** registries: `@register_base_model(name)` →
`BASE_MODEL_REGISTRY`, `@register_sae_model(name)` → `SAE_REGISTRY`,
read back via `get_base_model_class` / `get_sae_model_class`. A class
registers **only when its module is imported** — which is why
`model_loader.py` explicitly imports every model module (see the
gotcha in §8).

### `models/model_loader.py` — the loader
`load_base_model(path)` / `load_sae_model(path)`: read a
`config.json` (`model_type` + `architecture`) and a `model.pt`
(`state_dict`) from an artifact directory, rebuild the class via the
registry, load weights, return it in `.eval()`. **This is the
read-side counterpart of the plugin `"model"` saver** documented in
[`../plugins/README.md`](../plugins/README.md) — it is exactly what
the `base_model` / `sae_model` plugin loader strategies call.

### `models/base_model/` — the recommender itself
**This is the recommender model** — the collaborative-filtering model
that actually produces the item recommendations. `BaseModel` is the
interface every recommender implements: `encode` / `decode` /
**`recommend`** (the method that returns the ranked items) /
`get_config` / `normalize_relevance_scores`. The one shipped
implementation is `elsa/elsa.py` → **`ELSA`**
(`@register_base_model("elsa")`), a linear shallow autoencoder CFM —
the model `training_cfm` trains. **Used by:** `training_cfm` (trains
ELSA), `training_sae` (consumes the frozen base via the `BaseModel`
interface), `model_loader` (reconstruction).

*Extensible:* a different recommender architecture — say a
**variational autoencoder for recommendation** — would be a new
`BaseModel` subclass with a `@register_base_model("…")` entry and
nothing else changed. ELSA is simply the only one implemented today.

### `models/sae_model/` — the sparse autoencoders
`sae_model.py` → abstract **`SAE`**: encoder/decoder, input
standardisation, the loss kit (L1/L2/cosine + auxiliary
dead-neuron-revival + contrastive), dead-neuron tracking; subclasses
must implement `post_process_embedding` and `total_loss`. Three
concrete variants, each registered:
- `basic_sae/` → **`BasicSAE`** (`"BasicSAE"`) — plain L1-sparsity SAE.
- `topk_sae/` → **`TopKSAE`** (`"TopKSAE"`) — per-sample top-k activation.
- `batch_topk_sae/` → **`BatchTopKSAE`** (`"BatchTopKSAE"`) — batch-level top-k with a learned inference threshold.

**Used by:** `training_sae` (trains the chosen variant via
`get_sae_model_class`); `neuron_labeling` / `steering` load a trained
SAE through `model_loader`.

*Extensible:* a new sparsity scheme is a new `SAE` subclass
implementing `post_process_embedding` / `total_loss` with a
`@register_sae_model("…")` entry — the three above are siblings, not a
closed set.

### `models/fused_model/` — base ⊕ SAE
`FusedModel` composes a frozen base model and an SAE
(`encode → SAE → decode`) to produce recommendations *through* the
SAE bottleneck. **Used internally** by
`evaluation.evaluate_sparse_encoder` to measure how much the SAE
degrades recommendation quality; no plugin imports it directly.

### `models/steered_model/` — concept steering
`SteeredModel` wraps a frozen base model + SAE and applies
**neuron-level activation-redistribution steering**: rescale a user's
SAE activations, inject a budget `alpha` into the target neuron(s),
then restore the original total magnitude — shifting recommendations
toward a concept while preserving activation scale. **Used by:**
`steering/_steer.py` (the engine behind the steering plugins).

### `evaluation.py`
The metrics module: Recall@K / NDCG@K kernels plus three entry points
the plugins call —
- `evaluate_dense_encoder` → `{R20, NDCG20}` — used by `training_cfm`.
- `evaluate_sparse_encoder` → reconstruction (cosine), sparsity (L0,
  dead neurons), and recall/NDCG **degradation vs the base model** —
  used by `training_sae`.
- `compute_sae_item_activations` → an `(items × neurons)` activation
  matrix from one-hot item encodings — used by `neuron_labeling/tf_idf`
  to produce its `item_acts.pt`.

**The whole framework is one big extension point.** Each model
*type* above is independently extendable (see the per-type
*Extensible* notes); the registry + `model_loader` is the single
mechanism — a new class, a `@register_*` entry, and (for loadable
models) the matching `config.json` / `model.pt` round-trip. A more ambitious case — a
torch model that builds the SAE **inherently into the recommender**
instead of consuming a separate frozen one — also lands here, not in
a plugin. The point: model architecture is a `utils/torch/`
extension, never a plugin one.

---

## 🗃️ 6. `mlflow_manager.py`

`MLflowRunLoader(run_id)` — the **single, storage-agnostic API for
reading an MLflow run**: `get_json_artifact`, `get_npy_artifact`
(optional `allow_pickle`), `get_npz_artifact` (SciPy sparse by
default), `download_artifact` / `download_artifact_dir`,
`get_parameter(s)` / `get_metric(s)`, `artifact_exists`; the `run`
object is fetched once and cached. Module-level convenience wrappers
mirror the methods. It uses `mlflow.artifacts.download_artifacts`, so
callers don't care whether artifacts sit on local disk or elsewhere.

**Where it's used:** this is the foundation the **plugin
`BasePlugin` loaders/savers** are built on (the loader/saver strategy
table in [`../plugins/README.md`](../plugins/README.md) sits directly
on top of it), plus the backend's artifact proxy
(`app/api/routes_items`), item enrichment
(`app/core/item_enrichment`), and `BaseComparePlugin`'s past-run
access. If you change how artifacts are read, change it here.

---

## 🪵 7. `plugin_logger.py` & `plugin_notifier.py`

Two superficially similar things with **opposite audiences** — keep
them straight.

**`plugin_logger.py` — for developers.** `get_logger(name)` is a thin
stdlib-`logging` wrapper (INFO → stdout). Its purpose is **developer
logging / debugging** — tracing what the code is doing in the console
while building or diagnosing the app. It is *not* shown to the end
user. (Distinct from the backend's `sarssa` file logger,
`src/app/utils/logger.py` — this one is the plain console logger
plugins and utils use.)

**`plugin_notifier.py` — for the UI.** This exists for one concrete
reason: **pipeline steps can run for a long time** (training a model,
computing activations…), and a user watching the screen needs to know
*which step is running and where in the process it is*. Without this
channel the UI would have nothing to show — a run would just look
frozen. So a plugin calls `self.notifier.info(...)` / `warning` /
`success` / `error` to report progress, and:
- `NotificationMessage` — a `{timestamp, level, text}` record.
- `PluginNotifier` — those calls append to a `messages` list the
  pipeline worker **aliases into** `TaskState.messages`, so the
  polling endpoint (and ultimately the frontend's toasts) sees new
  messages immediately, with no copying.
- `NullNotifier` — a no-op used in tests/scripts so the same plugin
  code runs unchanged off-pipeline.

This is the source end of the progress-toast flow described from the
other side in [`../app/README.md`](../app/README.md) and
[`../../frontend/README.md`](../../frontend/README.md).

---

## ⚠️ 8. Gotchas

- **`OPENAI_API_KEY` is mandatory for embedder/LLM features.** OpenAI
  is the only provider for *both* `embedder/` and `llm/`; the key is
  read from the environment / a `.env` file. No key → the
  `labeling_evaluation` embedding plugins fail at request time.
- **Two different registry styles.** `embedder/` and `llm/` use an
  explicit `_PROVIDERS` dict + a `create_*` factory. `torch/` models
  use **decorator registration** that only takes effect once the
  module is imported — which is why `model_loader.py` imports every
  model module. A new torch model that nothing imports will be
  invisible to the registry.
- **`MLflowRunLoader` is the one MLflow read path.** The plugin
  loader/saver strategies are a thin layer over it — document/extend
  loaders there, not by re-implementing MLflow access.

---

## ➡️ 9. Where to go next

- 🔌 **The primary extension surface:**
  [`../plugins/README.md`](../plugins/README.md) — the plugin contract
  (the loader/saver strategies here back its `io_spec`).
- ⚙️ **Who orchestrates these utils:**
  [`../app/README.md`](../app/README.md) — the backend, the pipeline
  engine, the async task model the notifier feeds.
- 📂 **Where this sits in the tree:** [`../README.md`](../README.md).
- 📘 **Project overview, setup & Docker:**
  [root `README.md`](../../README.md).
