# SARSSA — Service Application for Recommender Systems with Sparse Autoencoders

> A research platform for **designing, running, and forensically
> analysing recommender-system pipelines built around Sparse
> Autoencoders (SAE)** — with a focus on interpretability, evaluation,
> and steering of recommendations. It operationalises the research in
> *"From Knots to Knobs: Towards Steerable Collaborative Filtering
> Using Sparse Autoencoders"*
> ([arXiv:2601.11182](https://arxiv.org/abs/2601.11182)), developed
> within the research initiative at Charles University.

---

## 📑 Table of Contents

1. [🎯 What it is & who it's for](#-1-what-it-is--who-its-for)
2. [🏗️ How it works](#-2-how-it-works)
3. [🧩 Extending SARSSA](#-3-extending-sarssa)
4. [📚 Documentation](#-4-documentation)
5. [🔐 Configuration & `.env`](#-5-configuration--env)
6. [🚀 Setup & run](#-6-setup--run)
7. [🗂️ Project layout](#-7-project-layout)
8. [🛠️ Technologies](#-8-technologies)
9. [🧑‍💻 Development](#-9-development)

---

## 🎯 1. What it is & who it's for

### Research background

SARSSA operationalises and generalises the approach introduced in
**"From Knots to Knobs: Towards Steerable Collaborative Filtering
Using Sparse Autoencoders"** - Spišák, Peška, Škoda, Vančura & Alves
([arXiv:2601.11182](https://arxiv.org/abs/2601.11182)) - joint
research from Recombee, Charles University and the Czech Technical
University, supported by Czech Science Foundation (GAČR).

It is the **first** to apply **sparse autoencoders (SAEs)** - a
mechanistic-interpretability technique from LLM research - to
**collaborative filtering**. A CFAE compresses a user's interactions
into a dense embedding; an SAE "hook" inserted between its encoder and
decoder re-expresses that as a sparse set of interpretable **"knobs"**
- neurons that align with human concepts (a *love-story* knob, a
*David Lynch* knob) despite being trained on interactions alone
(metadata only *labels* them). The knobs form a **control panel**:
turn one up or down to **steer** recommendations toward a concept
while largely preserving the base model's accuracy.

SARSSA turns that four-stage method into a **systematic, plugin-based
platform** - stages can be swapped, compared and extended instead of
re-implemented per experiment. The shipped pieces mirror the paper:
the **ELSA** CFAE, **Basic/TopK** SAEs, **TF-IDF** neuron labelling,
and activation-redistribution **steering**, over MovieLens-style
implicit feedback (ratings ≥ 4, min-interaction filtering).

### What it is, concretely

SARSSA is a **modular, reproducible, plugin-based platform** for
experimenting with SAE-enhanced recommender systems. A *pipeline* is a
sequence of steps — load a dataset, train a collaborative-filtering
model, train a sparse autoencoder on its embeddings, label the SAE
neurons with human concepts, then **evaluate / inspect / steer** the
resulting recommendations — every step tracked in MLflow so results
are reproducible and reusable across experiments.

**Who it's for:**

- **Researchers** who want to run SAE-recommender experiments, or —
  the headline use case — **bring their own dataset or method** and
  forensically evaluate the interpretability of embedded sparse
  autoencoders in collaborative-filtering models.
- **Contributors / maintainers** extending the platform (new plugins,
  new providers, new pipeline shapes).

**Two ways to use it:**

- *As-is* — drive the shipped pipeline from the web UI (configure
  cards → run → browse results).
- *Extended* — add your own dataset, model, labeling/steering method,
  or LLM/embedding provider. See [§3](#-3-extending-sarssa).

Key capabilities: plugin-based multi-step pipelines; MLflow experiment
tracking & reproducibility; a web UI for composing, running, and
inspecting pipelines; and reuse of intermediate results from previous
runs (start a new pipeline from an old run's outputs).

---

## 🏗️ 2. How it works

SARSSA is a **web application with three cooperating parts**:

| Part | Role |
|------|------|
| **Backend** (FastAPI, `src/app`) | The orchestrator: discovers plugins, runs pipelines as MLflow runs, exposes the HTTP API. |
| **Frontend** (React, `frontend/`) | The UI: compose pipelines as cards, launch them, watch progress, browse visual results. |
| **MLflow** | Experiment tracking + artifact store. Its **own native web UI is also usable** for browsing past runs, params, and metrics. |

**The pipeline, end to end:**

```
dataset_loading → training_cfm → training_sae → neuron_labeling
                                                      │
                              ┌───────────────────────┼───────────────────────┐
                              ▼                       ▼                       ▼
                       labeling_evaluation        inspection               steering
```

The first four stages build a pipeline once (`one_time`); the last
three are re-run repeatedly against a finished pipeline (`multi_run`).

**Nothing is hardcoded in the frontend.** The UI renders itself from a
contract served by the backend: the backend reads `config.yaml` (the
skeleton — which plugin *categories* exist and their order), then
**dynamically discovers** the plugin modules in each category along
with their parameters and output/display specs, and exposes all of it
at `GET /plugins/registry`. The frontend just renders that registry —
add a backend plugin and it appears in the UI with **zero** frontend
changes.

### MLflow — what it is, and how SARSSA uses it

**MLflow in brief.** [MLflow](https://mlflow.org/) is an open-source
platform for tracking machine-learning experiments. The vocabulary you
need: a **run** is one tracked execution that records **params**
(inputs), **metrics** (numbers), and **artifacts** (output files);
runs live under an **experiment**, can be **nested** (a parent run
with child runs underneath), and are browsable in MLflow's own web UI.
New to it? The [MLflow Tracking guide](https://mlflow.org/docs/latest/tracking.html)
is the place to start; a fuller, SARSSA-specific primer is in the
[backend doc](src/app/README.md).

**How SARSSA uses it.** Every pipeline you launch is **one parent
run**, and **each step is a nested run** under it. A step writes its
outputs as that nested run's **artifacts** (interaction matrices,
trained models, neuron labels, …) and **params**; the parent run
stores a small `context.json` mapping each finished step → its nested
run id. Later steps read upstream artifacts straight from MLflow
through those ids — which is exactly what makes runs **reproducible**
(everything is recorded) and **reusable** (a new pipeline can start
from a previous run's outputs instead of recomputing earlier stages).
MLflow serves at `http://localhost:5000`, and its **native UI is fully
usable on its own** for browsing past runs, their params, metrics and
artifacts — independently of the SARSSA frontend.

For the deeper mental models see the
[backend](src/app/README.md), [plugin-system](src/plugins/README.md),
and [frontend](frontend/README.md) docs (indexed in [§4](#-4-documentation)).

---

## 🧩 3. Extending SARSSA

Extensibility is layered — from the common case to reshaping the
pipeline itself.

### 3.1 Add a plugin — the usual path

**This is how you extend SARSSA 95% of the time.** A plugin is one
folder with one `Plugin` class inside the relevant
`src/plugins/<category>/` directory; it declares a small I/O contract
(`io_spec`) and pure `run()` logic, and the engine + UI pick it up
automatically — no backend or frontend changes. Bringing your own
**dataset**, **CFM/SAE trainer**, or **labeling / inspection /
steering** method is all this. Start with the plugin contract in
[`src/plugins/README.md`](src/plugins/README.md); for the headline
"bring your own dataset" walkthrough see
[`src/plugins/dataset_loading/README.md`](src/plugins/dataset_loading/README.md).

### 3.2 Add a provider or model — `utils/`

The **secondary** extension point. The LLM and text-embedding
providers are pluggable behind small registries — adding a non-OpenAI
provider is a self-contained change in `src/utils`. The torch model
framework is likewise extensible: a new recommender architecture (e.g.
a variational-autoencoder recommender) or a new SAE variant is a new
class + a registry entry. See
[`src/utils/README.md`](src/utils/README.md).

### 3.3 Reshape the pipeline — advanced

You can also change the *shape* of the pipeline, not just fill in its
slots: add an entirely new **category** (a new key in `config.yaml`'s
`plugin_categories` + a matching `src/plugins/<category>/` directory),
or restructure existing ones — e.g. a single category that fuses CFM
and SAE training into one step. The category contract and the
discovery rules are documented in
[`src/plugins/README.md`](src/plugins/README.md) and the backend's
config section ([`src/app/README.md`](src/app/README.md)).

> **Rule of thumb:** plugins first; `utils/` providers/models second;
> reshaping categories only when the *pipeline structure itself* needs
> to change.

---

## 📚 4. Documentation

The docs live next to the code they describe. Each is a focused,
durable reference — start here and dive into whichever matches your
task.

- **[`src/README.md`](src/README.md) — the source-tree map.**
  A 30-second orientation to the Python source: the import-root
  convention (the app runs from `src/`), what each of `app/`,
  `plugins/`, `utils/`, `tests/` is, and the pointer that extension
  happens in `plugins/<category>/`. Read this first if you're new to
  the repo.

- **[`src/app/README.md`](src/app/README.md) — the backend
  architecture.** The FastAPI service in depth: the pipeline execution
  engine (parent/nested MLflow runs, step-by-step vs batch modes),
  async execution with cooperative cancellation and the
  poll-based progress model, plugin discovery & the registry, the
  pipeline *context* object that wires steps together, the full HTTP
  API reference, MLflow integration & `config.yaml`, the data models,
  and a substantial operational-gotchas section. Read this to
  understand or debug pipeline execution.

- **[`src/plugins/README.md`](src/plugins/README.md) — the plugin
  contract.** The core contributor document: `BasePlugin` /
  `BaseComparePlugin`, the declarative `PluginIOSpec` (inputs,
  outputs, display specs, parameter UI hints), the
  `load_context → run → update_context` lifecycle the engine drives,
  the directory/discovery rules, a step-by-step "write your own
  plugin" tutorial (single *and* compare), and gotchas. Read this
  before writing any plugin.

- **[`src/plugins/dataset_loading/README.md`](src/plugins/dataset_loading/README.md)
  — bring your own dataset (headline tutorial).** The `DatasetLoader`
  `prepare()` pipeline, the exact output/artifact contract and file
  encodings downstream steps depend on, the three hard rules for
  `load_ratings()` (implicit feedback only, canonical
  `userId`/`itemId`, `pl.String`), the gitignored `data/` folder
  layout with real `ratings`/`tags`/`metadata` examples, a full
  custom-loader walkthrough, and the critical gotchas (e.g. a tagless
  dataset can't run neuron labeling). Read this to run SARSSA on your
  own data.

- **[`src/utils/README.md`](src/utils/README.md) — shared
  utilities.** The toolbox both backend and plugins build on: the
  batching `DataLoader`; the pluggable **embedder** and **LLM**
  provider registries (the secondary extension point, and the
  `OPENAI_API_KEY` requirement); the **torch** ML framework (device
  runtime, the model registry/loader, the ELSA base recommender, the
  SAE variants, the fused/steered models, the evaluation metrics); and
  `MLflowRunLoader` (the one artifact/param read API) plus the
  logger/notifier. Read this for model code or the MLflow layer.

- **[`frontend/README.md`](frontend/README.md) — the web UI.**
  A high-level, contributor-oriented map: tech stack
  (React/Vite/Tailwind/Zustand), the dev-vs-Docker/nginx build and the
  `5173 ↔ 8000` CORS contract, the folder map, the registry-driven
  data flow to the backend (run → 2 s poll → toasts), a
  "change X — where?" guide, and gotchas. Note this is the *uncommon*
  path — the FE is registry-driven, so most extension needs no FE
  changes.

> The per-category plugin docs (CFM/SAE training, neuron labeling,
> labeling-evaluation, inspection, steering) are forthcoming; until
> then the [plugin-system doc](src/plugins/README.md) is their
> umbrella contract.

---

## 🔐 5. Configuration & `.env`

Runtime secrets are supplied via a **`.env` file in the project
root**, loaded automatically (`python-dotenv`). It is **gitignored** —
never commit it.

**[`.env.sample`](.env.sample) is the committed template and the
single source of truth for required environment variables.** Today
there is exactly one:

| Variable | Needed for |
|----------|-----------|
| `OPENAI_API_KEY` | The OpenAI **embedding** and **chat-LLM** providers (`src/utils`). Required by any embedder/LLM feature — currently the `labeling_evaluation` embedding plugins. Without it those calls fail at request time. |

Set it up before running anything that touches embeddings/LLMs:

```bash
cp .env.sample .env      # then edit .env and set OPENAI_API_KEY
```

> Model names (e.g. `text-embedding-3-small`, `gpt-4o-mini`) are
> **plugin parameters**, *not* environment variables — only the API
> key lives in `.env`.

---

## 🚀 6. Setup & run

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — Python package/venv manager
- **[Node.js](https://nodejs.org/) 18+ & npm** — for the frontend
- **[just](https://github.com/casey/just)** — command runner
  (recommended; every recipe has a manual equivalent)
- *(optional)* **Docker** with the Compose plugin — for the
  containerised stack

### Quick start (local) — in order

```bash
# 1. Dependencies (--frozen = install the exact locked versions)
uv sync --frozen
just frontend-install                 # cd frontend && npm install

# 2. Credentials (see §5)
cp .env.sample .env                   # then set OPENAI_API_KEY

# 3. Datasets — the gitignored data/ folder is built locally.
#    Prefer the OSF "all" bundles: they include ratings + tags +
#    metadata (full pipeline + UI), unlike the raw download_*_dataset.sh.
bash scripts/download_movieLens_all.sh
bash scripts/download_lastFm1k_all.sh

# 4. Run the stack (three terminals, or use Docker — see below)
just run                              # backend  → http://localhost:8000
just mlflow                           # MLflow UI → http://localhost:5000
just frontend-dev                     # frontend → http://localhost:5173
```

Then open **http://localhost:5173**. A pipeline that starts with a
dataset-loading step needs step 3 done first, or it fails with
`FileNotFoundError`.

### Run with Docker

The whole stack is containerised with Docker Compose:

| Service | Description | URL |
|---------|-------------|-----|
| `backend` | FastAPI app (uvicorn) | http://localhost:8000 |
| `mlflow` | MLflow tracking + UI | http://localhost:5000 |
| `frontend` | React build served by nginx | http://localhost:5173 |

**Do the host-side setup first.** Docker changes *how* the services
run, not *what* they need — so credentials, datasets and the
bind-mount targets must exist on the host **before** the first build
(same as the local Quick start, §6):

```bash
cp .env.sample .env                       # then set OPENAI_API_KEY (§5)
bash scripts/download_movieLens_all.sh    # datasets (Quick start, step 3)
bash scripts/download_lastFm1k_all.sh
mkdir -p src/mlartifacts data && touch src/mlflow.db   # bind-mount targets
```

Then build and start the stack:

```bash
docker compose up --build -d          # or: just docker-up
```

**GPU acceleration (optional).** The backend trains models on a GPU when
one is available. By default the stack runs **CPU-only** (`GPU_COUNT=0`),
so it starts on any host. To use an NVIDIA GPU you need the **NVIDIA
driver** and the **NVIDIA Container Toolkit** installed on the host, then
start the stack with `GPU_COUNT` set:

```bash
GPU_COUNT=all docker compose up --build -d     # all GPUs (or e.g. GPU_COUNT=1)
```

Alternatively, add `GPU_COUNT=all` to your `.env` file (Compose reads it
automatically).

Verify the container sees it:

```bash
docker exec sarssa-backend-1 python -c "import torch; print(torch.cuda.is_available())"
```

Without a GPU (or with `GPU_COUNT=0`) training still runs — just much
slower on CPU.

**Launching several pipelines.** Compute tasks run **one at a time**:
each launch is accepted immediately, but waits in a FIFO queue until the
previous task finishes, then starts automatically. The header's
*Running* menu lists every task — the executing one and any queued ones
(clock icon). Serialising the runs keeps them reproducible (no shared
RNG or GPU contention) and means two trainings can never overlap into a
`CUDA out of memory`. The UI tracks **one active run per browser tab**,
so to launch a second run alongside a first, open the app in **another
browser tab** and start it there; a queued run can be cancelled
instantly from either tab.

Open the UI at **http://localhost:5173** — use this exact port: the
backend's CORS allow-list and the frontend's hardcoded API URL expect
frontend on `:5173`, backend on `:8000`.

Stop the stack:

```bash
docker compose down                   # or: just docker-down
```

(Add `-v` to also drop the named volumes. Your `data/`,
`src/mlartifacts/` and `src/mlflow.db` are host bind mounts, so they
survive `down` regardless.)

Notes:

- The root `.env` is read at runtime — never baked into the image.
- `data/`, `src/mlartifacts/` and `src/mlflow.db` are bind-mounted;
  the `mlflow` service reads the **same** files the backend writes,
  exactly like running `just run` + `just mlflow` locally.
- Other commands: `just docker-build` / `docker-down` / `docker-logs`.

> **Image size:** `pyproject.toml` pins `torch==2.7.1`, which the
> lockfile resolves to the CUDA build, so the backend image is several
> GB. It runs fine CPU-only; switch `torch` to a CPU wheel index for a
> smaller image.

### Common commands

`just` (run with no arguments to list all) wraps the everyday tasks:

```
just run / mlflow / frontend-dev      # run the services
just check                            # lint + type-check
just fix                              # format + autofix
just pre-commit                       # run all pre-commit hooks
just download-movielens / -lastfm     # raw datasets (ratings/tags only)
bash scripts/download_*_all.sh        # full OSF bundles (recommended)
```

Manual equivalents (no `just`): `uv sync --frozen`,
`cd src && uv run uvicorn app.main:app --reload`,
`uv run ruff format .`, `uv run ruff check --fix .`, `uv run ty`,
`uv run pre-commit run --all-files`.

---

## 🗂️ 7. Project layout

```
SARSSA/
├── src/
│   ├── app/        FastAPI backend + pipeline engine   → src/app/README.md
│   ├── plugins/    the plugin system + categories       → src/plugins/README.md
│   ├── utils/      shared ML / MLflow / provider utils   → src/utils/README.md
│   └── tests/      pytest suite (mirrors the tree)
│                                                         (overview: src/README.md)
├── frontend/       React/Vite web UI                     → frontend/README.md
├── scripts/        dataset download scripts (raw + _all OSF bundles)
├── data/           datasets (gitignored; built locally)
├── justfile        task runner (run with no args to list)
├── docker-compose.yml / Dockerfile.backend
├── pyproject.toml  deps + tooling config
└── .env.sample     env template (single source of truth — see §5)
```

Each `README.md` documents the folder it sits in; the table in
[§4](#-4-documentation) says when to read which.

---

## 🛠️ 8. Technologies

- **Backend:** FastAPI + uvicorn, MLflow (tracking + UI), PyTorch,
  polars, scipy/numpy, langchain-openai, python-dotenv.
- **Frontend:** React 19, Vite, Tailwind CSS 4, Zustand,
  react-router, sonner — built and served by nginx in Docker.

The development tooling (uv, ruff, ty, pytest, pre-commit, just) is
covered in [§9](#-9-development).

---

## 🧑‍💻 9. Development

For working **on** SARSSA, not just running it.

**Setup.** Install deps and the git hooks once:

```bash
uv sync --frozen
just install-hooks                    # uv run pre-commit install
```

**Everyday workflow** goes through the `justfile` (run `just` with no
arguments to list every recipe):

| Task | Command |
|------|---------|
| Format + autofix | `just fix` (ruff) |
| Lint + type-check | `just check` (ruff + ty) |
| Unit tests | `just test-unit` |
| Integration tests | `just test-integration` |
| All pre-commit hooks | `just pre-commit` |

**Testing conventions.** Tests live in `src/tests/` and **mirror the
source tree** — `tests/app`, `tests/plugins`, `tests/utils`. pytest is
configured in `pyproject.toml` (`testpaths = src/tests`,
`pythonpath = src`), so `uv run pytest` works from the repo root;
async/API tests use `pytest-asyncio` + `httpx`. Plugins are designed
to be unit-testable in isolation — a plugin's pure `run()` can be
exercised without a live pipeline (the notifier degrades to a silent
no-op off-pipeline; see the [plugin doc](src/plugins/README.md)).

**The pre-commit gate.** Every commit runs **ruff** (lint + format),
**ty** (type check), and the full **pytest** suite, plus
whitespace / end-of-file / large-file checks — a commit only lands if
they all pass. Run them on demand with `just pre-commit` (or
`just check` for lint + types only).
