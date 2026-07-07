# 🧩 SARSSA Backend (`src/app`)

> **What this is:** the FastAPI service that turns a list of plugin steps
> into an MLflow‑tracked recommender‑system experiment, and exposes the
> results to the web UI.
>
> **Who should read this:** contributors who need a mental model of the
> backend, anyone debugging pipeline execution, and plugin authors who
> want to understand *how* their plugin is driven. Plugin **internals**
> live in [`../plugins/README.md`](../plugins/README.md); the UI side
> lives in [`../../frontend/README.md`](../../frontend/README.md).

---

## 📑 Table of Contents

1. [🗺️ The big picture](#-1-the-big-picture)
2. [📁 Directory map](#-2-directory-map)
3. [🚀 Running the backend](#-3-running-the-backend)
4. [🛰️ HTTP API reference](#-4-http-api-reference)
5. [⚙️ The pipeline execution engine](#-5-the-pipeline-execution-engine)
6. [🔗 The pipeline context object](#-6-the-pipeline-context-object)
7. [🔌 Engine ↔ plugin contract](#-7-engine--plugin-contract)
8. [🧭 Plugin discovery & the registry](#-8-plugin-discovery--the-registry)
9. [🗃️ MLflow integration & configuration](#-9-mlflow-integration--configuration)
10. [🧱 Data models](#-10-data-models)
11. [⚠️ Operational notes & gotchas](#-11-operational-notes--gotchas)
12. [🔭 Where to go next](#-12-where-to-go-next)

---

## 🗺️ 1. The big picture

The backend is a thin **orchestration layer**. It does not contain
recommender‑system logic itself — that lives in plugins
(`src/plugins`) and shared ML utilities (`src/utils`). The backend's
job is to:

- discover the available plugins and describe them to the UI,
- run a chosen sequence of plugins as one **pipeline**,
- record every step in **MLflow** so results are reproducible and
  reusable, and
- proxy stored results back to the frontend.

```
                          ┌─────────────────────────────┐
  React UI  ──HTTP──▶     │        FastAPI app          │
 (localhost:5173)         │         (main.py)           │
                          │ /pipelines  /plugins  /items│
                          └───────────┬─────────────────┘
                                      │
                ┌─────────────────────┼──────────────────────┐
                ▼                     ▼                       ▼
        core/pipeline_engine   core/plugin_discovery   core/item_enrichment
        (run plugins as         (find plugins, build    (join item ids with
         MLflow runs)            the UI registry)        dataset metadata)
                │                                            │
                ▼                                            ▼
            src/plugins  ◀── load_context / run /        MLflow artifact
            (the actual      update_context                store (mlartifacts,
             experiment      lifecycle                     mlflow.db)
             logic)
```

A request to run a pipeline becomes a **parent MLflow run** with one
**nested run per plugin step**; the wiring between steps is a small
JSON document called the **context** (see §6).

---

## 📁 2. Directory map

| Path | Responsibility |
|------|----------------|
| `main.py` | FastAPI app creation, CORS, MLflow bootstrap, router mounting |
| `api/` | HTTP endpoints (`routes_pipelines`, `routes_plugins`, `routes_items`) |
| `core/pipeline_engine.py` | Runs plugin steps inside MLflow parent/nested runs |
| `core/pipeline_worker.py` | Worker functions executed by the task-queue dispatcher |
| `core/tasks/task_queue.py` | FIFO queue + dispatcher thread — compute tasks run one at a time |
| `core/tasks/task_store.py` | In‑memory registry of background tasks |
| `core/pipeline_runs.py` | Querying past pipeline runs & their context |
| `core/plugin_discovery/` | Walk `src/plugins`, import plugins, build the registry |
| `core/item_enrichment/` | Join item IDs with dataset metadata for the UI |
| `models/pipeline.py` | `StepDefinition`, `PipelineRequest`, `TaskState`, `TaskStatusResponse` |
| `models/plugin.py` | Category / parameter / display / registry schemas |
| `config/` | `config.yaml` + typed loader |
| `utils/logger.py` | Shared `sarssa` logger (console + `sarssa.log`) |

The backend also imports two shared utilities from outside `src/app`:
`utils.mlflow_manager.MLflowRunLoader` (artifact/param reads) and
`utils.plugin_notifier.PluginNotifier` (progress messages).

---

## 🚀 3. Running the backend

```bash
just run
# equivalent to:
cd src && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Two things matter here:

- **It runs from `src/`.** The plugin loader imports modules as
  `plugins.<dotted.path>` and the app as `app.main`, so `src/` must be
  the import root (`PluginManager.load`, `plugin_manager.py`). The
  `justfile` `run` recipe `cd src` first for exactly this reason.
- **The frontend origin is pinned.** CORS only allows
  `http://localhost:5173` (`main.py`). The UI must be served from
  that exact origin or browser requests are rejected.

The backend expects a running MLflow server (`just mlflow`) and talks
to it via `MLFLOW_TRACKING_URI` (see §9). Full setup, Docker, and
dataset download instructions live in the
[root README](../../README.md) — this document is about *architecture*,
not provisioning.

---

## 🛰️ 4. HTTP API reference

Three routers are mounted in `main.py`:

| Prefix | Router | Concern |
|--------|--------|---------|
| `/pipelines` | `api/routes_pipelines.py` | Run pipelines, poll tasks, query past runs |
| `/plugins` | `api/routes_plugins.py` | Plugin registry & dynamic dropdown choices |
| `/items` | `api/routes_items.py` | Artifact proxy & item metadata enrichment |

Interactive OpenAPI docs are available at `http://localhost:8000/docs`
when the server is running.

### `/pipelines` — execution & runs

| Method & path | Purpose | Body → Response | Code |
|---------------|---------|-----------------|------|
| `GET /pipelines/mlflow-info` | Resolve experiment name → ID + UI base URL for deep links | → `{ui_base_url, experiment_id}` | `get_mlflow_info` |
| `GET /pipelines/runs` | List parent pipeline runs, newest first. `?required_steps=a&required_steps=b` filters to runs whose context has all listed steps | → `[{run_id, run_name, status, start_time}]` | `list_runs` |
| `GET /pipelines/runs/{run_id}/context` | Fetch a run's `context.json` artifact | → context dict | `get_context` |
| `POST /pipelines/run-async` | Queue a full pipeline for execution (compute tasks run one at a time, FIFO) | `PipelineRequest` → `{task_id}` | `run_pipeline_async` |
| `GET /pipelines/tasks` | List queued + running tasks, newest first (backs the running-tasks menu) | → `[TaskSummary]` | `list_running_tasks` |
| `GET /pipelines/tasks/{task_id}` | Poll status/progress of a background task | → `TaskStatusResponse` | `get_task_status` |
| `POST /pipelines/tasks/{task_id}/cancel` | Cancel a queued or running task. A queued task is removed from the queue immediately; for a running one `?mode=graceful` (default) stops before the next step, `?mode=now` also aborts the current step in a cooperating plugin | → `{message}` (409 if not cancellable) | `cancel_task_endpoint` |
| `POST /pipelines/runs/{run_id}/execute-step` | Run **one** plugin step on an existing run (synchronous, scripting/testing; bypasses the queue) | `StepDefinition` → `{category, step_run_id}` | `execute_step` |
| `POST /pipelines/runs/{run_id}/execute-step-async` | Queue one plugin step on an existing run (used by the UI for multi‑run steps) | `StepDefinition` → `{task_id}` | `execute_step_async` |
| `POST /pipelines/run` | Execute all steps **synchronously** (legacy; blocks the request) | `PipelineRequest` → `{message, result}` | `run_pipeline` |

### `/plugins` — registry & dropdowns

| Method & path | Purpose | Code |
|---------------|---------|------|
| `GET /plugins/registry` | Full plugin registry: every category, its implementations, their parameters & display specs (the UI's source of truth) | `get_registry` |
| `GET /plugins/param-choices/{category}/{plugin_name:path}/{param_name}?run_id=…` | Resolve a **dynamic dropdown**: load the artifact named by the plugin's `DynamicDropdownHint`, run it through the plugin's formatter, return `[{label, value}]` | `get_param_choices` |

`plugin_name` is a `:path` parameter because it is a dotted module path
containing `.`/`/`. For **cascading** dropdowns the hint sets
`source_run_param`, in which case the supplied `run_id` is treated as a
*parent* run and the real artifact run is resolved through that run's
`context.json` (`_resolve_artifact_run_id`).

### `/items` — artifacts & enrichment

| Method & path | Purpose | Code |
|---------------|---------|------|
| `GET /items/artifact?run_id=…&filename=…` | Proxy a **JSON** artifact from any run (parsed) | `get_step_artifact` |
| `GET /items/artifact-raw?run_id=…&filename=…` | Serve a **raw** artifact file with a guessed MIME type (SVG/HTML/PDF visualisations) | `get_raw_artifact` |
| `GET /items/enrich?run_id=…&ids=a,b,c` | Join item IDs with `item_metadata.json` from a dataset‑loading run; missing items fall back to `{"id", "title": id}` | `get_enriched_items` |

The proxy endpoints exist so the browser never needs direct MLflow
access or credentials.

---

## ⚙️ 5. The pipeline execution engine

`PipelineEngine` (`core/pipeline_engine.py`) is the heart of the
backend. It has **two modes**.

### Step‑by‑step mode (what the UI uses)

```
start_run()            → create the parent MLflow run, then end it
execute_step(...)      → run ONE plugin as a nested run (repeat per step)
finalize_run(context)  → write context.json to the parent run, close it
```

- **`start_run(tags, description)`** creates the parent run
  named `pipeline_run_<timestamp>`, applies user tags (each key prefixed
  with `sarssa.`, `_TAG_PREFIX`) and a description, then immediately
  calls `mlflow.end_run()`. The parent run is therefore an empty
  shell that each step **re‑opens**.
- **`execute_step(plugin_name, params, context, notifier)`**:
  loads the plugin via `PluginManager.load`, optionally injects the
  `notifier`, opens the parent run *and* a **nested** run named after
  the plugin, then drives the plugin lifecycle
  `load_context → run → update_context` (§7) and records the nested
  run id under the plugin's category key:
  `context[category] = {"run_id": step_run.info.run_id}`.
- **`finalize_run(context)`** re‑opens the parent run and logs
  the full `context` as `context.json`.
- **`fail_run(context)`** logs the partial context, tags the
  run `cancellation=cancelled_by_user`, and sets MLflow status
  `FAILED`. Used on cancellation/fatal error.
- **`resume_run(run_id)`** re‑attaches to an existing parent
  run so additional (phase‑2 / multi‑run) steps can be appended.

### Batch mode (legacy)

`PipelineEngine(steps).run(context)` just chains
`start_run → execute_step* → finalize_run` in one blocking call. It is
used by `POST /pipelines/run` and is convenient for scripts/tests, but
it has **no notifier** so it produces no progress messages.

### Async execution & cooperative cancellation

The UI never blocks on a pipeline. The flow is:

```
POST /run-async ──▶ create_task(...) ──▶ submit(task, run_pipeline_worker)
       │                  │                        │  (task_queue.py, status "queued")
       └─▶ {task_id} ◀────┘                        ▼
                                    FIFO dispatcher thread — one task at a time:
                                    status "running" ──▶ run_pipeline_worker(task)
                                                   │
GET /tasks/{task_id}  (polled ~every 2s) ──reads── TaskState
```

- `create_task` (`task_store.py`) builds a `TaskState` with status
  `"queued"`, stores it in the process‑local `_tasks` dict, and
  returns it.
- `submit` (`task_queue.py`) enqueues the task. A single dispatcher
  thread executes queued workers strictly **one at a time** (FIFO), so
  concurrent launches never contest the GPU, the process-global RNG
  streams, or MLflow child numbering — the later task just waits as
  `"queued"` and starts automatically.
- `run_pipeline_worker` (`pipeline_worker.py`) creates its own
  `PluginNotifier`, **aliases** `task.messages` to the notifier's list
  (`pipeline_worker.py` — same object, so the polling endpoint sees
  messages immediately), then loops the requested steps. Before each
  step it checks `task.cancel_event`; if set, it marks the task
  `cancelled`, calls `engine.fail_run`, and returns.
- `cancel_task` (`task_store.py`) sets that event. A still-queued task
  is resolved to `cancelled` immediately (the dispatcher skips it); for
  a running one the cancel is **not immediate** — the currently
  executing step always runs to completion (it cannot be interrupted
  mid‑plugin) and the cancel takes effect at the next step boundary.
- On success the worker sets `task.status="completed"` and
  `task.context`; on exception, `task.status="error"` and `task.error`.

`run_step_worker` (`pipeline_worker.py`) is the single‑step variant
behind `execute-step-async`: it loads the parent run's context,
`resume_run`s it, executes the one step, and `finalize_run`s to
re‑persist `context.json`.

### Progress messages (no WebSockets)

There is **no streaming**. Plugins call
`self.notifier.info/warning/success/error(text)`; each call appends a
`{timestamp, level, text}` dict (`utils/plugin_notifier.py`,
`NotificationMessage`) to the shared list, and the UI surfaces them by
polling `GET /tasks/{task_id}` and reading `messages`. Outside a
pipeline (tests/scripts) plugins get a `NullNotifier` that discards
messages, so the same plugin code runs unchanged.

---

## 🔗 6. The pipeline context object

The **context** is how steps find each other's outputs. It is a plain
dict:

```json
{
  "dataset_loading": { "run_id": "a1b2c3…" },
  "training_cfm":    { "run_id": "d4e5f6…" },
  "training_sae":    { "run_id": "…"     }
}
```

- Each completed step writes its **nested run id** under its category
  key (`pipeline_engine.py`).
- `finalize_run` persists the whole dict as the `context.json`
  **artifact on the parent run** (`pipeline_engine.py`).
- A downstream plugin reads `context[<upstream category>]["run_id"]`
  and pulls that run's artifacts (via `MLflowRunLoader`) — this is how
  **intermediate results are reused** instead of recomputed.
- `get_run_context(run_id)` (`pipeline_runs.py`) downloads and
  parses that artifact; `get_eligible_pipeline_runs(required_steps)`
  returns only past runs whose context contains every required
  step. This powers the "compare against a past run" dropdowns and the
  "load from a previous run" UI: a new pipeline can start with an
  `initial_context` borrowed from an older run and skip stages that are
  already done.
- Parent vs nested runs are distinguished by the
  `mlflow.parentRunId` tag — `get_pipeline_runs` lists only runs
  *without* it (`pipeline_runs.py`).

---

## 🔌 7. Engine ↔ plugin contract

The engine treats every plugin as a `BasePlugin`
(`src/plugins/plugin_interface.py`). For each step it does, in
order:

1. `plugin = PluginManager.load(plugin_name)` — `importlib` imports
   `plugins.<dotted path>` and instantiates its `Plugin` class
   (`plugin_manager.py`).
2. `plugin.notifier = notifier` — inject the live notifier (skipped in
   batch mode).
3. `plugin.load_context(context)` — validate `io_spec.required_steps`
   and hydrate declared inputs from upstream MLflow runs onto `self.*`.
   Missing prerequisites raise `MissingContextError`
   (`plugin_interface.py`).
4. `plugin.run(**params)` — the plugin's *pure* business logic; `params`
   come straight from the `StepDefinition` (`plugin_interface.py`,
   abstract).
5. `plugin.update_context()` — log the plugin's declared outputs
   (params + artifacts) to the active nested run
   (`plugin_interface.py`).

The engine only knows these three lifecycle methods plus the
`name`, `io_spec`, and `notifier` attributes
(`plugin_interface.py`). **Everything about how a plugin
declares inputs/outputs, the `PluginIOSpec`, single vs compare
plugins, and how to author one is documented in
[`../plugins/README.md`](../plugins/README.md).** This section is only
the orchestration‑side view of that boundary.

---

## 🧭 8. Plugin discovery & the registry

`core/plugin_discovery` builds the catalogue the UI renders, with **no
manual registration** — plugins are found by directory convention.

- **Walk** (`_find_plugin_modules`, `plugin_registry.py`): under
  `src/plugins/<category>/`, a directory is a plugin when it contains a
  `.py` file whose stem equals the directory name (e.g.
  `elsa_trainer/elsa_trainer.py`). Folders starting with `_` or `.` are
  skipped; intermediate `single/` and `compare/` folders are traversed
  transparently and become part of the dotted module path.
- **Kind** (`_derive_kind`): if the path segment right after the
  category is `single` or `compare`, that becomes the plugin's `kind`;
  otherwise `kind` is `None` (the category doesn't use the
  single/compare split).
- **Parameters** (`_extract_parameters_from_instance`):
  `inspect.signature(plugin.run)` is read; every parameter except
  `self` becomes a `ParameterInfo` with type name, default,
  required‑ness, and a description parsed from a `typing.Annotated`
  string on the signature (`_parse_annotation`).
- **Widgets** (`_resolve_widget`): a parameter's UI hint
  (`DynamicDropdownHint` / `PastRunsDropdownHint` / `SliderHint`) maps
  to a widget type + `WidgetConfig`; otherwise it's a plain `"text"`
  input.
- **Display** (`_convert_display_spec`): the plugin's
  `io_spec.display` becomes either an `ItemRowsDisplayModel` or an
  `ArtifactDisplayModel` so the UI knows whether to render item cards or
  embedded files.
- `get_plugin_registry()` assembles all of this, keyed by
  category, and `GET /plugins/registry` serialises it.

---

## 🗃️ 9. MLflow integration & configuration

### What MLflow is (and how SARSSA uses it)

> **New to MLflow? Read this first** — the rest of this README leans on
> these terms (`run`, `nested run`, `artifact`, `param`) everywhere.

[MLflow](https://mlflow.org) is an open‑source **experiment‑tracking**
tool. SARSSA uses *only* its tracking + artifact‑storage features (not
its model registry or model serving). The vocabulary you need:

- **Experiment** — a named bucket of runs. SARSSA keeps everything in
  one experiment (`EXPERIMENT_NAME`).
- **Run** — one tracked execution. A run records **params** (inputs),
  **metrics** (numbers), **artifacts** (arbitrary output files), and
  **tags** (metadata). Runs can be **nested**: a parent run with child
  runs underneath it.
- **Tracking store** — the database holding run metadata, params and
  metrics. Here it is a SQLite file, `mlflow.db`, owned by the MLflow
  server; the backend talks to the server over HTTP
  (`MLFLOW_TRACKING_URI`).
- **Artifact store** — where the artifact *files* actually live. The
  server proxies uploads/downloads (`mlflow-artifacts:/` URIs) into its
  `--artifacts-destination`, the local `./mlartifacts` directory — so
  no machine-specific paths end up in the database.
- **MLflow UI** — a web app for browsing runs; the backend builds
  deep links into it from `MLFLOW_UI_BASE_URL`.

How that maps onto SARSSA:

- **One pipeline run = one parent run; each plugin step = one nested
  run** under it (§5). Running a pipeline produces a tree of runs you
  can open and inspect in the MLflow UI.
- A plugin's outputs are logged to **its own** nested run — scalar
  results as params/metrics, tensors and matrices as artifacts
  (`.npy` / `.npz` / `.json`). The step‑wiring document `context.json`
  (§6) is logged as an artifact on the **parent** run.
- Steps never recompute each other's work: a later step reads an
  earlier step's artifacts straight from MLflow (via `MLflowRunLoader`,
  below) using the run ids stored in `context.json`. This reuse is the
  whole point of the tracking layer — results stay **reproducible and
  shareable** across runs.

### `config/config.yaml`

Two sections, loaded and typed in `config/config.py`:

- **`mlflow`** → `EXPERIMENT_NAME`, `TRACKING_URI`
  (`sqlite:///mlflow-data/mlflow.db` — a fallback for tests/scripts; `just run`
  and Docker override it with `MLFLOW_TRACKING_URI` so tracking goes
  through the MLflow server), `ARTIFACT_ROOT` (`./mlartifacts`, used
  only with that fallback), `MLFLOW_UI_BASE_URL` (`config.py`).
- **`plugin_categories`** → an ordered map of category →
  `CategoryInfo` (`order`, `type`, `display_name`,
  `has_visual_results`) parsed into typed models
  (`config.py`). The current categories, in pipeline order:
  `dataset_loading` (0) → `training_cfm` (1) → `training_sae` (2) →
  `neuron_labeling` (3) are `one_time`; `labeling_evaluation` (4),
  `inspection` (5), `steering` (6) are `multi_run` with
  `has_visual_results: true`.

`one_time` vs `multi_run` is the contract behind phase‑1 vs phase‑2:
the first four stages build a pipeline once; the last three can be run
repeatedly against a finished pipeline (the `execute-step(-async)`
endpoints).

### Experiment bootstrap

`main.py` ensures the experiment exists *before* anything calls
`mlflow.set_experiment`. Created through the MLflow server with **no
explicit `artifact_location`**, the experiment gets a portable
`mlflow-artifacts:/<id>` root — the server resolves the physical
location at read/write time, so the tracking DB stays free of
machine-specific paths. The direct-SQLite fallback still pins
`ARTIFACT_ROOT` explicitly; otherwise MLflow would lazily auto‑create
the experiment under the default `./mlruns/<id>`.

### `MLflowRunLoader`

`utils.mlflow_manager.MLflowRunLoader` is the single abstraction for
reading a run: `get_json_artifact`, `get_npy_artifact`,
`get_npz_artifact` (sparse by default), `download_artifact`,
`artifact_exists`, `get_parameter(s)`, `get_metric(s)`. It uses
`mlflow.artifacts.download_artifacts`, so it is independent of whether
artifacts live on local disk or elsewhere. The backend uses it for
dropdown artifacts (`routes_plugins`), item metadata
(`item_enrichment`), and artifact proxying (`routes_items`); plugins
use it under the hood of `load_context`.

---

## 🧱 10. Data models

### `models/pipeline.py`

| Model | Kind | Role |
|-------|------|------|
| `StepDefinition` | Pydantic | One step: `plugin` (dotted path) + `params` |
| `PipelineRequest` | Pydantic | Request body: `steps`, `context`, `tags`, `description` |
| `TaskState` | dataclass | **Mutable** in‑memory state of a background task; mutated by the worker, read by the poll endpoint. Holds status, `run_id`, progress, `cancel_event`, and the shared `messages` list |
| `TaskStatusResponse` | Pydantic | Read‑only serialisation of `TaskState` + computed `total_steps` |

### `models/plugin.py`

`CategoryType` (`one_time`/`multi_run`), `CategoryInfo`,
`WidgetConfig` (dropdown/cascading/past‑runs/slider/text — the
docstring at `plugin.py` enumerates the exact field
combinations), `ParameterInfo`, the display models
(`DisplayRowSpec`, `ItemRowsDisplayModel`, `ArtifactFileModel`,
`ArtifactDisplayModel`, discriminated `DisplaySpec`),
`ImplementationInfo`, and `CategoryRegistryEntry`. These are the exact
shapes the frontend consumes from `/plugins/registry`, so the
[frontend types](../../frontend/README.md) mirror them.

---

## ⚠️ 11. Operational notes & gotchas

- **Tasks are in‑memory only.** `_tasks` is a module‑level dict
  (`task_store.py`). Restarting the backend loses all task state;
  the underlying MLflow runs survive, but their `task_id`s do not.
- **CORS is hard‑pinned** to `http://localhost:5173`
  (`main.py`). Serving the UI from any other origin breaks it.
- **Cancellation is cooperative**, never mid‑step
  (`pipeline_worker.py`). A long training step will finish before a
  cancel is honoured.
- **The parent run is created empty** and re‑opened by every step
  (`start_run` ends the run at `pipeline_engine.py`). Don't expect
  parent‑run artifacts before `finalize_run`.
- **Item metadata is cached** per `run_id` via `lru_cache(maxsize=32)`
  (`item_enrichment.py`). A re‑run with the same id reuses the
  cached metadata for the process lifetime.
- **Logging** goes to both stdout and `sarssa.log` at the repo root via
  the shared `sarssa` logger (`utils/logger.py`).

---

## 🔭 12. Where to go next

- 🔌 **Plugin internals & authoring:**
  [`src/plugins/README.md`](../plugins/README.md) — the
  `BasePlugin`/`BaseComparePlugin` contract, `PluginIOSpec`, and how to
  write your own plugin.
- 🖥️ **Frontend:** [`frontend/README.md`](../../frontend/README.md) —
  how the UI consumes the registry, runs pipelines, and renders
  results.
- 📘 **Project overview, setup & Docker:**
  [root `README.md`](../../README.md).
