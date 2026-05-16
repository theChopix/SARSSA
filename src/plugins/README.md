# 🔌 SARSSA Plugins (`src/plugins`)

> **What this is:** the plugin contract — how a plugin declares its
> inputs/outputs, what the engine does to it, and how to write your
> own. This is the core document for anyone extending SARSSA.
>
> **Who should read this:** contributors adding or maintaining a
> plugin (a dataset loader, a trainer, a labeling/steering method, a
> visualisation). The *orchestration* side — how plugins are run
> inside MLflow parent/nested runs — lives in
> [`../app/README.md`](../app/README.md); per‑category contracts live
> in each category's own `README.md` (see §7).

---

## 📑 Table of Contents

1. [🗺️ Philosophy](#-1-philosophy)
2. [🧱 Class hierarchy](#-2-class-hierarchy)
3. [📋 The `PluginIOSpec` reference](#-3-the-pluginiospec-reference)
4. [🔄 Context flow](#-4-context-flow)
5. [🧭 Discovery & directory contract](#-5-discovery--directory-contract)
6. [🛠️ Tutorial: write your own plugin](#-6-tutorial-write-your-own-plugin)
7. [🔭 Per‑category contracts](#-7-percategory-contracts)
8. [⚠️ Operational notes & gotchas](#-8-operational-notes--gotchas)
9. [➡️ Where to go next](#-9-where-to-go-next)

---

## 🗺️ 1. Philosophy

A plugin is **a declarative I/O contract plus pure business logic**.
It never talks to MLflow, never reads the pipeline context by hand,
and never wires itself into the registry. Three rules:

- **The engine owns the lifecycle.** For every step it calls, in
  order, `load_context(context)` → `run(**params)` →
  `update_context()` (`plugin_interface.py`). You implement
  `run()`; the base class implements the other two.
- **`io_spec` is the contract.** A `PluginIOSpec` class attribute
  declares what to pull from upstream runs (`self.*` is hydrated for
  you) and what to log afterwards (read back from `self.*`). The base
  class does every MLflow read/write from that spec.
- **`run()` is pure.** When `run()` is called, every declared input is
  already an attribute on `self`. `run()` computes results and assigns
  them to the `self.*` attributes the spec names as outputs. That's
  it — no logging, no artifact handling.

This is why a plugin's own code is short and testable: the I/O is data
(`io_spec`), not control flow.

---

## 🧱 2. Class hierarchy

There are only two base classes, and every plugin extends exactly one
of them: **`BasePlugin`** for ordinary single‑run plugins (the vast
majority), and **`BaseComparePlugin`** for the "compare against a past
run" plugins. The second *is* the first plus extra past‑run plumbing,
so read `BasePlugin` first.

### `BasePlugin` (`plugin_interface.py`)

**What it is:** the abstract base class that *every* plugin
ultimately extends. It is the "single‑run" plugin — it works only with
the **current** pipeline run.

**What it does for you:** all the repetitive wiring. Just by
subclassing it and declaring an `io_spec`, you automatically get:
validation that required upstream steps ran, download of your declared
inputs from upstream MLflow runs onto `self`, and logging of your
declared outputs back to MLflow. You write none of that plumbing.

**How you use it:** write `class Plugin(BasePlugin)`, set `name` and
`io_spec`, and implement `run()`. You never call the lifecycle methods
below yourself — the engine calls them, in order (§4). You only
*reference* the loader/saver strategy names from your `io_spec`.

**Class attributes** (override per plugin):

| Attr | Default | Role |
|------|---------|------|
| `name` | `None` | Human label for the UI. When `None`, the registry derives one from the directory name (`_make_display_name`, `plugin_registry.py`). |
| `io_spec` | shared `PluginIOSpec()` | The I/O contract. **Always declare your own** (see §8). |
| `notifier` | `NullNotifier()` | Progress channel. The engine swaps in a real `PluginNotifier` before `run()`; outside a pipeline it stays a silent no‑op, so the same code runs in tests unchanged. |

**Lifecycle methods:**

- `load_context(context)` — sets `self._context`, validates
  `io_spec.required_steps` (`_validate_required_steps`), then
  hydrates `self.*` from `input_artifacts`
  (`_load_input_artifacts`) and `input_params`
  (`_load_input_params`). Override only to add cross‑step
  assertions, and call `super().load_context(context)` first.
- `run(**params)` (abstract) — **the only method you must
  implement.** Inputs are already on `self.*`; assign outputs to
  `self.*`.
- `update_context()` — logs `output_params` via
  `mlflow.log_params` (`_log_output_params`) and `output_artifacts`
  via `mlflow.log_artifacts` (`_log_output_artifacts`). Override
  only for conditional/extra logging, calling `super()` first (the
  `MovieLens Loader` does this for optional tag artifacts).

**Private strategy tables** (you reference these by string in the
spec, you don't call them):

- `_load_artifact`: `npz`, `npy`, `json`, `base_model`,
  `sae_model`, `pt`.
- `_save_artifact`: `json`, `npz`, `npy`, `pt`, `model`.

Unknown strategy names raise `ValueError` immediately — fail‑fast.
Note the **loader/saver asymmetry** (§8): a model is *saved* with
`"model"` but *loaded* with `"base_model"`/`"sae_model"`.

### `BaseComparePlugin` (`compare_plugin_interface.py`)

**What it is:** a subclass of `BasePlugin` for plugins that need a
*second* pipeline run in addition to the current one.

**How it differs from `BasePlugin`:** a plain `BasePlugin` only ever
sees the current pipeline run. A `BaseComparePlugin` *also* loads a
**past** run that the user selects, runs the same analysis on both,
and presents them side‑by‑side. Concretely: *"did the SAE I just
trained label concepts better than last week's run?"* — you pick that
older run and the plugin shows now‑vs‑then.

**When you need it:** only for the evaluation‑style categories that
run *after* a finished pipeline and can be re‑run repeatedly — the
`multi_run` categories: **`labeling_evaluation`**, **`inspection`**,
**`steering`**. Each of those typically ships two implementations: a
`single/` one (extends `BasePlugin`, current run only) and a
`compare/` one (extends `BaseComparePlugin`, current vs a chosen past
run). The `one_time` stages (dataset loaders, trainers, neuron
labeling) are always plain `BasePlugin`.

**What you get for free:** it subclasses `BasePlugin` and, via
`__init_subclass__`, gives every subclass two things automatically:

1. **An auto‑injected `PastRunsDropdownHint`** for the `past_run_id`
   parameter, built from the subclass's `past_run_required_steps`
   (`_merge_past_run_hint`). You do **not** add this hint
   yourself — only your `DynamicDropdownHint`s go in `param_ui_hints`.
2. **A wrapped `run()`** (`_wrap_run_for_past_context`) that loads
   the past run's `context.json` into `self.past_context` *before*
   your `run()` body executes. The wrapper preserves the original
   signature (`__signature__`) so registry introspection still sees
   your real parameters.

Subclass requirements:

- Set `past_run_required_steps` (class attr) — step keys an eligible
  past run's `context.json` must contain.
- Declare `past_run_id: str` as a `run()` parameter (passed as a
  **keyword** by the engine).
- Use `self.load_past_artifact(step, filename, loader, **kwargs)`
  to pull past‑run artifacts. It resolves the per‑step run id
  through `self.past_context` and reuses `BasePlugin._load_artifact`,
  so every loader strategy works identically on the past side.

The current side is loaded exactly like a normal plugin (via
`io_spec.input_artifacts`); only the past side uses
`load_past_artifact`.

---

## 📋 3. The `PluginIOSpec` reference

`PluginIOSpec` (`plugin_interface.py`) — all fields optional,
default empty.

### Fields at a glance

| Field | Element type | Meaning |
|-------|--------------|---------|
| `required_steps` | `list[str]` | Context keys that must exist (with a `run_id`) before the plugin runs. Missing → `MissingContextError`. |
| `input_artifacts` | `ArtifactSpec` | Files to download from an upstream run and set on `self`. |
| `input_params` | `ParamSpec` | Params to read from an upstream run, cast, and set on `self`. |
| `output_artifacts` | `OutputArtifactSpec` | Files to save from `self.*` after `run()`. |
| `output_params` | `OutputParamSpec` | Params to log from `self.*` after `run()`. |
| `display` | `ItemRowsDisplaySpec` or `ArtifactDisplaySpec` | How the UI renders results. `None` = no visual output. |
| `param_ui_hints` | `DynamicDropdownHint` / `PastRunsDropdownHint` / `SliderHint` | How a `run()` parameter is rendered (default: plain text input). |

### Inputs - loaded onto `self` before `run()`

Declared via `required_steps`, `input_artifacts`, `input_params`; the
base class hydrates each as an attribute before `run()` is called.

**`ArtifactSpec(step, filename, attr, loader, loader_kwargs={})`** —
load `filename` from `context[step]["run_id"]` with strategy `loader`,
set as `self.<attr>`. `loader_kwargs` is forwarded to the loader (e.g.
`{"allow_pickle": True}` for `npy`, `{"device": "cpu"}` for
`base_model`/`sae_model`).

**`ParamSpec(step, param_name, attr, dtype=str)`** — read MLflow param
`param_name` from `context[step]`, cast via `dtype(raw)`, set as
`self.<attr>`. MLflow params are strings; see the `bool` foot‑gun in
§8.

### Outputs - logged after `run()`

Declared via `output_artifacts`, `output_params`; the base class reads
each from `self.*` and logs it to the active MLflow run.

**`OutputArtifactSpec(attr, filename, saver)`** — save `self.<attr>`
to `filename` with strategy `saver`.

**`OutputParamSpec(key, attr)`** — log `self.<attr>` to MLflow under
`key`.

### Display - how results are rendered

*How the frontend renders this plugin's results.* Optional: `None`
(the default for most plugins) means the plugin produces data
artifacts only and has no visual output. Plugins in the
`multi_run` categories marked `has_visual_results` in `config.yaml`
(`labeling_evaluation`, `inspection`, `steering`) set this so the UI
can render their output with **no plugin‑specific frontend code** —
the spec is converted to a model (`_convert_display_spec`) and the
frontend reads it from `/plugins/registry`. Two shapes:
- `ItemRowsDisplaySpec(rows=[DisplayRowSpec(key, label), …])` — each
  row's `key` is an output‑artifact JSON of item IDs; the UI enriches
  them with metadata and renders `ItemCard` rows.
- `ArtifactDisplaySpec(files=[ArtifactFileSpec(filename, label,
  content_type), …])` — render standalone artifacts inline (SVG as
  `<img>`, HTML as `<iframe>`, etc.).

### Parameter UI hints — how inputs are collected

*How the user supplies this plugin's `run()` parameters.* By default
the registry turns every `run()` parameter into a plain text box (its
type and default read from the signature).
A hint in `param_ui_hints` upgrades one parameter — matched by
`param_name` — to a richer widget (`_resolve_widget` maps each hint to
a widget type) so the user picks a valid value instead of typing a raw
string. In short: *display* governs how **outputs** are shown, UI
hints govern how **inputs** are collected.
- `DynamicDropdownHint(artifact_step, artifact_file, artifact_loader,
  formatter, source_run_param=None)` — populate a dropdown by loading
  an artifact and passing it through the plugin's `formatter`
  **static method** (returns `[{"label", "value"}]`). Set
  `source_run_param` to make it **cascade** off another parameter
  (e.g. a `past_run_id`) — the backend then treats that param's value
  as a parent run, resolves `artifact_step` through its `context.json`,
  and the frontend refetches choices when the watched param changes.
- `PastRunsDropdownHint(required_steps)` — dropdown of past pipeline
  runs whose `context.json` contains all `required_steps`. For compare
  plugins this is **auto‑injected** (§2); declare it manually only for
  a non‑compare plugin.
- `SliderHint(min_value, max_value, step)` — a range slider.

---

## 🔄 4. Context flow

```
context = { "dataset_loading": {"run_id": "a1b2…"},
            "neuron_labeling":  {"run_id": "c3d4…"} }
                       │
   load_context(context)         BasePlugin (you don't write this)
   ├─ _validate_required_steps   every required_steps key present + has run_id
   ├─ _load_input_artifacts      for each ArtifactSpec:
   │                               MLflowRunLoader(context[step].run_id)
   │                               → _load_artifact(loader) → setattr(self, attr)
   └─ _load_input_params         for each ParamSpec:
                                   loader.get_parameter → dtype(raw) → setattr
                       │
   run(**params)                 YOUR code: read self.<inputs>,
                                 compute, assign self.<outputs>
                       │
   update_context()              BasePlugin (you don't write this)
   ├─ _log_output_params         mlflow.log_params({key: self.attr, …})
   └─ _log_output_artifacts      tmpdir → _save_artifact(saver) per spec
                                 → mlflow.log_artifacts(tmpdir)
```

A failed input load is wrapped as `MissingContextError` with the
offending step/filename/run id (`plugin_interface.py`), so a
broken pipeline points at the exact missing dependency.

**Compare plugins** add one step: the wrapped `run()` calls
`_load_past_context(past_run_id)` (`compare_plugin_interface.py`)
to populate `self.past_context` before your body; you then call
`self.load_past_artifact(...)` for the past side.

---

## 🧭 5. Discovery & directory contract

No manual registration. `plugin_registry.py` finds plugins by
**directory convention** (`_find_plugin_modules`):

- Under `src/plugins/<category>/`, a directory is a plugin when it
  contains a `.py` file **whose stem equals the directory name** —
  `sae_trainer/sae_trainer.py`, not `sae_trainer/plugin.py`.
- That module **must define a class literally named `Plugin`** —
  `PluginManager.load` does `module.Plugin()` (`plugin_manager.py`).
- Directories starting with `_` or `.` are skipped (use `_helpers.py`
  / `_shared/` for non‑plugin code — see `inspection/_top_k.py`).
- Intermediate `single/` and `compare/` folders are traversed
  transparently and become part of the dotted path; the segment right
  after the category sets the plugin's **kind**
  (`_derive_kind`) — `single`, `compare`, or `None`.
- The category must also be listed in `config.yaml`'s
  `plugin_categories` (see [`../app/README.md`](../app/README.md) §9);
  a directory with no config entry is never surfaced, and a config
  entry with no directory yields an empty category.

The registry introspects `inspect.signature(plugin.run)`
(`_extract_parameters_from_instance`): every parameter except
`self` becomes a UI field, its type/default/required‑ness read from
the signature, its **description** from the first string in a
`typing.Annotated[...]` annotation (`_parse_annotation`), and its
widget from any matching `param_ui_hints` entry (`_resolve_widget`).

---

## 🛠️ 6. Tutorial: write your own plugin

### A single plugin

```
src/plugins/<category>/<impl>/<impl>.py      # e.g. steering/my_steer/my_steer.py
src/plugins/<category>/<impl>/__init__.py    # empty
```

```python
from typing import Annotated

from plugins.plugin_interface import (
    ArtifactSpec, BasePlugin, OutputArtifactSpec,
    OutputParamSpec, PluginIOSpec,
)
from utils.plugin_logger import get_logger

logger = get_logger(__name__)


class Plugin(BasePlugin):                       # MUST be named `Plugin`
    name = "My Steering Method"

    io_spec = PluginIOSpec(                      # declare your OWN io_spec
        required_steps=["dataset_loading", "training_sae"],
        input_artifacts=[
            ArtifactSpec("training_sae", "sae", "sae_model", "sae_model",
                         loader_kwargs={"device": "cpu"}),
        ],
        output_artifacts=[
            OutputArtifactSpec("recs", "recommendations.json", "json"),
        ],
        output_params=[OutputParamSpec("alpha", "alpha_param")],
    )

    def run(
        self,
        alpha: Annotated[float, "Steering strength applied to the concept."] = 0.5,
    ) -> None:
        # self.sae is already loaded (from io_spec.input_artifacts)
        self.recs = my_logic(self.sae, alpha)   # assign declared outputs
        self.alpha_param = alpha
        self.notifier.info(f"Steered with alpha={alpha}")
```

### A compare plugin

```python
from typing import Annotated

from plugins.compare_plugin_interface import BaseComparePlugin
from plugins.plugin_interface import PluginIOSpec, OutputArtifactSpec


class Plugin(BaseComparePlugin):
    name = "My Method (compare)"
    past_run_required_steps = ["dataset_loading", "training_sae"]

    io_spec = PluginIOSpec(
        required_steps=["dataset_loading", "training_sae"],
        # NOTE: no PastRunsDropdownHint here — it is auto-injected.
        output_artifacts=[
            OutputArtifactSpec("current", "current.json", "json"),
            OutputArtifactSpec("past", "past.json", "json"),
        ],
    )

    def run(
        self,
        past_run_id: Annotated[str, "Past pipeline run to compare against."],
        alpha: Annotated[float, "Steering strength."] = 0.5,
    ) -> None:
        self.current = my_logic(alpha)                        # current side
        past_sae = self.load_past_artifact(                   # past side
            "training_sae", "sae", "sae_model", device="cpu",
        )
        self.past = my_logic_on(past_sae, alpha)
```

### Checklist

1. Folder/file named identically; class named `Plugin`; empty
   `__init__.py`.
2. Category exists in `config.yaml` `plugin_categories`.
3. Own `io_spec`; every `run()` output assigned to a `self.*` named in
   the spec.
4. Annotate every `run()` parameter with `Annotated[type, "help text"]`
   — that string becomes the UI tooltip.
5. Dropdowns: add a `DynamicDropdownHint` + a matching `@staticmethod`
   formatter returning `[{"label", "value"}]`.
6. Compare: extend `BaseComparePlugin`, set `past_run_required_steps`,
   take `past_run_id` (keyword), use `load_past_artifact`.
7. Sanity check: it loads and the UI sees it — `GET /plugins/registry`
   should list it with the right params.

---

## 🔭 7. Per‑category contracts

SARSSA currently has **seven plugin categories**, run in this pipeline
order. Categories 0–3 build a pipeline once (`one_time`); 4–6 are
re‑runnable against a finished pipeline (`multi_run`, with visual
results):

| # | Category | Phase | What it does (briefly) |
|---|----------|-------|------------------------|
| 0 | `dataset_loading` | `one_time` | Load an interaction dataset; split into train/val/test matrices + item metadata |
| 1 | `training_cfm` | `one_time` | Train the base collaborative‑filtering recommender (ELSA) |
| 2 | `training_sae` | `one_time` | Train a sparse autoencoder on the base model's embeddings |
| 3 | `neuron_labeling` | `one_time` | Assign human‑readable concept labels to the SAE neurons |
| 4 | `labeling_evaluation` | `multi_run` · visual | Judge label quality — embedding maps, dendrograms, nearest‑label distances |
| 5 | `inspection` | `multi_run` · visual | Explore which items a concept neuron activates on most |
| 6 | `steering` | `multi_run` · visual | Steer recommendations by amplifying/suppressing concepts |

Each category has (or will have) its own `README.md` documenting the
exact artifacts/params it produces and expects, so a downstream plugin
knows what context keys and filenames to declare — `dataset_loading/`
is the headline "bring your own dataset" tutorial; the other six get
one `README.md` each.

This document is the **cross‑category contract**; those are the
**per‑category** specifics. Orchestration is in
[`../app/README.md`](../app/README.md); the UI side in
[`../../frontend/README.md`](../../frontend/README.md).

---

## ⚠️ 8. Operational notes & gotchas

These are real, code‑traced sharp edges — read before authoring.

- **The `"model"` saver writes a fixed `config.json` + `model.pt`.**
  It ignores `filename` and writes one model at the run's artifact
  root, so there is **one `"model"` artifact per plugin** —
  `OutputArtifactSpec.filename` must be `""`, otherwise the saver
  raises `ValueError` (enforced in `_save_artifact`, not a silent
  overwrite). Read it back with the `"base_model"` / `"sae_model"`
  loader — note the deliberate loader/saver name asymmetry.
- **Building the registry instantiates every plugin.** Every
  `GET /plugins/registry` call walks all categories and does
  `import module; module.Plugin()` for each (`plugin_registry.py`,
  `plugin_manager.py`). Module import and `Plugin.__init__` must be
  cheap and side‑effect‑free — **do all heavy work in `run()`**, never
  at import or construction (no model loading, no file I/O, no network
  in `__init__`).
- **Compare: `past_run_id` must be keyword.** The `run()` wrapper
  reads `kwargs.get("past_run_id")`
  (`compare_plugin_interface.py`); a positional call yields
  `MissingContextError`. The engine always calls `run(**params)`, so
  this only bites in hand‑written tests/scripts.
- **`ParamSpec` casts a string — avoid `dtype=bool`.** MLflow params
  come back as strings; `bool("False")` is `True`
  (`plugin_interface.py`). Use `int`/`float`/`str`, or pass the
  raw string and parse it yourself in `run()`.
- **Always declare your own `io_spec`.** The class‑level default is a
  single shared `PluginIOSpec()` instance (`plugin_interface.py`);
  a plugin that never sets its own would share one mutable spec with
  the base class. Every real plugin sets `io_spec = PluginIOSpec(...)`
  as a class attribute — do the same. (`BaseComparePlugin` mutates the
  spec safely via `dataclasses.replace`, not in place.)
- **`name` is optional but recommended.** Without it the UI label is
  the directory name title‑cased (`_make_display_name`), e.g.
  `movieLens_loader` → `Movielens Loader`. Set `name` for a clean
  label.

---

## ➡️ 9. Where to go next

- 📂 **Per‑category contracts:** each category's own `README.md`
  (`<category>/README.md`) — the exact artifacts and params it
  produces and consumes. Start with
  [`dataset_loading/README.md`](dataset_loading/README.md), the
  "bring your own dataset" tutorial (see the §7 table for all seven).
- ⚙️ **Backend & pipeline engine:**
  [`../app/README.md`](../app/README.md) — how the engine drives the
  `load_context → run → update_context` lifecycle inside MLflow
  parent/nested runs, and how the registry is served to the UI.
- 🖥️ **Frontend:**
  [`../../frontend/README.md`](../../frontend/README.md) — how the UI
  consumes the registry, renders parameter widgets, and displays
  plugin results.
- 📘 **Project overview, setup & Docker:**
  [root `README.md`](../../README.md).
