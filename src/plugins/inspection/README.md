# 🔬 Inspection (`src/plugins/inspection`)

> **What this is:** the sixth stage of SARSSA — a reality check on
> the labels. [`neuron_labeling`](../neuron_labeling/README.md) claims
> "neuron 42 means *sci-fi*"; inspection answers **"so which items
> does neuron 42 actually fire on?"** by showing the catalogue items
> that activate it most strongly.
>
> **Who should read this:** anyone verifying whether a label is
> earned or decorative, and contributors adding an inspection view
> (§6). The plugin contract in general lives in
> [`../README.md`](../README.md); this doc is the inspection
> specifics.

---

## 📑 Table of Contents

1. [🗺️ Big picture: the category contract](#1-big-picture)
2. [🔀 `single/` vs `compare/`](#2-single-vs-compare)
3. [📊 The plugins](#3-plugins)
4. [📥 Inputs · 📤 Outputs](#4-inputs-outputs)
5. [🎛️ `run()` parameters](#5-run-parameters)
6. [🛠️ Adding your own inspection plugin](#6-adding-your-own)
7. [⚠️ Operational notes & gotchas](#7-operational-notes-gotchas)
8. [➡️ Where to go next](#8-where-to-go)

---

<a id="1-big-picture"></a>

## 🗺️ 1. Big picture: the category contract

The whole category is one lookup:

```
pick a labelled neuron  ──▶  its column in item_acts  ──▶  top-k rows
                             (activation per item)          │
                                                            ▼
                                              rows of item cards in the UI
```

Three things define it:

- **No model is run.** The `item_acts.npz` matrix — one row per
  catalogue item, one column per neuron — was already computed
  during neuron labeling, where each item was pushed through the
  CFM and SAE encoders as a one-hot interaction row. Inspection just
  reads column `neuron_id` and takes its largest entries, so it
  needs **no checkpoints and no GPU** and returns instantly. It is
  the only `multi_run` stage that does not load a model (`steering`
  does).
- **Activation strength is the whole ranking.** Column `n` of
  `item_acts` is neuron `n`'s fingerprint across the catalogue; the
  top-k of that column is, by construction, "the items that most
  embody this concept". No embeddings, no distances, no thresholds.
- **Results are items, not charts.** Where
  [`labeling_evaluation`](../labeling_evaluation/README.md) renders
  an HTML page, inspection returns **lists of item ids** and lets the
  app present them, declared in `io_spec` (§3 of
  [`../README.md`](../README.md) is the full `PluginIOSpec` field
  reference):

  ```python
  display=ItemRowsDisplaySpec(
      type="item_rows",
      rows=[DisplayRowSpec("top_k_item_ids", "Top Items for Concept")],
  ),
  ```

  Each row names one output artifact holding item ids. The frontend
  fetches it, asks the backend to join those ids with the dataset's
  `item_metadata.json`, and renders a horizontally scrollable row of
  cards — poster, title, year, tags — so you judge the concept by
  *looking at the items*, not by reading numbers.

The category is `multi_run`: it does not extend a pipeline, it
inspects a finished one, and re-running it with another neuron costs
nothing.

---

<a id="2-single-vs-compare"></a>

## 🔀 2. `single/` vs `compare/`

Like the other `multi_run` categories
([`labeling_evaluation`](../labeling_evaluation/README.md),
`steering`), the plugins come in two flavours and the folder name
decides which:

| | `single/` | `compare/` |
|---|---|---|
| Base class | `BasePlugin` | `BaseComparePlugin` |
| Sees | the current run only | the current run **and** one past run |
| Extra params | — | `past_run_id`, `past_neuron_id` |
| Question | "what is this concept made of?" | "what does each run mean by its concept?" |

A compare plugin declares `past_run_required_steps`, and the base
class **auto-injects** the `past_run_id` dropdown, offering only past
runs that contain the required steps. Inside `run()`, the past side
is loaded with `self.load_past_artifact(...)` rather than through
`io_spec` (which only ever describes the *current* run).

**Here the two sides are chosen independently**, and that is
deliberate: you pick a neuron in the current run *and* a separate
neuron in the past run. Neuron ids are **not comparable across
runs** — SAE training is stochastic, so neuron 42 in two runs is two
unrelated directions that happen to share an index. Comparing
"neuron 42 then vs now" would be meaningless; comparing *"the neuron
labelled sci-fi in each run"* is the question worth asking, so the
UI makes you name both.

The past-side dropdown cascades off `past_run_id`: choose a run
first, and the second dropdown fills with **that run's** labels.

---

<a id="3-plugins"></a>

## 📊 3. The plugins

### 🔍 SAE Inspection (`single/sae_inspection`)

Pick a labelled neuron from the dropdown, get the *k* items whose
activation on it is highest, rendered as a row of item cards.

The dropdown is the category's main piece of UI. It is filled from
`neuron_labels.json` at form time, and each option carries what you
need to choose well:

```
sci-fi [neuron id 1319] · conf 0.701
```

— the label, the neuron id, and the labeling confidence, with the
option's background **tinted** by that confidence (see the
[labeling README](../neuron_labeling/README.md) for what the score
means). Low-confidence neurons are exactly the interesting ones to
inspect: the label is a guess, and the items either back it up or
expose it.

Alongside the item ids the plugin stores `top_k_activations.json`
— the raw activation values in the same order — so the drop-off
between rank 1 and rank *k* can be examined outside the UI.

**Answers:** does this neuron's label match what it actually
responds to?

### 🔍 SAE Inspection — compare (`compare/sae_inspection`)

The same lookup run twice, once per run, with two independently
chosen neurons, and displayed as two stacked rows: *Top Items for
Concept — Current Run* and *— Past Run*.

Both halves call the same shared helper (`_top_k.py`), so the two
sides are computed identically — only their inputs differ. The past
run's `items.npy`, `neuron_labels.json` and `item_acts.npz` are all
pulled from that run, which means the comparison also works when the
two runs used **different datasets**: each side is resolved against
its own catalogue.

**Answers:** do two separately trained SAEs mean the same thing by
the same label — do their "sci-fi" neurons pick out the same films?

---

<a id="4-inputs-outputs"></a>

## 📥 4. Inputs · 📤 Outputs

**Inputs** — identical for both plugins:

| From | What |
|---|---|
| `dataset_loading` | `items.npy` — maps activation row index → item id |
| `neuron_labeling` | `neuron_labels.json` — fills the dropdown; supplies the label |
| `neuron_labeling` | `item_acts.npz` — the (items × neurons) activation matrix, stored sparse |
| *(compare only)* the chosen past run | all three of the above, via `load_past_artifact` |

Note what is **not** required: no CFM checkpoint, no SAE checkpoint.
The activations were precomputed upstream.

**Outputs:**

| Plugin | Artifacts | Run params |
|---|---|---|
| `sae_inspection` | `top_k_item_ids.json`, `top_k_activations.json` | `label`, `k_used` |
| `sae_inspection` (compare) | `current_top_k_item_ids.json`, `current_top_k_activations.json`, `past_top_k_item_ids.json`, `past_top_k_activations.json` | `label`, `past_label`, `k_used` |

`k_used` is the *actual* k after clamping to the catalogue size, so
a run that asked for more items than exist stays self-documenting.

---

<a id="5-run-parameters"></a>

## 🎛️ 5. `run()` parameters

| Plugin | Param | Default | Meaning |
|---|---|---|---|
| both | `neuron_id` | *(required)* | the concept neuron to inspect, picked from the label dropdown |
| both | `k` | 10 | how many top-activating items to return (clamped to the catalogue size) |
| compare | `past_run_id` | *(required)* | the past run to compare against; dropdown of eligible runs |
| compare | `past_neuron_id` | *(required)* | the neuron on the past side, chosen from *that* run's labels (§2) |

In the compare plugin `k` applies to **both** sides, but each side
clamps against its own catalogue — so the rows differ in length only
when `k` exceeds one of them, and `k_used` records the current side's
value.

---

<a id="6-adding-your-own"></a>

## 🛠️ 6. Adding your own inspection plugin

The category is deliberately thin — a plugin here is a ranking rule
over `item_acts` plus a way to show the result. Ideas that fit:
bottom-k (what a neuron *suppresses*), items where two or more neurons fire
together, or a per-neuron activation histogram.

Keep the category's shape:

- **Declare** `required_steps=["dataset_loading", "neuron_labeling"]`
  and the three input artifacts above; that is all the data the
  category needs.
- **Reuse `_top_k.py`** (`compute_top_k_for_neuron`) if you rank by
  plain activation — it handles the sparse-column conversion, the
  clamping of `k`, and the row-index → item-id mapping. The
  underscore prefix keeps it out of plugin discovery.
- **Emit item ids** and declare them as `ItemRowsDisplaySpec` rows to
  get enriched item cards for free. A view that needs a chart instead
  can render HTML and use `ArtifactDisplaySpec`, exactly as the
  evaluation plugins do — the category is not restricted to item
  rows, it just happens that both current plugins want them.
- **Offer a neuron dropdown** with a `DynamicDropdownHint` pointing
  at `neuron_labels.json` plus a formatter; copying
  `_format_neuron_choices` gives you the label/confidence/tint
  options users already know from `steering`.
- **For a compare plugin**, set `past_run_required_steps` and read
  the other side with `load_past_artifact`. If your view involves
  choosing something on the past side too, give that param its own
  dropdown cascading off `past_run_id` (`source_run_param`).

---

<a id="7-operational-notes-gotchas"></a>

## ⚠️ 7. Operational notes & gotchas

- **Dead neurons are in the dropdown too.** Neurons the labeling step
  could not name appear as `None [neuron id 87]`, and their
  activation column is all zeros — the top-k then returns an
  arbitrary slice of the catalogue with activation `0.0`. If a row of
  items looks random, check the activations artifact before blaming
  the model.
- **Item cards degrade gracefully.** Posters, titles and tags come
  from the dataset's `item_metadata.json`; a dataset without it still
  renders, with the raw item id as the card title.

---

<a id="8-where-to-go"></a>

## ➡️ 8. Where to go next

- **Where the labels and activations come from:**
  [`../neuron_labeling/README.md`](../neuron_labeling/README.md)
- **Judging the label set as a whole:**
  [`../labeling_evaluation/README.md`](../labeling_evaluation/README.md)
- **Acting on a concept instead of observing it:**
  [`../steering/README.md`](../steering/README.md) — same neuron
  dropdown, but it nudges recommendations
- **Where the activation matrix originates:**
  [`../training_sae/README.md`](../training_sae/README.md)
- **The plugin contract** (discovery, `io_spec`, `run()` params,
  `BaseComparePlugin`): [`../README.md`](../README.md)
