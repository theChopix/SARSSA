# 🔍 Labeling Evaluation (`src/plugins/labeling_evaluation`)

> **What this is:** the fifth stage of SARSSA — the first one that
> runs *after* a pipeline is finished. Its plugins take the neuron
> labels produced by [`neuron_labeling`](../neuron_labeling/README.md)
> and ask **"is this label set any good?"** — does it have coherent
> semantic structure, does it cover the concepts you care about, and
> did it change since a previous run?
>
> **Who should read this:** anyone picking a plugin to judge a
> labeling method, and contributors adding an evaluation view (§6).
> The plugin contract in general lives in
> [`../README.md`](../README.md); this doc is the evaluation
> specifics.

---

## 📑 Table of Contents

1. [🗺️ Big picture: the category contract](#-1-big-picture-the-category-contract)
2. [🔀 `single/` vs `compare/`](#-2-single-vs-compare)
3. [📊 The seven plugins](#-3-the-seven-plugins)
4. [📥 Inputs · 📤 Outputs](#-4-inputs--outputs)
5. [🎛️ `run()` parameters](#-5-run-parameters)
6. [🛠️ Adding your own evaluation plugin](#-6-adding-your-own-evaluation-plugin)
7. [⚠️ Operational notes & gotchas](#-7-operational-notes--gotchas)
8. [➡️ Where to go next](#-8-where-to-go-next)

---

## 🗺️ 1. Big picture: the category contract

Every plugin here does the same three things:

```
neuron_labels.json  ──▶  embed the label TEXTS  ──▶  measure cosine
   (label strings)         (one vector each)         relationships
                                                          │
                                                          ▼
                                            one interactive HTML page
```

Three consequences follow, and they are what make this a category
rather than a pile of charts:

- **Only the label text is read.** Not the activations, not the
  items, not even the `confidence` score sitting next to each label
  in `neuron_labels.json`. A plugin here judges the label set as a
  *vocabulary* — whether it is well-structured, broad, or drifting —
  while the per-label "is this label right?" question is answered
  upstream by the confidence metric.
- **Semantics come from an embedding model.** Labels become vectors
  through the embedder registry (today: OpenAI), so "action" and
  "adventure" end up close together even though they share no
  characters. Every plugin therefore takes the same
  `embedding_provider` / `embedding_model` pair, and every plugin
  costs an embedding API call — see the shared cache in §7.
- **Cosine distance everywhere.** The tree clusters with cosine
  (`pdist`), the drift plugins measure with cosine (`cdist`), the
  keyword search ranks by cosine similarity, and the maps default to
  `umap_metric="cosine"`. One notion of "semantically close" runs
  through the whole category.

The category is `multi_run` and visual: it does not extend a
pipeline, it *inspects* a finished one, and can be re-run as often
as you like with different parameters.

**Every plugin here delivers its result the same way** — as a
self-contained Plotly HTML page, declared with one line in the
plugin's `io_spec` (§3 of [`../README.md`](../README.md) is the full
`PluginIOSpec` field reference):

```python
display=ArtifactDisplaySpec(
    files=[ArtifactFileSpec("embedding_map.html", "Embedding Map", "text/html")],
),
```

That is the whole contract with the frontend: the app fetches the
named artifact from the step's MLflow run and renders it inline in a
sandboxed iframe, using the `content_type` to decide how (an
`<iframe>` for HTML, an `<img>` for an image). Only the filename and
the panel title differ between plugins — which is why a new
evaluation view needs no frontend work at all, just an HTML file and
this declaration.

---

## 🔀 2. `single/` vs `compare/`

Like the other `multi_run` categories (`inspection`, `steering`), the
plugins come in two flavours, and the folder name decides which:

| | `single/` | `compare/` |
|---|---|---|
| Base class | `BasePlugin` | `BaseComparePlugin` |
| Sees | the current run only | the current run **and** one past run |
| Extra param | — | `past_run_id` (dropdown of eligible past runs) |
| Question | "what does *this* run look like?" | "how does this run differ from *that* one?" |

A compare plugin declares `past_run_required_steps`, and the base
class then **auto-injects** the `past_run_id` dropdown — the app
offers only past runs that actually contain a `neuron_labeling`
step. Inside `run()`, the past run's artifacts are pulled with
`self.load_past_artifact(...)` rather than through `io_spec`
(which only ever describes the *current* run).

The trick that makes a comparison meaningful: both runs' labels are
concatenated into a **single** embedding + projection pass and only
then split back apart. Two separately-computed UMAP layouts would be
incomparable — axes have no fixed meaning — so the shared pass is
what lets you read "these two points are close" as "these two labels
mean the same thing".

Two plugins are **compare-only** (`nearest_label_distance_*`): a
drift measurement has nothing to say about a single run.

---

## 📊 3. The seven plugins

### 🌳 Dendrogram (`single/dendrogram`)

Clusters the labels into a hierarchical tree by semantic similarity.
Neurons sharing the *same* label text collapse into one leaf
(`sci-fi ×4`), so the tree shows the shape of the label space
without duplicate noise.

Under the hood: embed the distinct labels → `pdist(metric="cosine")`
→ `scipy.cluster.hierarchy.linkage`. The `axis_mode` parameter
decides what the horizontal axis means — `distance` places each merge
at its true cosine distance (honest, but unreadable when a few
outliers dominate), `depth` spreads merges evenly by tree depth
(readable structure, distorted distances).

Output is an interactive Plotly page: hovering a merge lists the
labels beneath it, and a plain-text **neuron → label index** at the
bottom keeps every neuron id findable with Ctrl+F. A static
`dendrogram.pdf` with *all* ids expanded is saved alongside for
printing (download-only — it is not shown inline).

**Answers:** do the labels form coherent semantic families, or is it
one undifferentiated blob with a few outliers?

### 🗺️ Embedding Map (`single/embedding_map`, `compare/embedding_map`)

Projects the label embeddings into 2-D with **UMAP** and draws one
point per labelled neuron (no deduplication here — a concept covered
by five neurons *should* look dense). Hover shows the neuron id and
its label.

The **compare** variant projects both runs into the shared space and
colours by origin: past = orange circles, current = blue crosses,
sized so a coincident pair nests visibly instead of hiding each
other. Overlapping regions mean the runs found the same concepts;
a cluster of one colour alone marks coverage only one run has.

**Answers:** what concept space do the neurons collectively cover,
where is it dense, where are the loners — and (compare) did the two
runs converge on the same picture?

### 🔎 Embedding Map with Keyword Search (`single/…`, `compare/…`)

The same map, plus a search box for a concept *you* care about. Your
keyword is embedded **in the same batch** as the labels, so it lands
in the same coordinate space, and the closest labels are highlighted
in gold with a red cross marking the keyword itself. A sidebar lists
the top *k* matches with similarity scores; hovering a row enlarges
its point on the map.

Ranking runs on the **raw high-dimensional embeddings**, not on the
2-D coordinates — UMAP positions are a layout, and reading distances
off them would give a plausible-looking but wrong ranking.

The **compare** variant adds `search_scope`: `separate` ranks the
top *k* within each run independently (fair per-run view — "does each
run have this concept?"), `combined` pools both label sets and takes
the *k* closest overall (competitive view — "whose neurons are closer
to it?", with `current`/`past` badges showing who won).

**Answers:** does this SAE have neurons for concept X at all, and how
close does it get?

### 📏 Nearest Label Distance — Bars & Histogram (`compare/…`, compare-only)

Both take every current-run label, find its **nearest** past-run
label by cosine distance, and report how far that is — a direct
measurement of label drift. They share the computation and differ
only in presentation:

- **Bars** — one bar per neuron, sorted descending, hovering shows
  which past label was the nearest match. The per-label view: *which*
  concepts have no counterpart in the reference run (leftmost bars).
- **Histogram** — the same distances binned into a distribution. The
  aggregate view: *how many* labels stayed aligned versus drifted,
  and whether drift is a long tail or a broad shift.

Both log `mean_distance` and `median_distance` as run params, so the
two runs' alignment collapses to a single comparable number.

**Answers:** did retraining reshuffle the concept vocabulary, and if
so, uniformly or in a few specific places?

---

## 📥 4. Inputs · 📤 Outputs

**Inputs** — identical for all seven plugins:

| From | What |
|---|---|
| `neuron_labeling` | `neuron_labels.json` — neuron id → `{"label", "confidence"}`; only `label` is read |
| *(compare only)* the chosen past run | its own `neuron_labels.json`, fetched via `load_past_artifact` |

Nothing else is required, which is why this stage is cheap to re-run:
no model checkpoints, no interaction matrices, no GPU.

**Outputs** — every plugin produces exactly one inline HTML page plus
its raw data:

| Plugin | Artifacts | Run params |
|---|---|---|
| `dendrogram` | `dendrogram.html`, `dendrogram.pdf`, `linkage_matrix.npy` | `num_neurons`, `num_unique_labels` |
| `embedding_map` | `embedding_map.html`, `umap_coords.npy` | `num_neurons` |
| `embedding_map` (compare) | `embedding_map.html`, `current_umap_coords.npy`, `past_umap_coords.npy` | `num_neurons_current`, `num_neurons_past` |
| `…keyword_search` | `keyword_search_map.html`, `umap_coords.npy`, `top_k_matches.json` | `num_top_k_matches`, `num_neurons` |
| `…keyword_search` (compare) | `keyword_search_map.html`, both coord files, `top_k_matches.json` | `num_neurons_current`, `num_neurons_past` |
| `nearest_label_distance_bars` | `nearest_label_distance_bars.html`, `nearest_distances.json` | `num_neurons_current`, `num_neurons_past`, `mean_distance`, `median_distance` |
| `nearest_label_distance_histogram` | `nearest_label_distance_histogram.html`, `nearest_distances.json` | *(same four)* |

`nearest_distances.json` holds one record per current label —
`{neuron_id, label, distance, nearest_past_neuron_id,
nearest_past_label}` — so the matching can be inspected outside the
chart.

---

## 🎛️ 5. `run()` parameters

**Shared by all seven** (identical defaults and meaning):

| Param | Default | Meaning |
|---|---|---|
| `embedding_provider` | `openai` | which embedding backend turns label texts into vectors |
| `embedding_model` | `text-embedding-3-small` | the model, and therefore the semantic space |

**Shared by the four map plugins** (UMAP layout only — no effect on
any ranking):

| Param | Default | Meaning |
|---|---|---|
| `umap_n_neighbors` | 15 | low = fine local clusters, high = more global layout |
| `umap_min_dist` | 0.1 | low = tight clumps, high = evenly spread |
| `umap_metric` | `cosine` | distance metric on the high-dimensional embeddings |
| `umap_random_state` | 42 | fix for a reproducible layout |
| `point_size` | 8 | marker diameter, purely cosmetic |

**Per plugin:**

| Plugin | Param | Default | Meaning |
|---|---|---|---|
| `dendrogram` | `linkage_method` | `average` | SciPy linkage rule — how cluster distances aggregate, i.e. the tree's shape |
| `dendrogram` | `axis_mode` | `depth` | `distance` = true merge distances, `depth` = evenly spread by depth (§3) |
| `dendrogram` | `label_font_size` | 12 | leaf font size; row spacing scales with it |
| keyword search | `keyword` | *(required)* | the concept to search for; must not be empty |
| keyword search | `k` | 10 | how many closest labels to highlight (clamped to what exists) |
| keyword search (compare) | `search_scope` | `separate` | per-run top *k* vs pooled across both (§3) |
| histogram | `histogram_bins` | 30 | more bins = finer structure, noisier with few labels |
| all compare | `past_run_id` | *(required)* | the run to compare against; dropdown filled with eligible runs |

---

## 🛠️ 6. Adding your own evaluation plugin

This is the friendliest category to extend: no training, no GPU, and
one small input file. Drop a package under `single/` or `compare/`
and keep the category's shape:

- **Declare** `required_steps=["neuron_labeling"]` and the
  `neuron_labels.json` input artifact — the same `ArtifactSpec` every
  plugin here uses.
- **Embed through `embed_labels`** (`_embedding_cache.py`) rather
  than calling the embedder directly. That is what lets several
  plugins in one pipeline share a single API call — and pass the
  arguments **positionally**, or the cache silently misses.
- **Render one `text/html` artifact** and declare it in an
  `ArtifactDisplaySpec`; the app shows it inline in a sandboxed
  iframe, so a self-contained Plotly page works with no frontend
  changes.
- **For a compare plugin**, set `past_run_required_steps` (the
  `past_run_id` dropdown is injected for you) and read the other side
  with `load_past_artifact`. If your view puts both runs in one
  coordinate space, embed them in one pass and split by index.

Useful shared helpers, all `_`-prefixed so plugin discovery skips
them: `_embedding_map.py` (embed + UMAP in one call),
`_keyword_search.py` (top-*k* by cosine),
`_keyword_search_html.py` (the sidebar + plot page shell),
`_nearest_label_distance.py` (nearest-neighbour distances between two
label sets).

---

## ⚠️ 7. Operational notes & gotchas

- **🔴 Embedding calls cost money and need an API key.** Every plugin
  embeds the label set through the configured provider. A process-local
  LRU cache (4 entries, keyed by labels + provider + model) means
  running several of these plugins over the same run costs **one** call
  per distinct label set, not one per plugin — but a fresh worker
  process starts cold.
- **UMAP layouts are stochastic.** Two runs with different
  `umap_random_state` produce different-looking but equivalent maps.
  Never read absolute positions or axis values; only relative
  proximity carries meaning.
- **Unlabelled neurons are handled differently.** The dendrogram and
  the four map plugins skip neurons whose label is `None` (dead
  neurons, see the [labeling README](../neuron_labeling/README.md)).
  The two distance plugins do not — a `None` label is embedded as the
  literal string `"None"`, so expect a cluster of identical bars if
  the run has many dead neurons.
- **Compare needs a compatible past run.** The dropdown lists only
  runs containing a `neuron_labeling` step. Nothing checks that the
  two runs used the *same dataset* — comparing label sets across
  datasets will render fine and mean very little.

---

## ➡️ 8. Where to go next

- **Where the labels come from:**
  [`../neuron_labeling/README.md`](../neuron_labeling/README.md)
- **The other `multi_run` stages:**
  [`../inspection/README.md`](../inspection/README.md) (which items a
  neuron fires on),
  [`../steering/README.md`](../steering/README.md) (recommendations
  nudged by a concept)
- **The SAE whose neurons are being judged:**
  [`../training_sae/README.md`](../training_sae/README.md)
- **The plugin contract** (discovery, `io_spec`, `run()` params,
  `BaseComparePlugin`): [`../README.md`](../README.md)
