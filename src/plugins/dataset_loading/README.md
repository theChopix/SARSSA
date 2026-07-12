# 📦 Dataset Loading (`src/plugins/dataset_loading`)

> **What this is:** the first stage of every SARSSA pipeline — it turns
> a raw interaction dataset into the train/val/test matrices and item
> metadata that every later stage consumes. It is also the **most
> common extension point**: bringing your own dataset means writing one
> small plugin here.
>
> **Who should read this:** anyone who wants to **run SARSSA on their
> own data** (the headline use case — see §5, the step-by-step
> tutorial), plus contributors maintaining the loaders. The plugin
> contract in general lives in [`../README.md`](../README.md); this doc
> is the dataset-loading specifics.

---

## 📑 Table of Contents

1. [🗺️ Big picture](#-1-big-picture)
2. [🔄 The `prepare()` pipeline](#-2-the-prepare-pipeline)
3. [📤 The output contract (what every loader must produce)](#-3-the-output-contract-what-every-loader-must-produce)
4. [📚 The two reference loaders](#-4-the-two-reference-loaders)
5. [🛠️ **Tutorial: implement your own data loader**](#-5-tutorial-implement-your-own-data-loader)
6. [🗂️ The `data/` folder](#-6-the-data-folder)
7. [⚠️ Operational notes & gotchas](#-7-operational-notes--gotchas)
8. [➡️ Where to go next](#-8-where-to-go-next)

---

## 🗺️ 1. Big picture

Dataset loading is the **first stage** of a SARSSA pipeline (category
`order` 0) and runs **once per pipeline** (`type: one_time`). If the
"category", `one_time`, and `multi_run` terms are unfamiliar, the
plugins doc explains them, with the full seven-category catalogue:
[`../README.md`](../README.md) §7.

A dataset-loading plugin reads a raw dataset from disk and produces, as
MLflow artifacts, the **user × item interaction matrices** (full +
train/val/test splits) and identifier arrays that every downstream
stage — CFM training, SAE training, neuron labeling, inspection,
steering — depends on.

All the heavy lifting lives in one shared base class,
**`DatasetLoader`** (`_dataset_loader.py`; the `_` prefix means plugin
discovery skips it — it is not itself a plugin). A concrete dataset is
just a thin subclass that says *how to read its raw rows*; the base
class does filtering, matrix building, splitting, and the MLflow I/O.

```
raw file(s) in data/<dataset>/
        │   (your subclass: load_ratings)
        ▼
DatasetLoader.prepare()  ── filter ─ build CSR ─ split ─ optional data
        │
        ▼
MLflow artifacts:  users.npy  items.npy  full_csr.npz
                   train/valid/test_csr.npz  +_idx.npy  +_users.npy
        │
        ▼
training_cfm · training_sae · neuron_labeling · inspection · steering
```

Two loaders ship today: **`movieLens_loader`** (the full-featured
reference: ratings + tags + item metadata) and **`lastFm1k_loader`**
(the minimal reference: ratings only).

---

## 🔄 2. The `prepare()` pipeline

`DatasetLoader.prepare(val_ratio, test_ratio, seed)` runs a fixed
sequence. You only implement the first step; the rest is inherited:

| Step | Method | Who writes it | What it does |
|------|--------|---------------|--------------|
| 1 | `load_ratings()` | **you (abstract)** | Read your raw file(s) → `self.df_interactions`, a Polars DF with columns `userId`, `itemId` |
| 2 | `validate_interactions()` | base | Asserts the DF exists and has `userId`/`itemId` |
| 3 | `filter_interactions()` | base | Drops users/items below `MIN_USER_INTERACTIONS` / `MIN_ITEM_INTERACTIONS` |
| 4 | `build_csr_matrix()` | base | Builds `users`, `items`, `csr_interactions` |
| 5 | `split()` | base | Seeded **user-wise** split → train/valid/test CSR + index/user arrays |
| 6 | `load_optional_data()` | base hook (no-op) | Override to load tags/metadata/etc. |

`load_ratings()` is the only mandatory method **to produce the core
interaction matrices** — and that alone is enough to run *load → CFM
training → SAE training*. It is **not** enough for the *full*
pipeline: the neuron-labeling plugin that ships today
(`neuron_labeling/tf_idf`) needs **tag** artifacts, which are *not*
derivable from `userId`/`itemId` and must be supplied via a
`load_optional_data()` override (step 6). So if you want the whole
pipeline, treat tags as **effectively required, not optional** — see
the 🔴 box in §3.

Two constants tune step 3's interaction filter —
`MIN_USER_INTERACTIONS` and `MIN_ITEM_INTERACTIONS` (base defaults
20 / 200; each loader sets its own — MovieLens **5 / 1** (a threshold
of 1 disables the item filter), LastFM1k 5 / 10). A dataset with tags
adds a third, `MIN_TAG_INTERACTIONS`, which trims the **tag
vocabulary** in step 6 (MovieLens 100 — a tag must be applied at least
that many times, counted over all users, to survive). All three are
also surfaced as `run()` parameters defaulting to these constants, so
they are tunable per run from the UI without editing the subclass (see
§5 Step 3).

---

## 📤 3. The output contract (what every loader must produce)

After `prepare()`, the base class exposes everything via
`to_artifacts()` — a flat dict mapping the **canonical artifact/param
names** to values. The `Plugin` wrapper copies these onto `self.*` and
declares them in its `io_spec`, so the engine logs them to MLflow.

**Artifacts (always produced):**

| Artifact | Type | Consumed by |
|----------|------|-------------|
| `users.npy`, `items.npy` | `npy` | inspection, steering, neuron_labeling (`items`) |
| `full_csr.npz` | `npz` | steering |
| `train_csr.npz`, `valid_csr.npz`, `test_csr.npz` | `npz` | training_cfm, training_sae |
| `train_idx.npy`, `valid_idx.npy`, `test_idx.npy` | `npy` | (produced for reproducibility; no current in-repo consumer) |
| `train_users.npy`, `valid_users.npy`, `test_users.npy` | `npy` | (same) |

**Params (always produced):** `dataset_name`, `seed`, `val_ratio`,
`test_ratio`, `num_users`, `num_items`, `num_interactions`,
`num_train_users`, `num_valid_users`, `num_test_users`,
`min_user_interactions`, `min_item_interactions`,
`min_tag_interactions`, `has_tags` (consumed by the trainers). A
dataset with tags additionally logs the optional param `num_tags` —
the tag-vocabulary size after the `MIN_TAG_INTERACTIONS` filter.

### What's actually inside each file

The two file types have fixed encodings:

- **`.npy`** — one NumPy array (`np.save` / `np.load`). The *id*
  arrays hold **strings**, so they are object-dtype and consumers must
  load them with `allow_pickle=True`. The *idx* arrays are plain ints.
- **`.npz`** — a SciPy **CSR sparse matrix**
  (`scipy.sparse.save_npz` / `load_npz`), `dtype=float32`, and
  **binary**: an entry is `1.0` if the user interacted with the item,
  absent otherwise. There are **no rating values** — interactions are
  implicit (e.g. the MovieLens reader already collapses ratings ≥ 4
  into "interacted").

What each holds and how they line up:

| File | Encoding | Shape | Contents |
|------|----------|-------|----------|
| `users.npy` | 1-D str array | `(num_users,)` | every user id; **its position = that user's row index** in `full_csr` |
| `items.npy` | 1-D str array | `(num_items,)` | every item id; **its position = the column index** in every CSR |
| `full_csr.npz` | CSR `float32` | `(num_users, num_items)` | the full matrix; `M[u, i] == 1.0` ⇔ `users[u]` interacted with `items[i]` |
| `train_csr` / `valid_csr` / `test_csr` `.npz` | CSR `float32` | `(num_<split>_users, num_items)` | a **row subset** of `full_csr`; **same item columns** |
| `train_idx` / `valid_idx` / `test_idx` `.npy` | 1-D int array | `(num_<split>_users,)` | the original row positions this split took: `train_csr == full_csr[train_idx]` |
| `train_users` / `valid_users` / `test_users` `.npy` | 1-D str array | `(num_<split>_users,)` | that split's user ids: `train_users == users[train_idx]` |

The split is **user-disjoint and seeded**: every user lands in exactly
one of train/valid/test, and the same `seed` reproduces the identical
partition. **Items are never split** — all three CSRs span the full
item set; only the rows (users) differ. The **params** are plain
scalars logged to MLflow (ints / floats / strings / the `has_tags`
bool) and read back downstream via typed `ParamSpec`s.

### ⚠️ The optional tag/metadata artifacts

MovieLens additionally declares three **optional** artifacts
(`optional=True` in its `io_spec`) — they are skipped when the
attribute holds `None`:

| Optional artifact | Produced when | Required by |
|----------------|---------------|-------------|
| `tag_ids.json` | the dataset has tags | **`neuron_labeling/tf_idf`** |
| `tag_item_matrix.npz` | the dataset has tags | **`neuron_labeling/tf_idf`** |
| `item_metadata.json` | item metadata available | backend item enrichment (UI item cards) |

> **`tag_item_matrix.npz` holds counts, not 0/1.** It is a
> `(num_tags, num_items)` CSR of tag↔item **co-occurrence counts**,
> **restricted to the train users** (vocabulary trimmed by
> `MIN_TAG_INTERACTIONS` first) — counts, not mere incidence.

This matters enormously for "bring your own dataset":

> **🔴 `neuron_labeling` (and therefore inspection & steering) needs
> `tag_ids.json` + `tag_item_matrix.npz`.** A dataset with **no tags**
> can load and train CFM/SAE, but the pipeline will then **fail at
> `neuron_labeling` with `MissingContextError`**. To run the full
> interpretability pipeline your dataset must provide tags (or
> `neuron_labeling/tf_idf` must be adapted). Likewise, UI item cards
> only render if you emit `item_metadata.json`.

---

## 📚 4. The two reference loaders

**`movieLens_loader` — the template to follow (the default).**
`load_ratings()` reads `ratings.csv` (keeps ratings ≥ 4.0 as positive
implicit feedback); `load_optional_data()` reads `tags.csv` (dropping
tags applied fewer than `MIN_TAG_INTERACTIONS` times) and
`metadata.json`; it overrides `get_item_metadata()` (renaming
`genres` → `categories`) and declares the tag + metadata extras as
optional `io_spec` outputs. **This is the shape you should model your own loader
on** — it yields the full, working pipeline.

**`lastFm1k_loader` — the reduced fallback, *not* the recommended
shape.** `load_ratings()` reads a tab-separated `ratings.tsv`, skips a
handful of known-corrupt rows, treats every listening event as
positive feedback, and provides **no tags, no metadata**. Only model
on this if you *deliberately* want trained models and nothing more —
it cannot run neuron labeling / inspection / steering (the 🔴 box in
§3). Treat it as a cautionary example, not a starting point.

Both wrap their `DatasetLoader` subclass in a `Plugin(BasePlugin)`
whose `run(seed, val_ratio, test_ratio)` calls `prepare()` and copies
the results onto `self.*` matching the `io_spec`.

---

## 🛠️ 5. Tutorial: implement your own data loader

This is the headline workflow: **run SARSSA on your own dataset.** You
write one folder with one file.

The steps below use the shared `DatasetLoader` base class — the easy
path for the common case. **You are not required to use it.** If your
data or preparation logic is very different, implement it however you
like; only the *output contract* is binding — see *Going off-script*
at the end of this section.

### Step 1 — Create the plugin folder

Plugin discovery requires a directory whose name matches its module,
containing a class literally named `Plugin`:

```
src/plugins/dataset_loading/<your_dataset>_loader/
├── __init__.py                 # empty
└── <your_dataset>_loader.py    # file stem == folder name
```

### Step 2 — Subclass `DatasetLoader`: ratings + tags + metadata

You declare one file constant per input file and write a small reader
for each. The default (MovieLens-style) loader handles **all three**
files from §6; `load_ratings()` is mandatory, `load_optional_data()`
and `get_item_metadata()` cover tags/metadata. Each reader just maps
*your* raw columns to the canonical schema — see §6 for the exact
file formats.

**Three non-negotiable rules for `load_ratings()`** (the code below
obeys all three):

1. **Implicit feedback only — no ratings survive.** SARSSA builds a
   *binary* interaction matrix; `build_csr_matrix()` only ever writes
   `1.0` and there is nowhere to store a score. If your raw data has
   ratings / scores / play-counts, **threshold them into "interacted
   or not" and drop the score column** (MovieLens keeps
   `rating >= 4.0`; LastFM1k treats every play as a positive).
2. **Rename your raw columns to the canonical names.** The result must
   have exactly `userId` and `itemId`; your source columns can be
   named anything. `itemId` is whatever *you* define as "the item"
   (MovieLens → `movieId`; LastFM1k → the artist MBID).
3. **Both columns must be `pl.String`.** `build_csr_matrix()` converts
   them to Polars *Categoricals* to derive the `users` / `items`
   arrays and the CSR row/column codes — a non-string dtype breaks
   that. Cast explicitly.

```python
import json

import polars as pl
from plugins.dataset_loading._dataset_loader import DatasetLoader


class MyDatasetLoader(DatasetLoader):
    # tune the interaction-count filters for your data
    MIN_USER_INTERACTIONS: int = 5
    MIN_ITEM_INTERACTIONS: int = 10
    MIN_TAG_INTERACTIONS: int = 100   # min times a tag is applied to be kept

    # one literal per input file (model: MovieLens)
    RATINGS_FILE: str = "ratings.csv"
    TAGS_FILE: str = "tags.csv"
    METADATA_FILE: str = "metadata.json"

    def __init__(self, data_dir: str = "../data/my_dataset") -> None:
        super().__init__("MyDataset", data_dir)

    # REQUIRED: produce self.df_interactions with columns userId, itemId
    def load_ratings(self) -> None:
        if not self._file_exists(self.RATINGS_FILE):
            raise FileNotFoundError(
                f"{self._resolve_path(self.RATINGS_FILE)} not found."
            )
        self.df_interactions = (
            pl.scan_csv(self._resolve_path(self.RATINGS_FILE), has_header=True)
            .select(["user", "item", "rating"])        # your raw column names
            .rename({"user": "userId", "item": "itemId"})
            .cast({"userId": pl.String, "itemId": pl.String, "rating": pl.Float64})
            .filter(pl.col("rating") >= 4.0)           # keep positives; drop if N/A
            .select(["userId", "itemId"])
            .unique()
            .sort(["userId", "itemId"])
            .collect()
        )

    # REQUIRED for the full pipeline: tags → self.df_tags (userId, itemId, tag)
    def load_optional_data(self) -> None:
        if self._file_exists(self.TAGS_FILE):
            self.df_tags = (
                pl.scan_csv(self._resolve_path(self.TAGS_FILE), has_header=True)
                .select(["user", "item", "tag"])       # your raw column names
                .rename({"user": "userId", "item": "itemId"})
                .cast({"userId": pl.String, "itemId": pl.String, "tag": pl.String})
                .with_columns(pl.col("tag").str.to_lowercase().str.strip_chars())
                # drop rare tags (counted over all users, before the item filter)
                .filter(pl.col("tag").count().over("tag") >= self.MIN_TAG_INTERACTIONS)
                .filter(pl.col("itemId").is_in(self.items))
                # NOT de-duplicated: keeping repeated (user, item, tag) rows is
                # what makes tag_item_matrix hold co-occurrence counts, not 0/1
                .collect()
            )
        if self._file_exists(self.METADATA_FILE):
            with open(self._resolve_path(self.METADATA_FILE)) as f:
                self.metadata = json.load(f)            # {itemId: {...}}

    # REQUIRED for UI cards: per-item display dict, keys the UI expects
    def get_item_metadata(self) -> dict[str, dict]:
        if self.metadata is None or self.items is None:
            return {}
        keep = set(self.items)
        return {
            i: {"title": m["title"], "year": m["year"],
                "categories": m.get("genres", []), "image_url": m.get("image_url")}
            for i, m in self.metadata.items() if i in keep
        }
```

`prepare()` does the rest — filtering, the CSR matrix, the seeded
split, and it calls `load_optional_data()` for you. You do **not**
touch MLflow here. (The two tag-helper methods — `tag_ids()` and
`tag_item_matrix()` — and the optional `io_spec` entries that turn
`self.df_tags` into the `tag_ids.json` / `tag_item_matrix.npz` /
`item_metadata.json` artifacts are shown in Step 5.)

### Step 3 — Wrap it in a `Plugin`

The `Plugin` declares the output contract (§3) and copies the prepared
values onto `self`. This `io_spec` is the **same whether or not you
have tags** — the tag/metadata extras are declared as `optional=True`
entries (Step 5) that are simply skipped when the value is `None`:

```python
from typing import Annotated

from plugins.plugin_interface import (
    BasePlugin, OutputArtifactSpec, OutputParamSpec, ParamGroup, PluginIOSpec,
)


class Plugin(BasePlugin):
    name = "My Dataset Loader"

    io_spec = PluginIOSpec(
        output_artifacts=[
            OutputArtifactSpec("users", "users.npy", "npy"),
            OutputArtifactSpec("items", "items.npy", "npy"),
            OutputArtifactSpec("full_csr", "full_csr.npz", "npz"),
            OutputArtifactSpec("train_csr", "train_csr.npz", "npz"),
            OutputArtifactSpec("valid_csr", "valid_csr.npz", "npz"),
            OutputArtifactSpec("test_csr", "test_csr.npz", "npz"),
            OutputArtifactSpec("train_idx", "train_idx.npy", "npy"),
            OutputArtifactSpec("valid_idx", "valid_idx.npy", "npy"),
            OutputArtifactSpec("test_idx", "test_idx.npy", "npy"),
            OutputArtifactSpec("train_users", "train_users.npy", "npy"),
            OutputArtifactSpec("valid_users", "valid_users.npy", "npy"),
            OutputArtifactSpec("test_users", "test_users.npy", "npy"),
        ],
        output_params=[
            OutputParamSpec(k, k) for k in (
                "dataset_name", "seed", "val_ratio", "test_ratio",
                "num_users", "num_items", "num_interactions",
                "num_train_users", "num_valid_users", "num_test_users",
                "min_user_interactions", "min_item_interactions",
                "min_tag_interactions", "has_tags",
            )
        ],
        param_groups=[   # optional: lay the params out as collapsible UI sections
            ParamGroup("Data split", ["val_ratio", "test_ratio", "seed"]),
            ParamGroup("Filtering thresholds", [
                "min_user_interactions", "min_item_interactions",
                "min_tag_interactions",
            ]),
        ],
    )

    def run(
        self,
        seed: Annotated[int, "Seed for the train/val/test split."] = 42,
        val_ratio: Annotated[float, "Per-user validation fraction."] = 0.1,
        test_ratio: Annotated[float, "Per-user test fraction."] = 0.1,
        min_user_interactions: Annotated[
            int, "Drop users with fewer interactions than this."
        ] = MyDatasetLoader.MIN_USER_INTERACTIONS,
        min_item_interactions: Annotated[
            int, "Drop items with fewer interactions than this (1 keeps all)."
        ] = MyDatasetLoader.MIN_ITEM_INTERACTIONS,
        min_tag_interactions: Annotated[
            int, "Drop tags applied fewer times than this across all users."
        ] = MyDatasetLoader.MIN_TAG_INTERACTIONS,
    ) -> None:
        loader = MyDatasetLoader()
        # Thresholds default to the class constants; the loader reads them
        # via self.MIN_* during prepare(), so they're tunable per run.
        loader.MIN_USER_INTERACTIONS = min_user_interactions
        loader.MIN_ITEM_INTERACTIONS = min_item_interactions
        loader.MIN_TAG_INTERACTIONS = min_tag_interactions
        loader.prepare(seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
        # to_artifacts() returns every canonical name → value in one dict;
        # its keys are exactly the io_spec attr names above.
        for attr, value in loader.to_artifacts().items():
            setattr(self, attr, value)
        # to_artifacts() covers the interaction thresholds but not the
        # tag-vocabulary one, so log that separately.
        self.min_tag_interactions = min_tag_interactions
```

> The shipped loaders assign each attribute by hand instead of looping
> over `to_artifacts()`; both are equivalent — the loop is just less
> boilerplate. Whichever you choose, **every `io_spec` name must end up
> as a `self.<attr>`**.

### Step 4 — Put your raw data in place

Drop your file at `data/<your_dataset>/ratings.csv` (the `data/`
folder is gitignored — see §6). `data_dir` defaults to
`../data/<your_dataset>` because the app runs from `src/`; the path is
resolved relative to the **process working directory**, not this file.

### Step 5 — Tags & item metadata (required for the full pipeline)

This step is **skippable only if you stop after SAE training.** With
the plugins that ship today, the **full pipeline (neuron labeling →
inspection → steering) requires tag artifacts**, and **UI item cards
require item metadata** — a ratings-only loader will run *load → CFM →
SAE* and then fail at `neuron_labeling` with `MissingContextError`
(re-read the 🔴 box in §3). So unless you deliberately only want the
trained models, your dataset **must** also provide tags (and metadata
for the UI).

You already did half of this in Step 2: `load_optional_data()`
populated `self.df_tags` / `self.metadata`, and `get_item_metadata()`
shapes the per-item dict. What remains is (a) two tag-helper methods
on the loader and (b) declaring the extras as **optional outputs** in
the `Plugin`'s `io_spec` — the base `update_context()` skips any
`optional=True` entry whose attribute holds `None`, so the same spec
works with and without tags:

```python
import numpy as np, scipy.sparse as sp

# --- on MyDatasetLoader (the DatasetLoader subclass) ---
def tag_ids(self) -> list[str]:
    return self.df_tags["tag"].unique().sort().to_list()

def tag_item_matrix(self, user_subset: np.ndarray | None = None) -> sp.csr_matrix:
    # Vocabulary spans all users; user_subset restricts the counts to those
    # users (pass train_users so val/test tags don't leak). Rows aren't
    # de-duplicated (see load_optional_data), so equal (tag, item) pairs sum
    # into co-occurrence counts.
    tids = self.tag_ids()
    t_idx = {t: i for i, t in enumerate(tids)}
    i_idx = {i: j for j, i in enumerate(self.items)}
    df_tags = self.df_tags
    if user_subset is not None:
        df_tags = df_tags.filter(pl.col("userId").is_in([str(u) for u in user_subset]))
    rows = [t_idx[r["tag"]] for r in df_tags.iter_rows(named=True)]
    cols = [i_idx[r["itemId"]] for r in df_tags.iter_rows(named=True)]
    return sp.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)),
        shape=(len(tids), len(self.items)),
    )

# --- on Plugin: add to io_spec (next to the §3 outputs) ---
#     OutputArtifactSpec("_tag_ids", "tag_ids.json", "json", optional=True),
#     OutputArtifactSpec("_tag_item_matrix", "tag_item_matrix.npz", "npz", optional=True),
#     OutputArtifactSpec("_item_metadata", "item_metadata.json", "json",
#                        optional=True, saver_kwargs={"indent": None}),
#     ...
#     OutputParamSpec("num_tags", "num_tags", optional=True),

# --- on Plugin: in run(), after loader.prepare(), stash these ---
#     (initialise all of them to None first, then fill when available)
#     self._tag_ids         = loader.tag_ids()
#     # restrict the co-occurrence counts to the train users
#     self._tag_item_matrix = loader.tag_item_matrix(user_subset=loader.train_users)
#     self.num_tags         = len(self._tag_ids)
#     self._item_metadata   = loader.get_item_metadata() or None
```

This mirrors what `movieLens_loader` does — copy it and adjust.

### Step 6 — Checklist

1. Folder and file share the dataset name; class is named `Plugin`;
   empty `__init__.py`.
2. `load_ratings()` produces a Polars DF with **only** `userId`,
   `itemId` (both `pl.String`).
3. `MIN_USER_INTERACTIONS` / `MIN_ITEM_INTERACTIONS` sane for your
   data (too high → empty matrix).
4. Every `io_spec` artifact/param name is set on `self`.
5. Raw file present at `data/<your_dataset>/…`.
6. Tags/metadata added **if** you need neuron_labeling / UI cards.
7. Sanity check: `GET /plugins/registry` lists it under
   `dataset_loading`, and a pipeline run produces `train_csr.npz` etc.
   on the step's MLflow run.

### Going off-script — you don't have to subclass `DatasetLoader`

`DatasetLoader` is a **convenience for the common case, not a
requirement**. Nothing in the engine knows or cares about it — the
engine only drives the `Plugin` lifecycle (`load_context → run →
update_context`) and reads the `io_spec`. So if your situation is
different — a database or streaming source, a custom split strategy,
data that arrives already split, a totally different preparation
algorithm — you are free to **implement `run()` entirely your own
way** and skip `DatasetLoader`, `prepare()`, and `load_ratings()`
altogether.

What is **binding** is only the *output contract*: by the end of
`run()`, the same artifacts and params from §3 must be on `self.*` —
**same names, same formats** (`*_csr.npz` as SciPy CSR matrices,
`*.npy` as NumPy arrays, the params as listed) — because downstream
stages address them by name and type. Produce those correctly and the
pipeline neither knows nor cares how you built them. (Need the
tag/metadata extras for `neuron_labeling` / UI cards? Emit those too —
see the 🔴 box in §3.)

Rule of thumb: **lean on `DatasetLoader` when it helps, ignore it when
it doesn't — but never deviate from the §3 output contract.**

---

## 🗂️ 6. The `data/` folder

Loaders read from `data/<dataset>/` at the **repo root** (resolved as
`../data/<dataset>` from the `src/` working directory). This folder is
**gitignored**, so you create it locally — the structure below is not
in the repo, you build it yourself.

### Expected layout (the default — model your dataset on this)

The **expected, fully-functional shape** is the MovieLens shape: an
interactions file **plus tags plus item metadata**. This is the
default this tutorial assumes — *not* the ratings-only LastFM1k shape,
which gives you a reduced pipeline (see the 🔴 box in §3).

```
data/<your_dataset>/
├── ratings.csv      REQUIRED — the user–item interactions
├── tags.csv         REQUIRED for the full pipeline — per-item characteristics
└── metadata.json    REQUIRED for UI item cards — per-item display info
```

(`descriptions.json` is *not* needed — the code loads it but nothing
uses it; don't bother creating one.)

### What each file looks like

The exact column/key names below are what the **MovieLens reader**
expects; your loader's `load_ratings()` / `load_optional_data()` can
rename your raw columns to these (see §5 Step 2), so you have freedom
in your raw schema as long as your reader maps it.

**`ratings.csv`** — one row per interaction. CSV with a header
(TSV also fine — `lastFm1k` uses tab-separated). MovieLens columns:

```
userId,movieId,rating,timestamp
1,17,4.0,944249077
1,25,1.0,944250228
```

The reader keeps `userId`, `movieId` (→ `itemId`), and `rating`,
treats `rating >= 4.0` as a **positive implicit interaction**, and
drops the rest. *What it's for:* this is the core signal — who
interacted with what — from which all train/val/test matrices are
built. The bare minimum a dataset can provide.

**`tags.csv`** — the **array of characteristics of each item**. CSV
with a header; many rows per item (one row = one `(item, tag)` pair),
free-text tags. MovieLens columns:

```
userId,movieId,tag,timestamp
22,26479,Kevin Kline,1583038886
22,79592,misogyny,1581476297
22,247150,acrophobia,1622483469
```

The reader keeps `movieId` (→ `itemId`) and `tag` (lower-cased,
trimmed). *What it's for:* tags are the **human-readable vocabulary
that describes items** — "this movie is *noir*, *dystopian*,
*based-on-a-book*". `neuron_labeling/tf_idf` correlates SAE-neuron
activations against this tag↔item structure to name each neuron's
concept. **No tags ⇒ no neuron labels ⇒ no inspection/steering** (the
🔴 box in §3). Conceptually: ratings say *who liked what*; tags say
*what each item is about*.

**`metadata.json`** — per-item display info: a JSON object **keyed by
item id** (the string key must match the `itemId` values your
`ratings`/`tags` files use). Each value is a small object with this
**canonical schema** (the names the frontend actually reads):

| Field | Required? | Type | Notes |
|-------|-----------|------|-------|
| `title` | **Required** | string | The only must-have. If absent, the UI falls back to showing the raw item id |
| `year` | optional | int | Omit entirely if it doesn't apply to your domain |
| `categories` | optional | `string[]` | Labels shown on the card |
| `image_url` | optional | string | Poster / cover image |

Use `categories` directly — that is the canonical name. (You map your
raw fields to these names inside `get_item_metadata()`. MovieLens's
raw file happens to call it `genres`, so its `get_item_metadata()`
renames `genres` → `categories`; you don't need to mirror that
quirk — just emit `categories`.)

Example — movies (every field present):

```json
{
  "1":  { "title": "Toy Story (1995)", "year": 1995, "categories": ["Animation", "Comedy"], "image_url": "https://img.example/toy-story.jpg" },
  "10": { "title": "GoldenEye (1995)", "year": 1995, "categories": ["Action"] }
}
```

Example — a different domain, clothing (**no `year`** — it doesn't
apply, and that is fine because it's optional; the second item also
omits `image_url`):

```json
{
  "sku_4471": {
    "title": "Merino Wool Crew Sweater",
    "categories": ["knitwear", "sweaters", "wool", "men"],
    "image_url": "https://img.example/store/sku_4471.jpg"
  },
  "sku_9020": {
    "title": "Waxed Cotton Field Jacket",
    "categories": ["outerwear", "jackets", "waterproof"]
  }
}
```

*What it's for:* purely presentation — the web UI's item cards show
`title` / `year` / `categories` / `image_url`. It does **not** affect
training or labeling: omit `metadata.json` entirely and the pipeline
still runs; you just get bare item IDs in the UI instead of
titles/images.

### Download helpers (for the shipped datasets only)

Two script families exist; **prefer the `_all` ones** — they fetch a
single OSF bundle with the complete, ready-to-use artifact set:

- **`scripts/download_movieLens_all.sh`** *(recommended)* → OSF bundle
  → extracts the **full set** into `data/movieLens/`: `ratings.csv`,
  `tags.csv`, `metadata.json`, `descriptions.json`. This is the one
  that yields the full pipeline **and** UI item cards in a single step.
- **`scripts/download_lastFm1k_all.sh`** *(recommended)* → OSF bundle →
  `data/lastFm1k/ratings.tsv`.
- `scripts/download_movieLens_dataset.sh` → fetches the **raw** dataset
  from GroupLens: only `ratings.csv` + `tags.csv` (raw leftovers →
  `data/movieLens/raw/`). It does **not** produce `metadata.json` —
  use the `_all` script if you want UI item cards (see the §7 note).
- `scripts/download_lastFm1k_dataset.sh` → raw LastFM-1K from
  mtg.upf.edu → `data/lastFm1k/ratings.tsv` (same end result as the
  `_all` variant, different source).

---

## ⚠️ 7. Operational notes & gotchas

Real, code-traced sharp edges — especially relevant when adding a
dataset.

- **🔴 No tags → pipeline stops at `neuron_labeling`.** `tag_ids.json`
  and `tag_item_matrix.npz` are produced only when the dataset has
  tags (today: MovieLens only). Without them, CFM/SAE training still
  succeeds but `neuron_labeling/tf_idf` fails with
  `MissingContextError`, which blocks inspection and steering too.
- **`metadata.json` only comes from the `_all` download script.** The
  UI's item cards need `item_metadata.json`, which the loader derives
  from `metadata.json`. The raw `download_movieLens_dataset.sh` fetches
  only `ratings.csv`/`tags.csv`, so a dataset provisioned *that* way
  has **no UI item enrichment**. Use
  `scripts/download_movieLens_all.sh` (the OSF bundle — it includes
  `metadata.json` and `descriptions.json`) for the full UI experience.
- **`descriptions.json` is loaded but never used.** MovieLens reads it
  into `self.descriptions` and nothing consumes it — a ~52 MB no-op
  read. `item_descriptions.json` in the same folder is referenced by
  no code at all.
- **The tag/metadata artifacts are `optional=True` in `io_spec`.**
  They are skipped when the attribute holds `None`, so a run's actual
  artifact set depends on what the dataset provides — the spec lists
  the *maximum*, not the guaranteed output.
- **`data_dir` is relative to the process CWD.** It defaults to
  `../data/<dataset>` and only resolves correctly because the app runs
  from `src/`. Running a loader from another directory needs an
  absolute `data_dir`.
- **Name collision: `utils.data_loading.DataLoader` is unrelated.**
  That class is a *training-time mini-batch iterator* over the CSR
  matrices (used by the trainers and torch eval), **not** part of
  dataset-loading. Don't confuse the two.
- **The split is user-wise and seeded.** `split()` permutes *users*
  (not interactions) with `np.random.seed(seed)`; a user appears in
  exactly one of train/valid/test. Same `seed` → same partition.

---

## ➡️ 8. Where to go next

- 🔌 **The plugin contract in general:**
  [`../README.md`](../README.md) — `BasePlugin`, `PluginIOSpec`,
  discovery, the write-your-own-plugin tutorial.
- ⚙️ **What consumes these artifacts:**
  [`../../app/README.md`](../../app/README.md) — the pipeline engine,
  context wiring, and MLflow integration.
- 🧪 **Downstream stages:** the `training_cfm/`, `training_sae/`,
  `neuron_labeling/`, `inspection/`, `steering/` READMEs (per-category
  contracts).
- 📘 **Project overview, setup & Docker:**
  [root `README.md`](../../../README.md).
