# 🎚️ Steering (`src/plugins/steering`)

> **What this is:** the seventh and last stage of SARSSA — and the
> only one that *acts* instead of observing. Everything upstream
> exists so that this can work: a recommender is trained, an SAE
> decomposes its user embeddings into interpretable concepts, those
> concepts get labels. Steering then **turns one of them up** for a
> chosen user and shows what the recommendations become.
>
> **Who should read this:** anyone running the payoff demo — "make
> this user's feed more *sci-fi* and see what happens" — and
> contributors adding a steering view (§6). The plugin contract in
> general lives in [`../README.md`](../README.md); this doc is the
> steering specifics.

---

## 📑 Table of Contents

1. [🗺️ Big picture: the category contract](#-1-big-picture-the-category-contract)
2. [🎚️ How steering actually works](#-2-how-steering-actually-works)
3. [🔀 `single/` vs `compare/`](#-3-single-vs-compare)
4. [📊 The plugins](#-4-the-plugins)
5. [📥 Inputs · 📤 Outputs](#-5-inputs--outputs)
6. [🎛️ `run()` parameters](#-6-run-parameters)
7. [🛠️ Adding your own steering plugin](#-7-adding-your-own-steering-plugin)
8. [⚠️ Operational notes & gotchas](#-8-operational-notes--gotchas)
9. [➡️ Where to go next](#-9-where-to-go-next)

---

## 🗺️ 1. Big picture: the category contract

Every plugin here runs the same experiment:

```
one user  ──▶  their interaction vector  ──▶  base model     ──▶  original top-k
                        │                                              (baseline)
                        └──────────────────▶  steered model  ──▶  steered top-k
                                              (concept boosted)
```

Three things define the category:

- **It is a controlled experiment, not a single answer.** The output
  is always the *pair*: what the model recommends normally, and what
  it recommends when one concept is amplified — plus the user's
  interaction history as context for both. A steered list on its own
  says nothing; the difference between the two lists is the result.
- **It is the only stage that loads models.** Unlike
  [`inspection`](../inspection/README.md), which reads precomputed
  activations, steering runs real inference: it needs the CFM
  checkpoint *and* the SAE checkpoint, and it is therefore the
  heaviest `multi_run` stage — though still seconds, since it is one
  forward pass for one user.
- **Results are items, not charts.** Like `inspection`, the output is
  lists of item ids rendered as rows of item cards, declared in
  `io_spec` (§3 of [`../README.md`](../README.md) is the full
  `PluginIOSpec` field reference):

  ```python
  display=ItemRowsDisplaySpec(
      type="item_rows",
      rows=[
          DisplayRowSpec("interacted_items", "Interaction History"),
          DisplayRowSpec("original_recommendations", "Original Recommendations"),
          DisplayRowSpec("steered_recommendations", "Steered Recommendations"),
      ],
  ),
  ```

  The frontend joins each row's item ids with the dataset's
  `item_metadata.json` and renders posters and titles — so the effect
  of steering is something you *see* rather than compute.

---

## 🎚️ 2. How steering actually works

The user's interaction vector goes through the base model into an
embedding, and through the SAE into a sparse activation vector `e` —
one entry per concept neuron. Steering rewrites that vector before
decoding it back into item scores. With `s` the user's total
activation and `alpha` the strength:

```
e *= (1 - alpha) / s   # shrink everything the user already is
e[n] += alpha          # hand the freed share to the target neuron n
e *= s                 # restore the original total magnitude
```

`alpha` is the share of the representation handed to the concept;
`1 - alpha` stays with the user's own profile. At 0 the vector is
untouched, at 1 nothing of the user remains. (`SteeredModel` can split
the same share across several target neurons; today's plugins always
pass one, hence the plain `+= alpha` — see §7.)

The steered vector is then decoded back through the SAE and the base
model into item scores, already-seen items are masked out, and the
top *k* is returned. The baseline list comes from the same model with
no steering at all, so the two lists differ only by the concept nudge.

---

## 🔀 3. `single/` vs `compare/`

Like the other `multi_run` categories
([`labeling_evaluation`](../labeling_evaluation/README.md),
[`inspection`](../inspection/README.md)), plugins come in two
flavours and the folder name decides which:

| | `single/` | `compare/` |
|---|---|---|
| Base class | `BasePlugin` | `BaseComparePlugin` |
| Sees | the current run only | the current run **and** one past run |
| Extra params | — | `past_run_id`, `past_neuron_id` |
| Question | "what does boosting this concept do?" | "do two runs steer the same user the same way?" |

A compare plugin declares `past_run_required_steps`, and the base
class **auto-injects** the `past_run_id` dropdown. The past run's
artifacts are loaded with `self.load_past_artifact(...)` rather than
through `io_spec` (which only ever describes the *current* run).

Two constraints are worth understanding, and they pull in opposite
directions:

- **The neurons are chosen independently** — one dropdown for the
  current run, one for the past run — because neuron ids mean nothing
  across runs. SAE training is stochastic, so neuron 42 in two runs
  is two unrelated directions sharing an index; the comparison worth
  making is *"the neuron labelled sci-fi in each run"*.
- **The dataset must be the same.** Unlike `inspection`, where each
  side happily resolves against its own catalogue, here `user_id` is
  a **row index** into the interaction matrix — index 184676 is a
  different person in a different dataset. The compare plugin
  therefore verifies the two runs' `users.npy` are identical and
  fails fast with an explicit message if they are not, before loading
  any model.

---

## 📊 4. The plugins

### 🎚️ SAE Steering (`single/sae_steering`)

Pick a user, pick a concept neuron, set the strength, and get three
rows: the user's interaction history, the baseline recommendations,
and the steered ones.

Two dropdowns carry most of the UX here:

- **The neuron dropdown** is filled from `neuron_labels.json` and
  shows `sci-fi [neuron id 1319] · conf 0.701`, tinted by the
  labeling confidence — the same widget as in `inspection`.
- **The user dropdown is searched on the server.** Datasets run to
  hundreds of thousands of users, far too many to render at once, so
  typing queries the backend and the first 200 matches come back.
  Each option reads `user 87007 · 5525 interactions [row 184676]` —
  original id, activity, and the row index the plugin actually
  consumes — tinted by interaction count, with an optional
  *Sort by interaction count* that ranks **all** matches, not just the
  loaded page. Users with rich histories make the most legible demos,
  so they are easy to find deliberately.

The `alpha` slider (0 → 1) is where the experiment lives:
sweep it and watch how quickly the user's own taste gives way to the
concept.

**Answers:** what does this concept do to this user's feed, and how
hard do you have to push before it takes over?

### 🎚️ SAE Steering — compare (`compare/sae_steering`)

The same experiment run on two pipelines at once, for the same user,
each steered toward its own chosen neuron. Because both runs share
the dataset, the interaction history is identical and shown **once**;
the four recommendation lists are then interleaved so each pair sits
together:

```
Interaction History
Original Recommendations — Current Run
Original Recommendations — Past Run
Steered Recommendations — Current Run
Steered Recommendations — Past Run
```

The interleaving is the point: the two *Original* rows show whether
the recommenders agree at baseline, and the two *Steered* rows show
whether the concepts move them the same way. A pair that agrees at
baseline but diverges under steering isolates the SAE as the source
of the difference.

**Answers:** did the newly trained SAE learn a concept that steers
better — or just differently — than the previous one?

---

## 📥 5. Inputs · 📤 Outputs

**Inputs** — identical for both plugins, and the heaviest set in the
project: this is the only stage needing all four upstream steps.

| From | What |
|---|---|
| `dataset_loading` | `full_csr.npz` (the interaction matrix), `users.npy`, `items.npy` |
| `training_cfm` | the base recommender checkpoint |
| `training_sae` | the SAE checkpoint |
| `neuron_labeling` | `neuron_labels.json` — fills the dropdown, supplies the label |
| *(compare only)* the chosen past run | all of the above, via `load_past_artifact` |

**Outputs:**

| Plugin | Artifacts | Run params |
|---|---|---|
| `sae_steering` | `interacted_items.json`, `original_recommendations.json`, `steered_recommendations.json` | `user_original_id`, `label` |
| `sae_steering` (compare) | `current_` / `past_` versions of all three | `user_original_id`, `past_user_original_id`, `label`, `past_label` |

`user_original_id` records the dataset's own id for the row index
that was steered, and `label` the concept's name — so a run stays
readable long after you have forgotten which index you picked.

---

## 🎛️ 6. `run()` parameters

| Plugin | Param | Default | Meaning |
|---|---|---|---|
| both | `user_id` | *(required)* | 0-based row index of the user to steer, picked from the server-searched dropdown |
| both | `neuron_id` | *(required)* | the concept neuron to amplify, picked from the label dropdown |
| both | `alpha` | 0.3 | steering strength in [0, 1] — the share of the representation handed to the concept (§2) |
| both | `k` | 10 | number of recommendations per list |
| compare | `past_run_id` | *(required)* | the past run to compare against; must use the same dataset (§3) |
| compare | `past_neuron_id` | *(required)* | the neuron on the past side, chosen from *that* run's labels (§3) |

`alpha` outside [0, 1], an unknown `neuron_id`, or a `user_id` beyond
the matrix all raise before any model work happens.

---

## 🛠️ 7. Adding your own steering plugin

The analytical core is one function —
`compute_steered_recommendations` in `_steer.py` (underscore = not a
plugin) — which both current plugins call. Reuse it and a new view is
mostly `io_spec` plus presentation.

One capability is already built and unused: `SteeredModel.recommend`
accepts a **list** of neuron ids and splits the `alpha` budget
equally among them, but both plugins pass a single-element list. A
multi-concept steering plugin ("boost *sci-fi* **and** *noir*") needs
no model work at all — just a parameter that collects several neurons
and passes them through.

Other ideas that fit: sweeping `alpha` and showing the lists at
several strengths side by side, steering a *cohort* rather than one
user, or quantifying the effect (overlap between baseline and steered
top-k) instead of displaying it.

Keep the category's shape:

- **Declare all four `required_steps`** and the six input artifacts
  above; steering genuinely needs the models.
- **Emit item ids** and declare them as `ItemRowsDisplaySpec` rows to
  get enriched cards for free — and always include the baseline row,
  or the result is uninterpretable.
- **For a compare plugin**, set `past_run_required_steps`, verify the
  datasets match before doing any work, and give past-side params
  their own dropdown cascading off `past_run_id`
  (`source_run_param`).

---

## ⚠️ 8. Operational notes & gotchas

- **`alpha = 1` erases the user.** At full strength the entire
  activation budget sits on the target neuron, so every user steered
  toward the same concept gets nearly the same recommendations. For a
  demo that still looks personal, stay well below 1 — the 0.3 default
  is a reasonable starting point.
- **Users with an empty or near-empty profile steer badly.** The
  formula divides by the user's total activation (guarded against
  exactly zero), so a user with almost no interactions has almost no
  profile to trade away and the result is dominated by noise. The
  dropdown's interaction counts are there partly for this reason.
- **Dead neurons are in the dropdown too**, shown as
  `None [neuron id 87]`. Steering toward a neuron the labeling step
  could not name is well-defined arithmetically but meaningless in
  practice.
- **Compare refuses mismatched datasets** with an explicit error
  rather than silently steering two different people (§3). If you hit
  it, pick a past run whose `dataset_loading` matches.
- **Already-seen items never appear** in either recommendation list —
  both are masked against the user's history, so the two lists are
  comparable and neither is padded with things the user already
  interacted with.

---

## ➡️ 9. Where to go next

- **Where the concepts and their labels come from:**
  [`../neuron_labeling/README.md`](../neuron_labeling/README.md)
- **Checking a concept before steering with it:**
  [`../inspection/README.md`](../inspection/README.md)
- **Judging the label set as a whole:**
  [`../labeling_evaluation/README.md`](../labeling_evaluation/README.md)
- **The models being steered:**
  [`../training_cfm/README.md`](../training_cfm/README.md),
  [`../training_sae/README.md`](../training_sae/README.md)
- **The plugin contract** (discovery, `io_spec`, `run()` params,
  `BaseComparePlugin`): [`../README.md`](../README.md)
