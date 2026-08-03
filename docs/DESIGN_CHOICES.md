# SARSSA | Design Choices

## Motivation

Research on sparse autoencoders (SAEs) in recommender systems is a
multi-step affair: prepare a dataset, train a recommender, train an
SAE on top of it, give its neurons human-readable labels, and then
evaluate, inspect, and steer the result. Every step has alternative
methods, and in practice each experiment tends to get its own ad-hoc
scripts wired together for one specific combination. Such experiments
are hard to reproduce, hard to extend, and nearly impossible to
compare against each other.

SARSSA is my answer to that: a platform that turns the whole recipe -
introduced in the paper *From Knots to Knobs: Towards Steerable
Collaborative Filtering Using Sparse Autoencoders* - into a pipeline
of interchangeable steps. It generalises the entire workflow, so that the research group
can swap methods in and out, re-run what changed, and build further
work (my master's thesis included) on top of it.

```
 Build phase - runs once per pipeline, left to right:

 ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐
 │ Dataset Loading │  │ Training CFM │  │ Training SAE │  │ Neuron Labeling │
 │ ◉ movieLens     │  │ ◉ ELSA       │  │ ◉ TopK SAE   │  │ ◉ TF-IDF        │
 │ ○ steam-games   │  │              │  │ ○ BatchTopK… │  │ ○ tag corr.     │
 └─────────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘

 Analysis phase - runs repeatedly on top of a finished pipeline:

 ┌─────────────────────┐  ┌──────────────────┐  ┌────────────────┐
 │ Labeling Evaluation │  │ Inspection       │  │ Steering       │
 │ ◉ embedding map     │  │ ◉ SAE inspection │  │ ◉ SAE steering │
 │ ○ dendrogram …      │  │                  │  │                │
 └─────────────────────┘  └──────────────────┘  └────────────────┘
```

*The app's home screen in a sketch: each card is one **step** of the
pipeline, and the choices inside it are the available **plugins** -
interchangeable implementations of that step. Running the top row
left to right builds a **pipeline**; the bottom-row cards are
analyses run on top of it, as many times as needed.*

## The decisions that shaped the system

**Plugins declare, the engine executes.** Every pipeline step is a
plugin: a small module that *declares* what it needs from previous
steps and what it produces, plus the computation itself. Everything
around it - finding the inputs, saving the outputs, recording what
happened - is handled by one shared engine. Adding a new method
therefore means adding one new folder, with no changes anywhere else.
And because the rest of the pipeline depends only on a step's
*category* (say, "neuron labeling"), any two plugins of the same
category are interchangeable.

**Everything lives in MLflow.** MLflow is an open-source journal for
machine-learning experiments. I decided early that it would be the
single source of truth: a pipeline is one parent record, every step
is a nested record inside it, and *all* files, parameters, and
metrics pass through it - nothing is stored on the side. The files a
step records are at the same time the interface its successors read.
The payoff is that reproducibility is not a discipline anyone has to
maintain; it is how the system works. Every result can be traced back
to exactly how it was produced, MLflow's own web interface comes for
free as a second way to browse history, and whole runs can be moved
between machines - train on a GPU cluster, serve the results from a
modest university server.

**Build once, evaluate many times.** The stages fall into two
kinds. Building stages (dataset, recommender, SAE, labels) run once
per pipeline. Evaluation stages (label evaluation, inspection,
steering) run repeatedly on top of a finished pipeline, each time
with different questions. The specification sketched "Inspect &
Steer" as a single step; I split the evaluation side into three
independent categories instead. This mirrors how the research is
actually done: you build a pipeline once, then interrogate it many
times.

**New experiments start from old ones.** Any finished run can serve
as the base of a new one: its completed steps are loaded as-is, and
only what changes is re-run. The new work is recorded as a new run;
the base stays untouched. Expensive training stages are never
recomputed just to try a different labeling method, one-variable
experiments over an identical base become the natural way of working,
and shared baseline runs give everyone the same starting point.

**The interface does not know the plugins.** The web frontend has no
knowledge of any concrete plugin. It renders itself from a registry
the backend builds by inspecting the plugins - their parameters,
input widgets, and result views. A new plugin therefore shows up in
the interface, configuration form and all, without a single frontend
change - and the interface can never drift out of sync with what the
backend actually offers.

**Comparison is built in.** Evaluation plugins come in two flavours:
one runs against the current pipeline, the other runs against the
current pipeline *and* a chosen past run, side by side. "Did my
change actually help?" is answered with one click, not by arranging
two browser windows next to each other.

**Many users, one deployment.** For shared instances I chose
experiment spaces over user accounts: there is one shared base
experiment, anyone can create further spaces straight from the
header, and the shared runs are always offered as building material.
Several people can work on one deployment without overwriting each
other's lines of work - an organisational model that fits a trusting
research group.

**Long computations are first-class.** A training step can take
hours, which is unusual territory for a web application. Compute
tasks are queued and run one at a time - so two trainings never fight
over the GPU or the random number generators, which would silently
break reproducibility. Progress is reported live with an estimated
time remaining, a running step can be cancelled (gracefully or
immediately), and runs interrupted by a crash are detected and marked
on the next start.

**Validated defaults.** The default configuration matches a
published, replicated setup, so every experiment starts from a
verified baseline. Extensions (additional losses, model variants) are
opt-in: when a result differs, it is because of what you changed, not
because of a different default.

**One API, two clients.** The web interface and a small command-line
client drive the same HTTP API, so scripted experiments follow
exactly the same path as clicks in the browser.

## Beyond the reference paper

**Generalisation.** The two shipped datasets were chosen for domain
spread - MovieLens as the classic research benchmark, Steam games as
a niche domain to demonstrate the platform is not tied to movies -
and dataset loading is a documented extension point with a
bring-your-own-dataset tutorial. Beyond the paper's setup, SARSSA
ships three SAE variants (including BatchTopK, which has no paper
analogue), two neuron-labeling methods, seven plugins for evaluating
the geometry of the produced labels, and the side-by-side compare
mode across the whole evaluation phase.

**A finding of our own.** The paper's TF-IDF labeling method turns
out to systematically prefer rare tags - labels that look impressive
but whose items the neuron does not actually respond to. This
surfaced precisely because SARSSA shows a label for every neuron in
its interface, which the paper never does. The platform's answer is
shipped in code: a second labeling method based on correlation, and a
per-neuron confidence score displayed directly in the interface as
colour tint and sort order - so a weak label is visible as weak
before anyone builds conclusions on it.

## Evaluation and where this leads

**Against the specification's criteria.** The specification also
defined how the result should be judged - correctness and stability
of pipeline execution, reproducibility of results, the ability to
replicate published research, and usability of the interface. Where
the delivered system stands: *correctness and stability* are covered
by an automated test suite, safe cancellation of running steps, and
detection of runs interrupted by a crash. *Reproducibility* is built
in - every run records everything needed to repeat
it (via "Everything lives in MLflow"). *Replication* was verified
directly - the default setup reproduces the reference paper's
published results (Recall@20 of 0.393 vs. 0.396 reported).
*Usability* is exercised daily by the public demo instance, and the
interface and documentation were shaped by several rounds of
supervisor feedback.

**What it enables next.** The platform is the intended base of my
master's thesis, a natural harness for comparative studies of
labeling methods, and the confidence metric together with the
rare-tag finding are material for a future paper.
