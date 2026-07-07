/**
 * GuidePage — in-app user guide.
 *
 * Standalone route at `/guide`, reached via the "Guide" link in the
 * Header. Self-contained: no store reads, no async loads — the page
 * is pure static JSX so it works in any tab (including the new tab
 * the results page opens into) without needing the registry loaded.
 *
 * Layout:
 *
 *   ┌──────────┬──────────────────────────────────────┐
 *   │ Contents │  Guide                                │
 *   │ (TOC)    │                                       │
 *   │          │  ── Section ─────────────────────     │
 *   │ (sticky) │  (section content)                    │
 *   │          │                                       │
 *   │          │  ── Section ─────────────────────     │
 *   │          │  ...                                  │
 *   └──────────┴──────────────────────────────────────┘
 *
 * The aside is `position: sticky` so the TOC stays in view while
 * the main column scrolls. The main column is capped at `max-w-3xl`
 * so prose stays comfortably narrow on wide screens.
 *
 * Both the TOC and the rendered `<section>` blocks are driven by the
 * same `SECTIONS` array, so the list and the anchors cannot drift
 * out of sync. Each section uses `scroll-mt-8` so that anchor jumps
 * land with a little breathing room above the heading instead of
 * flush against the viewport top.
 */

import type { ReactNode } from "react";

interface GuideSection {
  id: string;
  title: string;
  body: ReactNode;
}

const SECTIONS: GuideSection[] = [
  // ── Overview ───────────────────────────────────────────
  {
    id: "overview",
    title: "Overview",
    body: (
      <>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          <strong>SARSSAe</strong> is a research platform for designing,
          running, and inspecting recommender-system experiments built
          around <strong>Sparse Autoencoders (SAEs)</strong> — an
          interpretability technique applied to collaborative filtering.
        </p>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          A trained recommender produces a <strong>dense user embedding</strong>{" "}
          — a single vector summarising what a user likes — which is
          useful for predictions but largely opaque. An SAE re-expresses
          that embedding in a higher-dimensional space where only a
          handful of neurons are active at a time, and each active
          neuron tends to correspond to a single, <strong>granular,
          human-interpretable concept</strong> (a <em>love-story</em>{" "}
          concept, a <em>sci-fi</em> concept, a <em>David Lynch</em>{" "}
          concept). Those concepts become <strong>"knobs"</strong>: turn
          one up or down and the recommendations <strong>steer</strong>{" "}
          toward it while preserving most of the base model's accuracy.
        </p>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          This web UI is the experiment cockpit: compose a pipeline
          from cards, run it, browse the visual results, and steer the
          trained model. Every run is also recorded in{" "}
          <strong>MLflow</strong> so experiments stay reproducible and
          reusable across the project.
        </p>
        <p className="text-sm text-gray-500 leading-relaxed">
          Background: see the{" "}
          <a
            href="https://github.com/theChopix/SARSSA"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-500 hover:text-blue-700"
          >
            GitHub repository
          </a>{" "}
          and its{" "}
          <code className="px-1 py-0.5 rounded bg-gray-100 text-gray-800 text-xs">
            README.md
          </code>
          , or the underlying paper{" "}
          <a
            href="https://arxiv.org/abs/2601.11182"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-500 hover:text-blue-700"
          >
            <em>From Knots to Knobs</em> (arXiv:2601.11182)
          </a>
          .
        </p>
      </>
    ),
  },

  // ── Cards layout ───────────────────────────────────────
  {
    id: "cards-layout",
    title: "Cards layout",
    body: (
      <>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          The home page is split into <strong>two rows of cards</strong>
          , each card representing one stage of a pipeline.
        </p>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          <strong>Top row — sequential steps.</strong> These cards form
          the ordered backbone of an experiment (load a dataset, train
          a collaborative-filtering model, train the SAE, label the
          SAE's neurons). They must run in order, left to right. Each
          card has a <em>Run up to this step</em> button that executes
          itself <strong>and</strong> every card before it as one
          MLflow parent run.
        </p>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          <strong>Bottom row — multi-run steps.</strong> Once the top
          row has finished (or has been loaded from a previous run),
          each bottom card can be executed repeatedly against that
          pipeline state via <em>Execute step</em>. Use these to
          inspect a trained pipeline, evaluate its neuron labels, or
          steer its recommendations.
        </p>
        <pre
          className="bg-gray-50 border border-gray-200 rounded p-3
                     text-xs text-gray-700 my-3 overflow-x-auto font-mono"
        >
{`┌─ Top row ─────────────────────────────────────┐
│  [card]  [card]  [card]  [card]               │   sequential
└───────────────────────────────────────────────┘
┌─ Bottom row ──────────────────────────────────┐
│  [card]  [card]  [card]                       │   multi-run
└───────────────────────────────────────────────┘`}
        </pre>
        <p className="text-sm text-gray-700 leading-relaxed">
          <strong>
            Why <em>Run up to this step</em> is on every card, not just
            the last one.
          </strong>{" "}
          You don't have to commit to running the whole pipeline up
          front. You can stop after any step, look at what it produced
          — both inside the UI and in MLflow — and decide whether to
          continue, change a parameter, or branch off. This lets you
          develop and debug a pipeline incrementally instead of waiting
          for the full chain every time.
        </p>
      </>
    ),
  },

  // ── Plugins ────────────────────────────────────────────
  {
    id: "plugins",
    title: "Plugins",
    body: (
      <>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          Each card lists one or more <strong>plugins</strong> as radio
          buttons. A plugin is one backend module — one Python class —
          implementing that step in a particular way. A training card,
          for example, exposes every trainer the backend currently
          ships for that stage; pick one, click <em>Configure</em> to
          fill in its parameters, then run.
        </p>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          Parameter widgets are not hand-coded in the UI. The backend
          describes each parameter (name, type, default, description,
          widget kind) and the frontend renders the appropriate input —
          a text box, a slider, a dropdown of dynamic choices, or a
          "past runs" picker. Hover the ⓘ icon next to a parameter
          name to read its description. The same ⓘ appears next to
          each card title with a one-line summary of what that
          category does, and next to each plugin's name with a short
          description of what that plugin does.
        </p>
        <p className="text-sm text-gray-700 leading-relaxed">
          Adding a new plugin on the backend — your own dataset loader,
          your own SAE variant, your own steering method — makes it
          appear in the matching card automatically; there are no
          frontend changes to make. The plugin contract lives in{" "}
          <code className="px-1 py-0.5 rounded bg-gray-100 text-gray-800 text-xs">
            src/plugins/README.md
          </code>{" "}
          in the repository.
        </p>
      </>
    ),
  },

  // ── Set up new vs Load from previous run ───────────────
  {
    id: "setup-vs-load",
    title: "Set up new vs Load from previous run",
    body: (
      <>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          Top-row cards offer two modes, selected by the toggle at the
          top of the card:
        </p>
        <ul className="list-disc pl-5 space-y-1 mb-3 text-sm text-gray-700 leading-relaxed">
          <li>
            <strong>Set up new</strong> — pick a plugin, fill its
            parameters, run.
          </li>
          <li>
            <strong>Load from previous run</strong> — reuse the outputs
            of a past pipeline run instead of recomputing them. This is
            the platform's most useful feature for slow stages: train a
            CFM and an SAE once, then build many downstream experiments
            on top of them.
          </li>
        </ul>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          <strong>How the load dropdown works.</strong> The dropdown
          lists <strong>every</strong> past pipeline run, unfiltered.
          Filtering happens at <em>selection time</em>: when you pick a
          run, only the steps{" "}
          <strong>at or before the card whose dropdown you opened</strong>{" "}
          get loaded into the corresponding cards (marked <em>done</em>{" "}
          + in <em>Load</em> mode). Any downstream top-row cards are
          reset so you can configure and run them fresh.
        </p>
        <p className="text-sm text-gray-700 leading-relaxed">
          Think of it as picking a <strong>prefix</strong> of a past
          pipeline to keep, then continuing from there with whatever
          combination of new and loaded steps you want.
        </p>
      </>
    ),
  },

  // ── Single vs Compare plugins ──────────────────────────
  {
    id: "single-vs-compare",
    title: "Single vs Compare plugins",
    body: (
      <>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          Multi-run cards (the bottom row) sometimes expose two{" "}
          <strong>kinds</strong> of plugins, separated by a small tab
          strip at the top of the card:
        </p>
        <ul className="list-disc pl-5 space-y-1 mb-3 text-sm text-gray-700 leading-relaxed">
          <li>
            <strong>single</strong> — operates on the current pipeline
            run. Inspect a neuron in your trained SAE; produce a set of
            recommendations for a user.
          </li>
          <li>
            <strong>compare</strong> — takes a second past pipeline run
            as a parameter and produces a side-by-side comparison. The
            second run is chosen via a "past runs" dropdown that is
            filtered to runs containing the steps the comparison needs.
          </li>
        </ul>
        <p className="text-sm text-gray-700 leading-relaxed">
          Some categories ship only <em>single</em> plugins (or only{" "}
          <em>compare</em> plugins). When that's the case the tab strip
          isn't shown and the card behaves as usual.
        </p>
      </>
    ),
  },

  // ── Viewing results ────────────────────────────────────
  {
    id: "view-results",
    title: "Viewing results",
    body: (
      <>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          Where you see a step's results depends on whether the card is
          sequential or multi-run.
        </p>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          <strong>Sequential cards</strong> show, once completed, the
          locked pipeline-run name plus deep links into MLflow ("See
          pipeline run", "See step run"). All of the step's outputs —
          interaction matrices, trained models, neuron labels, … — live
          in MLflow as artifacts of that step's nested run; click
          through to browse them.
        </p>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          <strong>Multi-run cards</strong> expose the same MLflow links{" "}
          <em>and</em>, when the plugin declares a visual display, a
          green <strong>View Results</strong> button that opens a
          dedicated results page <strong>in a new browser tab</strong>.
          Two display kinds are supported:
        </p>
        <ul className="list-disc pl-5 space-y-1 mb-3 text-sm text-gray-700 leading-relaxed">
          <li>
            <strong>Item rows</strong> — labelled rows of enriched item
            cards (interaction history, original recommendations,
            steered recommendations, …). Used by inspection and
            steering plugins.
          </li>
          <li>
            <strong>Artifact viewer</strong> — static images or HTML
            reports. Used by plugins that produce a chart or a
            generated report instead of a list of items.
          </li>
        </ul>
        <p className="text-sm text-gray-700 leading-relaxed">
          If a plugin doesn't declare a display spec, the{" "}
          <em>View Results</em> button simply doesn't appear; the step
          still ran fine — it just has nothing visual to show beyond
          MLflow.
        </p>
      </>
    ),
  },

  // ── MLflow integration ─────────────────────────────────
  {
    id: "mlflow",
    title: "MLflow integration",
    body: (
      <>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          <a
            href="https://mlflow.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-500 hover:text-blue-700"
          >
            MLflow
          </a>{" "}
          is the experiment-tracking system SARSSAe writes everything
          to. Two terms matter:
        </p>
        <ul className="list-disc pl-5 space-y-1 mb-3 text-sm text-gray-700 leading-relaxed">
          <li>
            A <strong>run</strong> is one tracked execution. It stores{" "}
            <strong>params</strong> (inputs), <strong>metrics</strong>{" "}
            (numbers), and <strong>artifacts</strong> (output files).
          </li>
          <li>
            Runs can be <strong>nested</strong> — a parent run with
            child runs underneath.
          </li>
        </ul>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          Every pipeline launch in SARSSAe maps to{" "}
          <strong>one parent MLflow run</strong> with{" "}
          <strong>one child run per step</strong>. The parent stores a
          small{" "}
          <code className="px-1 py-0.5 rounded bg-gray-100 text-gray-800 text-xs">
            context.json
          </code>{" "}
          mapping each finished step → its child run id, which is
          exactly how <em>Load from previous run</em> knows what to
          reuse.
        </p>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          You reach MLflow from the UI in two places:
        </p>
        <ul className="list-disc pl-5 space-y-1 text-sm text-gray-700 leading-relaxed">
          <li>
            <strong>Header (top-right)</strong> — the{" "}
            <em>Pipeline Experiments Results</em> link opens the MLflow
            experiment page in a new tab. Use it to browse the full
            history, filter by tags, or grab artifacts you don't see in
            the SARSSAe UI.
          </li>
          <li>
            <strong>Per-card</strong> — completed cards expose{" "}
            <em>See pipeline run</em> and <em>See step run</em> links
            straight to those runs in MLflow.
          </li>
        </ul>
      </>
    ),
  },

  // ── Launching a run (Launch modal) ─────────────────────
  {
    id: "launch-modal",
    title: "Launching a run",
    body: (
      <>
        <p className="text-sm text-gray-700 leading-relaxed mb-3">
          Clicking <em>Run up to this step</em> doesn't kick off the
          run immediately. It first opens the{" "}
          <strong>Launch Pipeline Run</strong> modal, where you can
          attach:
        </p>
        <ul className="list-disc pl-5 space-y-1 mb-3 text-sm text-gray-700 leading-relaxed">
          <li>
            <strong>Key-value tags</strong> — short labels for the run.
            Pre-filled with the plugin names you selected (e.g.{" "}
            <code className="px-1 py-0.5 rounded bg-gray-100 text-gray-800 text-xs">
              plugin.training_cfm = elsa_trainer
            </code>
            ); add, edit, or remove them freely.
          </li>
          <li>
            <strong>A description</strong> — free-text notes about what
            makes this run interesting (a hypothesis, a configuration
            change you're testing).
          </li>
        </ul>
        <p className="text-sm text-gray-700 leading-relaxed">
          Both are stored as{" "}
          <code className="px-1 py-0.5 rounded bg-gray-100 text-gray-800 text-xs">
            sarssa.*
          </code>{" "}
          tags on the MLflow parent run. They don't change what the
          pipeline does, but they make past runs much easier to find
          later — both in the <em>Load from previous run</em> dropdown
          and in MLflow's own search and filter UI. Press{" "}
          <em>Launch</em> to start the run,{" "}
          <em>Cancel</em> or{" "}
          <kbd className="px-1.5 py-0.5 rounded border border-gray-300 bg-gray-50 text-xs font-mono text-gray-700">
            Esc
          </kbd>{" "}
          to dismiss without launching.
        </p>
      </>
    ),
  },

  // ── Tips & troubleshooting ─────────────────────────────
  {
    id: "tips",
    title: "Tips & troubleshooting",
    body: (
      <>
        <ul className="list-disc pl-5 space-y-2 text-sm text-gray-700 leading-relaxed">
          <li>
            <strong>Progress.</strong> While a pipeline is running, the
            bottom action bar shows <em>Step i / N</em>. Individual
            plugin progress messages surface as toasts at the top of
            the screen — they're polled every ~2 seconds, so there's a
            small delay between the backend writing one and it
            appearing.
          </li>
          <li>
            <strong>Running tasks menu.</strong> When a pipeline is
            running or waiting its turn, a <em>Running (N)</em> pill
            appears in the header (top-left). Open it to see every task
            with its current step — queued ones carry a clock icon and a{" "}
            <em>queued</em> badge; click one to load it into the cards
            and follow its progress — handy when several tasks are
            lined up, or to jump back into a run from a fresh tab.
          </li>
          <li>
            <strong>Launching several pipelines.</strong> Compute tasks
            run <strong>one at a time</strong>: launching a pipeline (or
            an evaluation step) while another is still executing is
            perfectly fine — the new task waits in a queue and starts
            automatically as soon as the previous one finishes. This
            keeps runs reproducible and prevents two trainings from
            fighting over GPU memory. Each tab tracks one active run, so
            to <em>launch</em> a second pipeline open SARSSAe in{" "}
            <strong>another browser tab</strong> and start it there; the{" "}
            <em>Running</em> menu then lists both — the executing one and
            the queued one.
          </li>
          <li>
            <strong>Refreshing is safe.</strong> If you reload the page
            mid-run, the UI restores the running layout and reconnects to
            it — the run keeps going on the backend regardless. It's
            per-tab and session-scoped, so closing the tab clears it.
          </li>
          <li>
            <strong>Cancellation.</strong> A running pipeline shows two
            buttons. <em>Cancel after this step</em> lets the step that's
            currently running finish, then stops the pipeline — no later
            steps run. <em>Cancel now</em> also tries to stop the running
            step right away. Long training steps react almost
            immediately, because they keep checking for a stop request as
            they go. A step that can't be stopped partway simply finishes
            first — so for those, <em>Cancel now</em> ends up behaving the
            same as <em>Cancel after this step</em>. Tip: if you already
            clicked <em>Cancel after this step</em> and changed your mind,
            you can still click <em>Cancel now</em>. A run that's still{" "}
            <em>queued</em> is simply removed from the queue right away —
            both buttons do the same thing there.
          </li>
          <li>
            <strong>Reset.</strong> <em>Reset pipeline settings</em>{" "}
            clears every card's plugin selection and parameters back to
            defaults. It does not delete any past run from MLflow —
            just the local UI state.
          </li>
          <li>
            <strong>"Nothing loads."</strong> The frontend expects the
            backend on port 8000 and itself on port 5173 (the backend's
            CORS allow-list). If the page renders but every action
            errors out, that's usually the cause — confirm both
            services are running on those exact ports.
          </li>
        </ul>
      </>
    ),
  },
];

export function GuidePage() {
  return (
    <div className="flex-1 flex">
      {/* ── Sticky table of contents ───────────────────── */}
      <aside
        className="w-64 shrink-0 border-r border-gray-200
                   px-6 py-8 self-start sticky top-0
                   max-h-screen overflow-y-auto"
      >
        <p
          className="text-xs font-semibold text-gray-500
                     uppercase tracking-wide mb-3"
        >
          Contents
        </p>
        <nav aria-label="Table of contents">
          <ul className="space-y-1.5 text-sm">
            {SECTIONS.map((section) => (
              <li key={section.id}>
                <a
                  href={`#${section.id}`}
                  className="text-blue-500 hover:text-blue-700"
                >
                  {section.title}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </aside>

      {/* ── Main content ──────────────────────────────── */}
      <main className="flex-1 px-8 py-8 max-w-3xl">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Guide</h1>
        <p className="text-sm text-gray-500 mb-8">
          How to use the SARSSAe web UI.
        </p>

        {SECTIONS.map((section) => (
          <section
            key={section.id}
            id={section.id}
            className="mb-12 scroll-mt-8"
          >
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              {section.title}
            </h2>
            {section.body}
          </section>
        ))}
      </main>
    </div>
  );
}
