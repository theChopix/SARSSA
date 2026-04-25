/**
 * Root application component.
 *
 * This is the top-level React component rendered by `main.tsx`.
 * It orchestrates the full page layout matching the UI mockup:
 *
 *   ┌──────────────────────────────────────────────────┐
 *   │  SARSSAe               mlflow … Results (link)   │  ← header
 *   ├──────────────────────────────────────────────────┤
 *   │  "Run new pipeline experiment"                   │  ← title
 *   │                                                  │
 *   │  ┌────┐ ┌────┐ ┌────┐ ┌────┐                    │  ← row 1
 *   │  │Card│ │Card│ │Card│ │Card│   (one_time)        │    (4 cols)
 *   │  └────┘ └────┘ └────┘ └────┘                    │
 *   │                                                  │
 *   │  ┌────┐ ┌────┐ ┌────┐                            │  ← row 2
 *   │  │Card│ │Card│ │Card│          (multi_run)       │    (3 cols)
 *   │  └────┘ └────┘ └────┘                            │
 *   │                                                  │
 *   │  [ Run full pipeline ]                           │  ← bottom bar
 *   └──────────────────────────────────────────────────┘
 *
 * ┌──────────────────────────────────────────────────────┐
 * │  React hooks refresher                                │
 * │                                                       │
 * │  `useEffect(fn, [])` — runs `fn` once when the       │
 * │  component first appears ("mounts"). The empty `[]`   │
 * │  means "no dependencies — don't re-run".              │
 * │  We use it to fetch the registry on page load.        │
 * │                                                       │
 * │  `useMemo(fn, [deps])` — caches the return value     │
 * │  of `fn` and only recalculates when `deps` change.   │
 * │  We use it to split categories into rows.             │
 * └──────────────────────────────────────────────────────┘
 */

import { useEffect, useMemo } from "react";
import { Loader2, X, Ban } from "lucide-react";
import { Toaster } from "sonner";

import PipelineCard from "./components/PipelineCard";
import LaunchModal from "./components/LaunchModal";
import { usePipelineStore, mlflowExperimentUrl } from "./store/pipelineStore";
import type { StepDefinition } from "./types/pipeline";

/**
 * Coerce a string value from an HTML input to the correct JS type
 * based on the Python type string from the plugin registry.
 *
 * HTML inputs always produce strings, but the backend expects
 * native types (int → number, float → number, bool → boolean).
 */
function coerceParamValue(value: string, pythonType: string): unknown {
  switch (pythonType) {
    case "int":
      return parseInt(value, 10);
    case "float":
      return parseFloat(value);
    case "bool":
      return value.toLowerCase() === "true";
    default:
      return value;
  }
}

/**
 * App — the top-level React component.
 *
 * On mount, it fetches the plugin registry from the backend.
 * Then it splits categories into two rows (one_time vs multi_run)
 * and renders a grid of PipelineCard components.
 */
function App() {
  // ── Read state from store ───────────────────────────
  const registry = usePipelineStore((s) => s.registry);
  const cards = usePipelineStore((s) => s.cards);
  const pipelineRunning = usePipelineStore((s) => s.pipelineRunning);
  const loadRegistry = usePipelineStore((s) => s.loadRegistry);
  const resetCards = usePipelineStore((s) => s.resetCards);
  const setPendingSteps = usePipelineStore((s) => s.setPendingSteps);
  const currentRunId = usePipelineStore((s) => s.currentRunId);
  const runSingleStep = usePipelineStore((s) => s.runSingleStep);
  const errorMessage = usePipelineStore((s) => s.errorMessage);
  const clearError = usePipelineStore((s) => s.clearError);
  const currentStepIndex = usePipelineStore((s) => s.currentStepIndex);
  const totalSteps = usePipelineStore((s) => s.totalSteps);
  const mlflowInfo = usePipelineStore((s) => s.mlflowInfo);
  const loadMlflowInfo = usePipelineStore((s) => s.loadMlflowInfo);
  const cancellationPending = usePipelineStore((s) => s.cancellationPending);
  const cancelPipeline = usePipelineStore((s) => s.cancelPipeline);

  // ── Load registry on mount ──────────────────────────
  // This runs ONCE when the page loads. It calls the backend
  // to fetch all available plugins and categories.
  useEffect(() => {
    loadRegistry();
    loadMlflowInfo();
  }, [loadRegistry, loadMlflowInfo]);

  // ── Split categories into two rows ──────────────────
  // `useMemo` caches this computation so it only re-runs
  // when the registry changes, not on every render.
  const { oneTimeKeys, multiRunKeys } = useMemo(() => {
    if (!registry) return { oneTimeKeys: [], multiRunKeys: [] };

    const oneTime: string[] = [];
    const multiRun: string[] = [];

    // Sort categories by their `order` field, then split by type.
    const sorted = Object.entries(registry).sort(
      ([, a], [, b]) => a.category_info.order - b.category_info.order
    );

    for (const [key, entry] of sorted) {
      if (entry.category_info.type === "one_time") {
        oneTime.push(key);
      } else {
        multiRun.push(key);
      }
    }

    return { oneTimeKeys: oneTime, multiRunKeys: multiRun };
  }, [registry]);

  // ── Handler: "Run up to this step" ──────────────────
  // Collects all one_time steps from the first card up to
  // (and including) the clicked card, then starts the pipeline.
  const handleRunUpTo = (targetCategoryKey: string) => {
    if (!registry) return;

    const steps: StepDefinition[] = [];

    for (const key of oneTimeKeys) {
      const card = cards[key];
      if (!card?.selectedPlugin) continue;

      // Build params: start from defaults, override with user input.
      const entry = registry[key];
      const impl = entry.implementations.find(
        (i) => i.plugin_name === card.selectedPlugin
      );
      const params: Record<string, unknown> = {};
      if (impl) {
        for (const p of impl.params) {
          const userVal = card.params[p.name];
          if (userVal !== undefined && userVal !== "") {
            params[p.name] = coerceParamValue(userVal, p.type);
          } else if (p.default != null) {
            params[p.name] = p.default;
          }
        }
      }

      steps.push({ plugin: card.selectedPlugin, params });

      // Stop after the target category.
      if (key === targetCategoryKey) break;
    }

    if (steps.length > 0) {
      setPendingSteps(steps);
    }
  };

  // ── Handler: "Execute step" (multi_run) ─────────────
  // Runs a single plugin step on the current pipeline run.
  const handleExecuteStep = (categoryKey: string) => {
    if (!currentRunId || !registry) return;

    const card = cards[categoryKey];
    if (!card?.selectedPlugin) return;

    const entry = registry[categoryKey];
    const impl = entry.implementations.find(
      (i) => i.plugin_name === card.selectedPlugin
    );
    const params: Record<string, unknown> = {};
    if (impl) {
      for (const p of impl.params) {
        const userVal = card.params[p.name];
        if (userVal !== undefined && userVal !== "") {
          params[p.name] = coerceParamValue(userVal, p.type);
        } else if (p.default != null) {
          params[p.name] = p.default;
        }
      }
    }

    runSingleStep(currentRunId, { plugin: card.selectedPlugin, params });
  };


  // ── Loading state ───────────────────────────────────
  if (!registry) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
      </div>
    );
  }

  // ── Render ──────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* ── Toast notifications ──────────────── */}
      <Toaster richColors position="bottom-right" />
      {/* ── Launch confirmation modal ────────── */}
      <LaunchModal />
      {/* ── Header ─────────────────────────────────── */}
      <header className="border-b border-gray-200 bg-white px-8 py-4 flex items-center justify-between">
        <h1 className="text-lg font-bold text-gray-900">SARSSAe</h1>
        <a
          href={mlflowInfo ? mlflowExperimentUrl(mlflowInfo) : "#"}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm text-blue-500 hover:text-blue-700"
        >
          <img
            src="/mlflow-logo.png"
            alt="mlflow"
            className="h-5"
          />
          Pipeline Experiments Results
        </a>
      </header>

      {/* ── Error / cancellation banner ───────────────── */}
      {errorMessage && (
        <div
          className={`mx-8 mt-4 flex items-center justify-between gap-3 rounded-lg
                      border px-4 py-3 ${
                        errorMessage.toLowerCase().includes("cancelled")
                          ? "border-amber-200 bg-amber-50"
                          : "border-red-200 bg-red-50"
                      }`}
        >
          <p
            className={`text-sm ${
              errorMessage.toLowerCase().includes("cancelled")
                ? "text-amber-700"
                : "text-red-700"
            }`}
          >
            {errorMessage}
          </p>
          <button
            onClick={clearError}
            className={`shrink-0 transition-colors cursor-pointer ${
              errorMessage.toLowerCase().includes("cancelled")
                ? "text-amber-400 hover:text-amber-600"
                : "text-red-400 hover:text-red-600"
            }`}
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* ── Page title ─────────────────────────────── */}
      <div className="px-8 pt-8 pb-4">
        <h2 className="text-xl font-bold text-gray-900 text-center">
          Run new pipeline experiment
        </h2>
      </div>

      {/* ── Row 1: one_time cards ──────────────────── */}
      <section className="px-8 pb-4">
        <div className="grid grid-cols-4 gap-4">
          {oneTimeKeys.map((key) => (
            <PipelineCard
              key={key}
              categoryKey={key}
              onRunUpTo={handleRunUpTo}
            />
          ))}
        </div>
      </section>

      {/* ── Row 2: multi_run cards ─────────────────── */}
      <section className="px-8 pb-8">
        <div className="grid grid-cols-3 gap-4">
          {multiRunKeys.map((key) => (
            <PipelineCard
              key={key}
              categoryKey={key}
              onExecuteStep={handleExecuteStep}
            />
          ))}
        </div>
      </section>

      {/* ── Spacer pushes bottom bar down ──────────── */}
      <div className="flex-1" />

      {/* ── Bottom bar: Reset / Running + Cancel ────── */}
      <div className="px-8 pb-8">
        {pipelineRunning ? (
          <div className="flex gap-3">
            <div className="flex-1 py-3 rounded-lg text-white font-medium text-sm
                            bg-blue-500 opacity-70 text-center">
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                {cancellationPending
                  ? "Cancelling... waiting for current step to finish"
                  : `Running pipeline... Step ${currentStepIndex + 1} / ${totalSteps}`}
              </span>
            </div>
            <button
              onClick={cancelPipeline}
              disabled={cancellationPending}
              className="px-5 py-3 rounded-lg text-sm font-medium
                         bg-red-500 text-white
                         hover:bg-red-600 disabled:opacity-50
                         disabled:cursor-not-allowed
                         transition-colors cursor-pointer
                         flex items-center gap-2"
            >
              <Ban className="h-4 w-4" />
              {cancellationPending ? "Cancelling..." : "Cancel"}
            </button>
          </div>
        ) : (
          <button
            onClick={resetCards}
            className="w-full py-3 rounded-lg text-sm font-medium
                       bg-blue-500 text-white
                       hover:bg-blue-600 transition-colors cursor-pointer"
          >
            Reset pipeline settings
          </button>
        )}
      </div>
    </div>
  );
}

export default App;
