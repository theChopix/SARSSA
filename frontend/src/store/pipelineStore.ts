/**
 * Zustand store for pipeline state management.
 *
 * ┌──────────────────────────────────────────────────────────┐
 * │  What is Zustand?                                        │
 * │                                                          │
 * │  Zustand is a small state management library for React.  │
 * │  Think of it as a global variable that React components  │
 * │  can read from and write to — and when the value changes │
 * │  the components automatically re-render.                 │
 * │                                                          │
 * │  In Python terms: imagine a shared dict that all parts   │
 * │  of your app can access, and changing it triggers a UI   │
 * │  refresh.                                                │
 * │                                                          │
 * │  Usage in a component:                                   │
 * │    const registry = usePipelineStore(s => s.registry);   │
 * │    // `registry` updates when the store changes          │
 * └──────────────────────────────────────────────────────────┘
 *
 * This store holds ALL pipeline-related state:
 * - Plugin registry (from backend)
 * - Per-category card state (selected plugin, params, status)
 * - Pipeline run state (run ID, context, running flag)
 * - Past runs list (for "Load from previous run")
 */

import { create } from "zustand";

import { fetchPluginRegistry } from "../api/plugins";
import {
  fetchPipelineRuns,
  fetchRunContext,
  startPipelineTask,
  getTaskStatus,
  executeStep,
} from "../api/pipelines";
import type { PluginRegistry } from "../types/plugin";
import type {
  PipelineContext,
  PipelineRun,
  StepDefinition,
} from "../types/pipeline";

// ── Per-category card state ─────────────────────────────

/**
 * Status of a single pipeline card.
 *
 * - "idle"     → nothing happening, ready to configure
 * - "running"  → plugin is currently executing (show spinner)
 * - "done"     → plugin finished successfully (show checkmark)
 * - "error"    → plugin execution failed
 */
export type CardStatus = "idle" | "running" | "done" | "error";

/**
 * State for a single category card in the UI.
 *
 * Each card tracks which plugin the user selected, what parameter
 * values they entered, and whether the plugin is currently running.
 */
export interface CardState {
  /** Dotted plugin name selected by the user (e.g. "training_cfm.elsa_trainer.elsa_trainer"). */
  selectedPlugin: string | null;
  /** Parameter values entered by the user, keyed by param name. */
  params: Record<string, string>;
  /** Whether the "Configure" section is expanded. */
  configOpen: boolean;
  /** Current execution status of this card. */
  status: CardStatus;
  /** MLflow run ID of the nested run (set after step completes). */
  stepRunId: string | null;
}

// ── Store shape ─────────────────────────────────────────

/**
 * The full Zustand store interface.
 *
 * Split into two sections:
 * 1. **State** — the data (plain values).
 * 2. **Actions** — functions that modify the state.
 *
 * Components read state via selectors:
 *   `const registry = usePipelineStore(s => s.registry);`
 *
 * Components call actions directly:
 *   `usePipelineStore.getState().loadRegistry();`
 *   — or via the hook:
 *   `const loadRegistry = usePipelineStore(s => s.loadRegistry);`
 */
interface PipelineStore {
  // ── State ───────────────────────────────────────────

  /** Plugin registry fetched from backend. `null` until loaded. */
  registry: PluginRegistry | null;

  /** Per-category card state, keyed by category name. */
  cards: Record<string, CardState>;

  /** Whether the full pipeline is currently running. */
  pipelineRunning: boolean;

  /** MLflow run ID of the current/last pipeline run. */
  currentRunId: string | null;

  /** Context from the current or loaded pipeline run. */
  context: PipelineContext | null;

  /** List of past pipeline runs (for "Load from previous run"). */
  pastRuns: PipelineRun[];

  /** The run ID selected in "Load from previous run" mode. */
  targetRunId: string | null;

  /** Error message from the last failed pipeline run. */
  errorMessage: string | null;

  /** 0-based index of the step currently executing. */
  currentStepIndex: number;

  /** Total number of steps in the current pipeline run. */
  totalSteps: number;

  // ── Actions ─────────────────────────────────────────

  /**
   * Fetch the plugin registry from the backend and initialise
   * a CardState for every category.
   */
  loadRegistry: () => Promise<void>;

  /** Fetch the list of past pipeline runs from the backend. */
  loadPastRuns: () => Promise<void>;

  /**
   * Load context from a past run and populate card states.
   * Used by "Load from previous run" buttons.
   */
  loadFromPreviousRun: (runId: string) => Promise<void>;

  /** Select a plugin implementation for a category card. */
  selectPlugin: (category: string, pluginName: string) => void;

  /** Update a parameter value for a category card. */
  setParam: (category: string, paramName: string, value: string) => void;

  /** Toggle the "Configure" section of a category card. */
  toggleConfig: (category: string) => void;

  /**
   * Run the full pipeline (all one_time steps) via background task + polling.
   * Updates card statuses as polling responses arrive.
   */
  runPipeline: (steps: StepDefinition[]) => Promise<void>;

  /** Clear the error message banner. */
  clearError: () => void;

  /**
   * Execute a single step on an existing pipeline run.
   * Used for multi_run plugins (Inspection, Steering, etc.).
   */
  runSingleStep: (
    runId: string,
    step: StepDefinition
  ) => Promise<void>;

  /** Reset all card statuses back to "idle". */
  resetCards: () => void;
}

// ── Default card state ──────────────────────────────────

/** Factory for a fresh card state. */
const defaultCard = (): CardState => ({
  selectedPlugin: null,
  params: {},
  configOpen: false,
  status: "idle",
  stepRunId: null,
});

// ── Store implementation ────────────────────────────────

/**
 * The main Zustand store.
 *
 * `create<PipelineStore>()` takes a function that receives `set` and `get`:
 * - `set(partial)` — merges a partial state update (like React's setState).
 * - `get()`        — reads the current state (useful inside actions).
 *
 * Every action is just a regular function that calls `set(...)`.
 */
export const usePipelineStore = create<PipelineStore>((set, get) => ({
  // ── Initial state ───────────────────────────────────
  registry: null,
  cards: {},
  pipelineRunning: false,
  currentRunId: null,
  context: null,
  pastRuns: [],
  targetRunId: null,
  errorMessage: null,
  currentStepIndex: 0,
  totalSteps: 0,

  // ── Actions ─────────────────────────────────────────

  loadRegistry: async () => {
    const registry = await fetchPluginRegistry();

    // Create a CardState for every category in the registry.
    const cards: Record<string, CardState> = {};
    for (const key of Object.keys(registry)) {
      cards[key] = defaultCard();
    }

    set({ registry, cards });
  },

  loadPastRuns: async () => {
    const pastRuns = await fetchPipelineRuns();
    set({ pastRuns });
  },

  loadFromPreviousRun: async (runId: string) => {
    const context = await fetchRunContext(runId);

    // Update cards: mark categories that were in the loaded run as "done".
    const cards = { ...get().cards };
    for (const [category, data] of Object.entries(context)) {
      if (cards[category]) {
        cards[category] = {
          ...cards[category],
          status: "done",
          stepRunId: data.run_id,
        };
      }
    }

    set({ context, currentRunId: runId, targetRunId: runId, cards });
  },

  selectPlugin: (category: string, pluginName: string) => {
    const cards = { ...get().cards };
    cards[category] = {
      ...cards[category],
      selectedPlugin: pluginName,
      params: {},
      configOpen: false,
    };
    set({ cards });
  },

  setParam: (category: string, paramName: string, value: string) => {
    const cards = { ...get().cards };
    cards[category] = {
      ...cards[category],
      params: { ...cards[category].params, [paramName]: value },
    };
    set({ cards });
  },

  toggleConfig: (category: string) => {
    const cards = { ...get().cards };
    cards[category] = {
      ...cards[category],
      configOpen: !cards[category].configOpen,
    };
    set({ cards });
  },

  runPipeline: async (steps: StepDefinition[]) => {
    // Reset cards that will be executed to idle.
    const cards = { ...get().cards };
    for (const step of steps) {
      const cat = step.plugin.split(".")[0];
      if (cards[cat]) {
        cards[cat] = { ...cards[cat], status: "idle", stepRunId: null };
      }
    }
    set({
      cards,
      pipelineRunning: true,
      currentRunId: null,
      errorMessage: null,
      currentStepIndex: 0,
      totalSteps: steps.length,
    });

    try {
      // Start the background task, passing any pre-loaded context.
      const existingContext = get().context ?? {};
      const { task_id } = await startPipelineTask(steps, existingContext);

      // Poll every 2 seconds until the task finishes.
      await new Promise<void>((resolve, reject) => {
        const interval = setInterval(async () => {
          try {
            const status = await getTaskStatus(task_id);

            // Update run ID as soon as it's available.
            if (status.run_id) {
              set({ currentRunId: status.run_id });
            }

            // Update progress counters.
            set({
              currentStepIndex: status.current_step_index,
              totalSteps: status.total_steps,
            });

            // Mark completed steps as "done".
            const c = { ...get().cards };
            for (const completed of status.completed_steps) {
              if (c[completed.category]) {
                c[completed.category] = {
                  ...c[completed.category],
                  status: "done",
                  stepRunId: completed.run_id,
                };
              }
            }

            // Mark the currently executing step as "running".
            if (status.current_step && c[status.current_step]) {
              c[status.current_step] = {
                ...c[status.current_step],
                status: "running",
              };
            }
            set({ cards: c });

            // Check terminal states.
            if (status.status === "completed") {
              clearInterval(interval);
              set({
                pipelineRunning: false,
                context: status.context as PipelineContext | null,
                currentRunId: status.run_id,
              });
              resolve();
            } else if (status.status === "error") {
              clearInterval(interval);
              // Mark any currently-running cards as error.
              const ec = { ...get().cards };
              for (const key of Object.keys(ec)) {
                if (ec[key].status === "running") {
                  ec[key] = { ...ec[key], status: "error" };
                }
              }
              set({
                cards: ec,
                pipelineRunning: false,
                errorMessage: status.error ?? "Pipeline failed",
              });
              resolve();
            }
          } catch (pollError) {
            clearInterval(interval);
            reject(pollError);
          }
        }, 2000);
      });
    } catch (error) {
      console.error("Pipeline error:", error);
      set({
        pipelineRunning: false,
        errorMessage:
          error instanceof Error ? error.message : "Pipeline failed",
      });
    }
  },

  clearError: () => {
    set({ errorMessage: null });
  },

  runSingleStep: async (runId: string, step: StepDefinition) => {
    const category = step.plugin.split(".")[0];

    // Mark card as running.
    const cards = { ...get().cards };
    if (cards[category]) {
      cards[category] = { ...cards[category], status: "running" };
      set({ cards });
    }

    try {
      const result = await executeStep(runId, step);

      // Mark card as done with the new step run ID.
      const c = { ...get().cards };
      if (c[result.category]) {
        c[result.category] = {
          ...c[result.category],
          status: "done",
          stepRunId: result.step_run_id,
        };
        set({ cards: c });
      }
    } catch (error) {
      console.error("Execute step error:", error);
      const c = { ...get().cards };
      if (c[category]) {
        c[category] = { ...c[category], status: "error" };
        set({ cards: c });
      }
    }
  },

  resetCards: () => {
    const cards = { ...get().cards };
    for (const key of Object.keys(cards)) {
      cards[key] = defaultCard();
    }
    set({
      cards,
      pipelineRunning: false,
      currentRunId: null,
      context: null,
      targetRunId: null,
      errorMessage: null,
      currentStepIndex: 0,
      totalSteps: 0,
    });
  },
}));
