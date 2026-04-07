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
  subscribeToPipelineStream,
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

  /** Whether the full pipeline is currently streaming. */
  pipelineRunning: boolean;

  /** MLflow run ID of the current/last pipeline run. */
  currentRunId: string | null;

  /** Context from the current or loaded pipeline run. */
  context: PipelineContext | null;

  /** List of past pipeline runs (for "Load from previous run"). */
  pastRuns: PipelineRun[];

  /** The run ID selected in "Load from previous run" mode. */
  targetRunId: string | null;

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
   * Run the full pipeline (all one_time steps) via SSE streaming.
   * Updates card statuses in real time as events arrive.
   */
  runPipeline: (steps: StepDefinition[]) => Promise<void>;

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
    set({ pipelineRunning: true });

    // Reset all cards to idle before starting.
    const cards = { ...get().cards };
    for (const key of Object.keys(cards)) {
      cards[key] = { ...cards[key], status: "idle", stepRunId: null };
    }
    set({ cards, context: null, currentRunId: null });

    await subscribeToPipelineStream(steps, {
      onRunStarted: (data) => {
        set({ currentRunId: data.run_id });
      },

      onStepStarted: (data) => {
        const c = { ...get().cards };
        if (c[data.category]) {
          c[data.category] = { ...c[data.category], status: "running" };
          set({ cards: c });
        }
      },

      onStepCompleted: (data) => {
        const c = { ...get().cards };
        if (c[data.category]) {
          c[data.category] = {
            ...c[data.category],
            status: "done",
            stepRunId: data.run_id,
          };
          set({ cards: c });
        }
      },

      onRunCompleted: (data) => {
        set({
          pipelineRunning: false,
          context: data.context,
          currentRunId: data.run_id,
        });
      },

      onError: (error) => {
        console.error("Pipeline stream error:", error);
        set({ pipelineRunning: false });
      },
    });
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
    });
  },
}));
