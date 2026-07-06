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
import { toast } from "sonner";

import { fetchPluginRegistry } from "../api/plugins";
import {
  fetchPipelineRuns,
  fetchRunContext,
  fetchMlflowInfo,
  startPipelineTask,
  getTaskStatus,
  executeStepAsync,
  cancelTask,
  fetchRunningTasks,
} from "../api/pipelines";
import { ApiError } from "../api/errors";
import {
  loadLatestRunSnapshot,
  loadRunSnapshot,
  saveRunSnapshot,
  clearRunSnapshot,
} from "./runPersistence";
import type { PluginRegistry } from "../types/plugin";
import type {
  MlflowInfo,
  PipelineContext,
  PipelineRun,
  StepDefinition,
  TaskSummary,
} from "../types/pipeline";

// ── MLflow URL helpers ───────────────────────────────────

/**
 * Construct a deep link to a specific run in the MLflow UI.
 *
 * @param info  - MLflow connection info (base URL + experiment ID).
 * @param runId - The MLflow run ID to link to.
 * @returns Full URL to the run's detail page.
 */
export function mlflowRunUrl(info: MlflowInfo, runId: string): string {
  return `${info.ui_base_url}/#/experiments/${info.experiment_id}/runs/${runId}`;
}

/**
 * Construct a deep link to the experiment's run list in the MLflow UI.
 *
 * @param info - MLflow connection info (base URL + experiment ID).
 * @returns Full URL to the experiment page.
 */
export function mlflowExperimentUrl(info: MlflowInfo): string {
  return `${info.ui_base_url}/#/experiments/${info.experiment_id}/runs`;
}

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
/** Card input mode: "setup" for new config, "load" for loading a past run. */
export type CardMode = "setup" | "load";

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
  /** Whether the card is in "Set up new" or "Load from previous run" mode. */
  mode: CardMode;
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

  /** MLflow UI connection info for constructing deep links. */
  mlflowInfo: MlflowInfo | null;

  /** 0-based index of the step currently executing. */
  currentStepIndex: number;

  /** Total number of steps in the current pipeline run. */
  totalSteps: number;

  /** Task ID of the currently running background pipeline task. */
  currentTaskId: string | null;

  /** Whether a cancellation request has been sent but not yet acknowledged. */
  cancellationPending: boolean;

  /** Whether a hard ("Cancel now") abort has been requested. */
  abortPending: boolean;

  /**
   * Steps waiting for modal confirmation before launching.
   * When non-null the LaunchModal is visible.
   */
  pendingSteps: StepDefinition[] | null;

  /** Active (running) tasks backing the header "running tasks" menu. */
  runningTasks: TaskSummary[];

  // ── Actions ─────────────────────────────────────────

  /**
   * Fetch the plugin registry from the backend and initialise
   * a CardState for every category.
   */
  loadRegistry: () => Promise<void>;

  /** Fetch the list of past pipeline runs from the backend. */
  loadPastRuns: () => Promise<void>;

  /** Fetch MLflow UI base URL and experiment ID from the backend. */
  loadMlflowInfo: () => Promise<void>;

  /**
   * Load context from a past run and populate card states
   * up to and including the given category.
   *
   * Categories chronologically after `upToCategory` are left untouched.
   */
  loadFromPreviousRun: (runId: string, upToCategory: string) => Promise<void>;

  /** Select a plugin implementation for a category card. */
  selectPlugin: (category: string, pluginName: string) => void;

  /** Update a parameter value for a category card. */
  setParam: (category: string, paramName: string, value: string) => void;

  /** Toggle the "Configure" section of a category card. */
  toggleConfig: (category: string) => void;

  /**
   * Set the card mode ("setup" or "load") for a category.
   * When switching to "setup", all chronologically following one_time
   * cards are also reset to "setup" with no selected plugin.
   */
  setCardMode: (category: string, mode: CardMode) => void;

  /**
   * Stage steps for launch — opens the LaunchModal.
   * Pass `null` to close the modal without launching.
   */
  setPendingSteps: (steps: StepDefinition[] | null) => void;

  /**
   * Confirm launch from the modal: clears pendingSteps and
   * runs the pipeline with the user-provided tags, description, and
   * optional pipeline name.
   */
  confirmLaunch: (
    tags: Record<string, string>,
    description: string,
    pipelineName: string
  ) => Promise<void>;

  /**
   * Run the full pipeline (all one_time steps) via background task + polling.
   * Updates card statuses as polling responses arrive.
   */
  runPipeline: (
    steps: StepDefinition[],
    tags?: Record<string, string>,
    description?: string,
    pipelineName?: string
  ) => Promise<void>;

  /**
   * Restore an in-progress run after a page refresh from the persisted
   * session snapshot and resume polling. No-op if nothing was running.
   */
  resumeRun: () => Promise<void>;

  /** Refresh the list of running tasks backing the header menu. */
  loadRunningTasks: () => Promise<void>;

  /**
   * Load a specific running task into the main view and resume polling
   * it — restoring from the session snapshot when present, otherwise
   * rebuilding the card layout from the task's requested steps.
   */
  loadRunningTask: (summary: TaskSummary) => Promise<void>;

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

  /**
   * Request cancellation of the currently running pipeline.
   * Sets cancellationPending immediately; the polling loop will
   * detect the "cancelled" status and update the UI.
   */
  cancelPipeline: () => Promise<void>;

  /**
   * Hard-cancel ("Cancel now"): in addition to the graceful stop, signal
   * a cooperating plugin to abort the currently executing step. Can be
   * called to escalate after {@link cancelPipeline}.
   */
  abortPipeline: () => Promise<void>;

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
  mode: "setup",
});

// ── Polling helper ──────────────────────────────────────

type StoreSet = (partial: Partial<PipelineStore>) => void;
type StoreGet = () => PipelineStore;

/**
 * Token for the single in-flight poll loop. When a new run starts or a
 * different task is loaded, the previous loop's token is flipped so it
 * stops on its next tick — preventing two pollers from writing the store.
 */
let activePoll: { cancelled: boolean } | null = null;

/**
 * Rebuild a card layout from a task's requested steps. Used when loading
 * a running task with no session snapshot (e.g. one started in another
 * tab): each step's plugin + params populate its category card; the live
 * status is filled in by the first poll.
 */
function cardsFromSteps(
  base: Record<string, CardState>,
  steps: StepDefinition[]
): Record<string, CardState> {
  const cards = { ...base };
  for (const step of steps) {
    const category = step.plugin.split(".")[0];
    if (cards[category]) {
      cards[category] = {
        ...cards[category],
        selectedPlugin: step.plugin,
        params: Object.fromEntries(
          Object.entries(step.params ?? {}).map(([k, v]) => [k, String(v)])
        ),
        status: "idle",
        stepRunId: null,
      };
    }
  }
  return cards;
}

/**
 * Overlay pre-loaded upstream steps onto a rebuilt card layout. Each
 * category present in a task's initial context (steps that were "loaded
 * from a previous run" and so are absent from steps_requested) is marked
 * as a completed "load"-mode card carrying its step run id — mirroring
 * {@link loadFromPreviousRun}. Used when adopting a running task started
 * on top of such a selection, so its upstream cards show up as done too.
 */
function markLoadedContext(
  base: Record<string, CardState>,
  initialContext: PipelineContext | undefined
): Record<string, CardState> {
  if (!initialContext) return base;
  const cards = { ...base };
  for (const [category, data] of Object.entries(initialContext)) {
    if (cards[category]) {
      cards[category] = {
        ...cards[category],
        status: "done",
        stepRunId: data.run_id,
        mode: "load",
      };
    }
  }
  return cards;
}

/**
 * Reset to a clean idle state after polling a tracked run failed — the
 * task was evicted/404'd or the request errored. Shared by the
 * refresh-resume and load-task flows.
 */
function handleTrackingFailure(
  error: unknown,
  taskId: string,
  set: StoreSet,
  get: StoreGet
): void {
  const gone = error instanceof ApiError && error.status === 404;
  clearRunSnapshot(taskId);
  const cards = { ...get().cards };
  for (const key of Object.keys(cards)) {
    if (cards[key].status === "running") {
      cards[key] = { ...cards[key], status: "idle" };
    }
  }
  set({
    cards,
    pipelineRunning: false,
    currentTaskId: null,
    cancellationPending: false,
    abortPending: false,
    errorMessage: gone
      ? "The run is no longer available (it may have just finished, or the backend restarted)."
      : error instanceof Error
        ? error.message
        : "Failed to track the run.",
  });
}

/**
 * Poll a background pipeline task every 2s until it reaches a terminal
 * state, updating card statuses, progress, and toasts as responses arrive.
 *
 * Shared by `runPipeline` (fresh launch) and `resumeRun` (restore after a
 * refresh). On every tick the live state is mirrored into sessionStorage via
 * {@link saveRunSnapshot} so a refresh can restore the running layout; the
 * snapshot is cleared on any terminal state or polling error.
 *
 * @param skipMessageBacklog - When resuming, suppress toasts for messages
 *   already emitted before the refresh so only genuinely new ones fire.
 */
function pollTaskUntilDone(
  taskId: string,
  set: StoreSet,
  get: StoreGet,
  { skipMessageBacklog = false }: { skipMessageBacklog?: boolean } = {}
): Promise<void> {
  // Supersede any previous poller (e.g. switching tasks from the menu).
  if (activePoll) activePoll.cancelled = true;
  const abort = { cancelled: false };
  activePoll = abort;

  return new Promise<void>((resolve, reject) => {
    let seenMessageCount = 0;
    let firstTick = true;
    const interval = setInterval(async () => {
      // Stop quietly if a newer poller has taken over.
      if (abort.cancelled) {
        clearInterval(interval);
        resolve();
        return;
      }
      try {
        const status = await getTaskStatus(taskId);

        // Track the run id and load past runs until this run is in the list,
        // so completed-step cards show its friendly name, not the raw id.
        if (status.run_id) {
          set({ currentRunId: status.run_id });
          if (!get().pastRuns.some((r) => r.run_id === status.run_id)) {
            get().loadPastRuns();
          }
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

        // On a resume the message backlog was already shown before the
        // refresh — adopt the current count once so only new ones toast.
        if (firstTick && skipMessageBacklog) {
          seenMessageCount = status.messages.length;
        }
        firstTick = false;

        // Fire toasts for any new plugin notifications.
        const newMessages = status.messages.slice(seenMessageCount);
        for (const msg of newMessages) {
          if (msg.level === "success") toast.success(msg.text);
          else if (msg.level === "warning") toast.warning(msg.text);
          else if (msg.level === "error") toast.error(msg.text);
          else toast.info(msg.text);
        }
        seenMessageCount += newMessages.length;

        // Mirror live state so a refresh can restore the running layout.
        saveRunSnapshot({
          taskId,
          cards: get().cards,
          currentRunId: get().currentRunId,
          currentStepIndex: status.current_step_index,
          totalSteps: status.total_steps,
          savedAt: Date.now(),
        });

        // Check terminal states.
        if (status.status === "completed") {
          clearInterval(interval);
          clearRunSnapshot(taskId);
          set({
            pipelineRunning: false,
            context: status.context as PipelineContext | null,
            currentRunId: status.run_id,
          });
          // Refresh past runs so the new run name is available.
          get().loadPastRuns();
          resolve();
        } else if (status.status === "cancelled") {
          clearInterval(interval);
          clearRunSnapshot(taskId);
          // Mark any currently-running cards as idle.
          const cc = { ...get().cards };
          for (const key of Object.keys(cc)) {
            if (cc[key].status === "running") {
              cc[key] = { ...cc[key], status: "idle" };
            }
          }
          set({
            cards: cc,
            pipelineRunning: false,
            cancellationPending: false,
            abortPending: false,
            currentTaskId: null,
            errorMessage: status.error ?? "Pipeline cancelled by user.",
          });
          resolve();
        } else if (status.status === "error") {
          clearInterval(interval);
          clearRunSnapshot(taskId);
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
            currentTaskId: null,
            errorMessage: status.error ?? "Pipeline failed",
          });
          resolve();
        }
      } catch (pollError) {
        clearInterval(interval);
        clearRunSnapshot(taskId);
        reject(pollError);
      }
    }, 2000);
  });
}

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
  mlflowInfo: null,
  currentStepIndex: 0,
  totalSteps: 0,
  currentTaskId: null,
  cancellationPending: false,
  abortPending: false,
  pendingSteps: null,
  runningTasks: [],

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

  loadMlflowInfo: async () => {
    const mlflowInfo = await fetchMlflowInfo();
    set({ mlflowInfo });
  },

  loadFromPreviousRun: async (runId: string, upToCategory: string) => {
    let fullContext: PipelineContext;
    try {
      fullContext = await fetchRunContext(runId);
    } catch (error) {
      const msg =
        error instanceof ApiError && error.status === 404
          ? `Cannot load run ${runId}: its context.json is missing. ` +
            `The run exists in MLflow but its artifacts are unavailable.`
          : `Failed to load run ${runId}: ${
              error instanceof Error ? error.message : String(error)
            }`;
      set({ errorMessage: msg });
      return;
    }

    const { registry, cards: prev } = get();
    if (!registry) return;

    // Determine which one_time categories are at or before the target.
    const sorted = Object.entries(registry)
      .filter(([, e]) => e.category_info.type === "one_time")
      .sort(([, a], [, b]) => a.category_info.order - b.category_info.order);

    const allowedCategories = new Set<string>();
    for (const [key] of sorted) {
      allowedCategories.add(key);
      if (key === upToCategory) break;
    }

    // Build scoped context with only the allowed categories.
    const scopedContext: PipelineContext = {};
    for (const [cat, data] of Object.entries(fullContext)) {
      if (allowedCategories.has(cat)) {
        scopedContext[cat] = data;
      }
    }

    // Update cards: mark allowed categories as "done" + "load" mode,
    // and reset any one_time cards that are NOT in the loaded context.
    const cards = { ...prev };
    for (const [key] of sorted) {
      if (!cards[key]) continue;
      if (allowedCategories.has(key) && scopedContext[key]) {
        // Clean base so no stale setup selection leaks into a loaded card.
        cards[key] = {
          ...defaultCard(),
          status: "done",
          stepRunId: scopedContext[key].run_id,
          mode: "load",
        };
      } else {
        cards[key] = { ...defaultCard() };
      }
    }

    set({
      context: scopedContext,
      currentRunId: runId,
      targetRunId: runId,
      cards,
      errorMessage: null,
    });
  },

  selectPlugin: (category: string, pluginName: string) => {
    const cards = { ...get().cards };
    cards[category] = {
      ...cards[category],
      selectedPlugin: pluginName,
      params: {},
      configOpen: false,
      status: "idle",
      stepRunId: null,
    };
    set({ cards });
  },

  setParam: (category: string, paramName: string, value: string) => {
    const cards = { ...get().cards };
    const current = cards[category];
    const isRunning = current.status === "running";
    cards[category] = {
      ...current,
      params: { ...current.params, [paramName]: value },
      // Preserve "running" so cascade-driven onChange("") (from the
      // dropdown stale-value cleanup effect) does not flip the card
      // back to "idle" while the backend is still computing.
      status: isRunning ? current.status : "idle",
      stepRunId: isRunning ? current.stepRunId : null,
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

  setCardMode: (category: string, mode: CardMode) => {
    const { registry, cards: prev, context: prevContext } = get();
    const cards = { ...prev };
    // Prune context alongside every card we reset, so a stale run_id from a
    // previously loaded run can't leak into the next pipeline request.
    const context: PipelineContext = { ...(prevContext ?? {}) };
    if (mode === "setup") {
      // Fully reset the clicked card when switching away from "load".
      cards[category] = { ...defaultCard() };
      delete context[category];
    } else {
      cards[category] = { ...cards[category], mode };
    }

    // When switching to "setup", also reset all following one_time cards
    // and ALL multi_run cards (their baseline config has changed).
    if (mode === "setup" && registry) {
      const sorted = Object.entries(registry)
        .filter(([, e]) => e.category_info.type === "one_time")
        .sort(([, a], [, b]) => a.category_info.order - b.category_info.order);

      let pastTarget = false;
      for (const [key] of sorted) {
        if (key === category) {
          pastTarget = true;
          continue;
        }
        if (pastTarget && cards[key]) {
          cards[key] = { ...defaultCard() };
          delete context[key];
        }
      }

      for (const [key, entry] of Object.entries(registry)) {
        if (entry.category_info.type === "multi_run" && cards[key]) {
          cards[key] = { ...defaultCard() };
          delete context[key];
        }
      }
    }

    set({ cards, context });
  },

  setPendingSteps: (steps: StepDefinition[] | null) => {
    set({ pendingSteps: steps });
  },

  confirmLaunch: async (
    tags: Record<string, string>,
    description: string,
    pipelineName: string
  ) => {
    const steps = get().pendingSteps;
    if (!steps) return;
    set({ pendingSteps: null });
    await get().runPipeline(steps, tags, description, pipelineName);
  },

  runPipeline: async (
    steps: StepDefinition[],
    tags: Record<string, string> = {},
    description: string = "",
    pipelineName: string = ""
  ) => {
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
      const { task_id } = await startPipelineTask(
        steps,
        existingContext,
        tags,
        description,
        pipelineName
      );
      set({ currentTaskId: task_id });

      // Persist immediately so even a refresh within the first poll
      // interval restores the running layout.
      saveRunSnapshot({
        taskId: task_id,
        cards: get().cards,
        currentRunId: get().currentRunId,
        currentStepIndex: 0,
        totalSteps: steps.length,
        savedAt: Date.now(),
      });

      // Poll every 2 seconds until the task finishes.
      await pollTaskUntilDone(task_id, set, get);
    } catch (error) {
      console.error("Pipeline error:", error);
      clearRunSnapshot(get().currentTaskId ?? "");
      set({
        pipelineRunning: false,
        errorMessage:
          error instanceof Error ? error.message : "Pipeline failed",
      });
    }
  },

  resumeRun: async () => {
    // Already tracking a run in this session — don't start a second poller.
    if (get().pipelineRunning) return;

    const snapshot = loadLatestRunSnapshot();
    if (!snapshot) return;

    // Restore the running layout over the registry-initialised card skeleton.
    set({
      cards: { ...get().cards, ...snapshot.cards },
      pipelineRunning: true,
      currentTaskId: snapshot.taskId,
      currentRunId: snapshot.currentRunId,
      currentStepIndex: snapshot.currentStepIndex,
      totalSteps: snapshot.totalSteps,
      errorMessage: null,
    });

    try {
      await pollTaskUntilDone(snapshot.taskId, set, get, {
        skipMessageBacklog: true,
      });
    } catch (error) {
      handleTrackingFailure(error, snapshot.taskId, set, get);
    }
  },

  loadRunningTasks: async () => {
    try {
      const runningTasks = await fetchRunningTasks();
      set({ runningTasks });
    } catch (error) {
      // Non-fatal: the menu just keeps its previous contents this tick.
      console.error("Failed to load running tasks:", error);
    }
  },

  loadRunningTask: async (summary: TaskSummary) => {
    // Already the active run — nothing to switch to.
    if (get().currentTaskId === summary.task_id && get().pipelineRunning) {
      return;
    }

    // Prefer the exact session snapshot (full card fidelity); otherwise
    // rebuild the layout from the task's requested steps plus any upstream
    // steps it was loaded on top of, so cards that came from a previous run
    // are restored as done too (steps_requested holds only the new steps).
    const snapshot = loadRunSnapshot(summary.task_id);
    set({
      cards: snapshot
        ? { ...get().cards, ...snapshot.cards }
        : markLoadedContext(
            cardsFromSteps(get().cards, summary.steps_requested),
            summary.initial_context
          ),
      pipelineRunning: true,
      currentTaskId: summary.task_id,
      currentRunId: summary.run_id,
      currentStepIndex: snapshot?.currentStepIndex ?? summary.current_step_index,
      totalSteps: snapshot?.totalSteps ?? summary.total_steps,
      context: snapshot ? null : summary.initial_context,
      errorMessage: null,
    });

    try {
      await pollTaskUntilDone(summary.task_id, set, get, {
        skipMessageBacklog: true,
      });
    } catch (error) {
      handleTrackingFailure(error, summary.task_id, set, get);
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
      set({ cards, errorMessage: null });
    }

    try {
      const { task_id } = await executeStepAsync(runId, step);

      // Poll every 2 seconds until the task reaches a terminal state.
      await new Promise<void>((resolve, reject) => {
        let seenMessageCount = 0;
        const interval = setInterval(async () => {
          try {
            const status = await getTaskStatus(task_id);

            // Fire toasts for any new plugin notifications.
            const newMessages = status.messages.slice(seenMessageCount);
            for (const msg of newMessages) {
              if (msg.level === "success") toast.success(msg.text);
              else if (msg.level === "warning") toast.warning(msg.text);
              else if (msg.level === "error") toast.error(msg.text);
              else toast.info(msg.text);
            }
            seenMessageCount += newMessages.length;

            if (status.status === "completed") {
              clearInterval(interval);
              const completed = status.completed_steps[0];
              const c = { ...get().cards };
              if (completed && c[completed.category]) {
                c[completed.category] = {
                  ...c[completed.category],
                  status: "done",
                  stepRunId: completed.run_id,
                };
                set({ cards: c });
              }
              resolve();
            } else if (status.status === "error") {
              clearInterval(interval);
              const c = { ...get().cards };
              if (c[category]) {
                c[category] = { ...c[category], status: "error" };
              }
              set({
                cards: c,
                errorMessage: status.error ?? "Step execution failed.",
              });
              resolve();
            } else if (status.status === "cancelled") {
              clearInterval(interval);
              const c = { ...get().cards };
              if (c[category]) {
                c[category] = { ...c[category], status: "idle" };
              }
              set({
                cards: c,
                errorMessage: status.error ?? "Step cancelled.",
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
      console.error("Execute step async error:", error);
      const c = { ...get().cards };
      if (c[category]) {
        c[category] = { ...c[category], status: "error" };
      }
      set({
        cards: c,
        errorMessage:
          error instanceof Error ? error.message : "Step execution failed.",
      });
    }
  },

  cancelPipeline: async () => {
    const taskId = get().currentTaskId;
    if (!taskId) return;

    set({ cancellationPending: true });

    try {
      await cancelTask(taskId);
    } catch (error) {
      console.error("Cancel pipeline error:", error);
      set({ cancellationPending: false });
    }
  },

  abortPipeline: async () => {
    const taskId = get().currentTaskId;
    if (!taskId) return;

    // Implies graceful too — no further steps start.
    set({ cancellationPending: true, abortPending: true });

    try {
      await cancelTask(taskId, "now");
    } catch (error) {
      console.error("Abort pipeline error:", error);
      set({ abortPending: false });
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
      currentTaskId: null,
      cancellationPending: false,
      abortPending: false,
      pendingSteps: null,
    });
  },
}));
