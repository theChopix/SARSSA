import { create } from "zustand";
import type {
  PluginCategory,
  PipelineRun,
  StepDefinition,
} from "@/api/types";
import {
  fetchPluginRegistry,
  fetchPipelineRuns,
  runPipeline,
} from "@/api/client";

// --- Per-card state ---

export type CardMode = "new" | "load";
export type CardStatus = "idle" | "running" | "done" | "error";

export interface CardState {
  mode: CardMode;
  selectedPlugin: string | null;
  params: Record<string, Record<string, unknown>>;
  configOpen: string | null;
  status: CardStatus;
  error: string | null;
  loadedRunId: string | null;
}

// --- Store shape ---

interface PipelineStore {
  // Data from backend
  categories: PluginCategory[];
  previousRuns: PipelineRun[];
  loading: boolean;

  // Per-category card state, keyed by category name
  cards: Record<string, CardState>;

  // Global pipeline execution
  pipelineStatus: "idle" | "running" | "done" | "error";
  pipelineError: string | null;

  // Actions: data fetching
  fetchRegistry: () => Promise<void>;
  fetchRuns: () => Promise<void>;

  // Actions: card state
  setCardMode: (category: string, mode: CardMode) => void;
  selectPlugin: (category: string, pluginPath: string) => void;
  setParam: (category: string, pluginPath: string, paramName: string, value: unknown) => void;
  toggleConfig: (category: string, pluginPath: string | null) => void;
  setCardStatus: (category: string, status: CardStatus, error?: string) => void;
  setLoadedRun: (category: string, runId: string | null) => void;

  // Actions: pipeline execution
  setPipelineStatus: (status: "idle" | "running" | "done" | "error", error?: string) => void;
  runUpTo: (targetCategory: string) => Promise<void>;
  runFullPipeline: () => Promise<void>;
}

function defaultCardState(): CardState {
  return {
    mode: "new",
    selectedPlugin: null,
    params: {},
    configOpen: null,
    status: "idle",
    error: null,
    loadedRunId: null,
  };
}

export const usePipelineStore = create<PipelineStore>((set, get) => ({
  categories: [],
  previousRuns: [],
  loading: false,
  cards: {},
  pipelineStatus: "idle",
  pipelineError: null,

  fetchRegistry: async () => {
    set({ loading: true });
    try {
      const registry = await fetchPluginRegistry();
      const cards: Record<string, CardState> = {};
      for (const cat of registry.categories) {
        const existing = get().cards[cat.name];
        if (existing) {
          cards[cat.name] = existing;
        } else {
          const card = defaultCardState();
          if (cat.implementations.length > 0) {
            card.selectedPlugin = cat.implementations[0].plugin_path;
          }
          cards[cat.name] = card;
        }
      }
      set({ categories: registry.categories, cards, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  fetchRuns: async () => {
    try {
      const runs = await fetchPipelineRuns();
      set({ previousRuns: runs });
    } catch {
      // silently ignore
    }
  },

  setCardMode: (category, mode) =>
    set((state) => ({
      cards: {
        ...state.cards,
        [category]: { ...(state.cards[category] ?? defaultCardState()), mode },
      },
    })),

  selectPlugin: (category, pluginPath) =>
    set((state) => ({
      cards: {
        ...state.cards,
        [category]: {
          ...(state.cards[category] ?? defaultCardState()),
          selectedPlugin: pluginPath,
        },
      },
    })),

  setParam: (category, pluginPath, paramName, value) =>
    set((state) => {
      const card = state.cards[category] ?? defaultCardState();
      const pluginParams = card.params[pluginPath] ?? {};
      return {
        cards: {
          ...state.cards,
          [category]: {
            ...card,
            params: {
              ...card.params,
              [pluginPath]: { ...pluginParams, [paramName]: value },
            },
          },
        },
      };
    }),

  toggleConfig: (category, pluginPath) =>
    set((state) => {
      const card = state.cards[category] ?? defaultCardState();
      return {
        cards: {
          ...state.cards,
          [category]: {
            ...card,
            configOpen: card.configOpen === pluginPath ? null : pluginPath,
          },
        },
      };
    }),

  setCardStatus: (category, status, error) =>
    set((state) => ({
      cards: {
        ...state.cards,
        [category]: {
          ...(state.cards[category] ?? defaultCardState()),
          status,
          error: error ?? null,
        },
      },
    })),

  setLoadedRun: (category, runId) =>
    set((state) => ({
      cards: {
        ...state.cards,
        [category]: {
          ...(state.cards[category] ?? defaultCardState()),
          loadedRunId: runId,
        },
      },
    })),

  setPipelineStatus: (status, error) =>
    set({ pipelineStatus: status, pipelineError: error ?? null }),

  runUpTo: async (targetCategory) => {
    const { categories, cards } = get();
    const oneTime = categories
      .filter((c) => c.type === "one_time")
      .sort((a, b) => a.order - b.order);

    const targetIdx = oneTime.findIndex((c) => c.name === targetCategory);
    if (targetIdx === -1) return;

    const stepsToRun = oneTime.slice(0, targetIdx + 1);

    const steps: StepDefinition[] = stepsToRun.map((cat) => {
      const card = cards[cat.name];
      const pluginPath = card?.selectedPlugin ?? cat.implementations[0]?.plugin_path;
      const userParams = card?.params[pluginPath] ?? {};
      return { plugin: pluginPath, params: userParams };
    });

    // Mark cards as running
    for (const cat of stepsToRun) {
      get().setCardStatus(cat.name, "running");
    }
    get().setPipelineStatus("running");

    try {
      await runPipeline({}, { steps });
      for (const cat of stepsToRun) {
        get().setCardStatus(cat.name, "done");
      }
      get().setPipelineStatus("done");
      get().fetchRuns();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      for (const cat of stepsToRun) {
        get().setCardStatus(cat.name, "error", msg);
      }
      get().setPipelineStatus("error", msg);
    }
  },

  runFullPipeline: async () => {
    const { categories } = get();
    const oneTime = categories
      .filter((c) => c.type === "one_time")
      .sort((a, b) => a.order - b.order);
    const last = oneTime[oneTime.length - 1];
    if (last) {
      await get().runUpTo(last.name);
    }
  },
}));
