/**
 * PipelineCard — a single pipeline category card.
 *
 * This is the main UI building block. One card is rendered for each
 * category in the plugin registry (e.g. "Dataset Loading", "Training CFM").
 *
 * The card is **data-driven**: it receives a category key and reads
 * everything it needs from the Zustand store + registry. This means
 * the same component works for both "one_time" and "multi_run" cards.
 *
 * ┌──────────────────────────────────────────────────────────┐
 * │  React component anatomy                                 │
 * │                                                          │
 * │  A component is a function that returns JSX.             │
 * │  "Props" are the arguments passed to the component:      │
 * │                                                          │
 * │    <PipelineCard categoryKey="dataset_loading" />        │
 * │                  ↑ this becomes props.categoryKey         │
 * │                                                          │
 * │  Inside the component, we read from the store using      │
 * │  selectors, and call store actions on user interaction.  │
 * └──────────────────────────────────────────────────────────┘
 *
 * Visual structure of a one_time card:
 *
 *   ┌─────────────────────────────────────┐
 *   │  Category Title          ✓ / ⟳ / ✗ │  ← header + status
 *   │                                     │
 *   │  [Set up new] [Load from prev run]  │  ← mode toggle (one_time only)
 *   │                                     │
 *   │  ◉ Plugin A          ⚙ Configure   │  ← "Set up new" mode
 *   │  ○ Plugin B          ⚙ Configure   │
 *   │  ┌─ Parameter form (if open) ────┐  │
 *   │  │  epochs  (int)    [100]       │  │
 *   │  └──────────────────────────────-┘  │
 *   │  [ Run up to this step ]            │
 *   │                                     │
 *   │  OR                                 │
 *   │                                     │
 *   │  ▾ Select a previous run            │  ← "Load from prev" mode
 *   └─────────────────────────────────────┘
 */

import { useEffect, useState, useCallback, useRef } from "react";
import { Settings, CheckCircle2, Loader2, AlertCircle, Eye } from "lucide-react";

import { usePipelineStore, mlflowRunUrl } from "../store/pipelineStore";
import { fetchParamChoices } from "../api/plugins";
import type { ParamChoice } from "../api/plugins";
import type { ImplementationInfo, ParameterInfo } from "../types/plugin";
import type { MlflowInfo, PipelineContext } from "../types/pipeline";
import type { CardStatus, CardMode } from "../store/pipelineStore";

// ── Props ───────────────────────────────────────────────

/**
 * Props accepted by PipelineCard.
 *
 * - `categoryKey`  – The registry key (e.g. "dataset_loading").
 * - `onRunUpTo`    – Called when user clicks "Run up to this step".
 *                     Only provided for one_time cards.
 * - `onExecuteStep`– Called when user clicks "Execute step".
 *                     Only provided for multi_run cards.
 */
interface PipelineCardProps {
  categoryKey: string;
  onRunUpTo?: (categoryKey: string) => void;
  onExecuteStep?: (categoryKey: string) => void;
}

// ── Status icon helper ──────────────────────────────────

/**
 * Renders a small icon in the card header based on execution status.
 *
 * - idle    → nothing
 * - running → spinning loader
 * - done    → green checkmark
 * - error   → red alert
 */
function StatusIcon({ status }: { status: CardStatus }) {
  switch (status) {
    case "running":
      return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
    case "done":
      return <CheckCircle2 className="h-5 w-5 text-green-500" />;
    case "error":
      return <AlertCircle className="h-5 w-5 text-red-500" />;
    default:
      return null;
  }
}

// ── Parameter form row ──────────────────────────────────

/**
 * A single row in the parameter configuration form.
 *
 * Shows: param name, type badge, and either a text input or
 * a dropdown depending on the `widget` field.
 *
 * ┌──────────────────────────────────────────────┐
 * │  epochs  (int)    [ 100                    ] │  ← text
 * │  neuron  (str)    [ ▾ sci-fi [neuron id 0] ] │  ← dropdown
 * └──────────────────────────────────────────────┘
 */
function ParamRow({
  param,
  value,
  onChange,
  context,
}: {
  param: ParameterInfo;
  value: string;
  onChange: (value: string) => void;
  context: PipelineContext | null;
}) {
  return (
    <div className="flex items-center gap-3 py-1.5">
      {/* Parameter name */}
      <span className="text-sm text-gray-700 min-w-[100px]">
        {param.name}
      </span>

      {/* Type badge */}
      <span className="text-xs text-gray-400 min-w-[40px]">
        ({param.type})
      </span>

      {/* Widget: dropdown or text input */}
      {param.widget === "dropdown" && param.widget_config?.choices_endpoint ? (
        <DropdownSelect
          param={param}
          value={value}
          onChange={onChange}
          context={context}
        />
      ) : (
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={param.required ? "required" : ""}
          className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded
                     focus:outline-none focus:ring-2 focus:ring-blue-400
                     text-gray-800 bg-white"
        />
      )}
    </div>
  );
}

// ── Dropdown select for dynamic choices ─────────────────

/**
 * A `<select>` dropdown that fetches its options from the
 * backend param-choices endpoint.
 *
 * Requires a `run_id` from the pipeline context (looked up
 * via `widget_config.run_id_source`).
 */
function DropdownSelect({
  param,
  value,
  onChange,
  context,
}: {
  param: ParameterInfo;
  value: string;
  onChange: (value: string) => void;
  context: PipelineContext | null;
}) {
  const [options, setOptions] = useState<ParamChoice[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const wrapperRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const endpoint = param.widget_config!.choices_endpoint!;
  const sourceStep = param.widget_config!.run_id_source;
  const runId = sourceStep ? context?.[sourceStep]?.run_id : null;

  // Fetch options when the run_id becomes available.
  const loadOptions = useCallback(async () => {
    if (!runId) return;

    setLoading(true);
    setError(null);
    try {
      const choices = await fetchParamChoices(endpoint, runId);
      setOptions(choices);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load options");
    } finally {
      setLoading(false);
    }
  }, [endpoint, runId]);

  useEffect(() => {
    loadOptions();
  }, [loadOptions]);

  // Close dropdown when clicking outside.
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
        setFilter("");
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Auto-focus the search input when the dropdown opens.
  useEffect(() => {
    if (open) searchRef.current?.focus();
  }, [open]);

  // ── Render states ───────────────────────────────────

  if (!runId) {
    return (
      <span className="flex-1 text-sm text-gray-400 italic">
        Run the {sourceStep ?? "upstream"} step first
      </span>
    );
  }

  if (loading) {
    return (
      <span className="flex-1 text-sm text-gray-400 italic flex items-center gap-1">
        <Loader2 className="h-3 w-3 animate-spin" /> Loading options…
      </span>
    );
  }

  if (error) {
    return (
      <span className="flex-1 text-sm text-red-500">
        {error}
      </span>
    );
  }

  const selectedLabel = options.find((o) => o.value === value)?.label;

  return (
    <div ref={wrapperRef} className="flex-1 relative">
      {/* Trigger button */}
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="w-full px-2 py-1 text-sm border border-gray-300 rounded
                   focus:outline-none focus:ring-2 focus:ring-blue-400
                   text-gray-800 bg-white text-left flex items-center justify-between
                   cursor-pointer"
      >
        <span className={selectedLabel ? "" : "text-gray-400"}>
          {selectedLabel ?? "— select —"}
        </span>
        <svg className="h-4 w-4 text-gray-400 shrink-0" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
        </svg>
      </button>

      {/* Options list — opens below, left-aligned */}
      {open && (
        <div
          className="absolute left-0 right-0 top-full mt-1 z-50
                     rounded border border-gray-300 bg-white shadow-lg"
        >
          {/* Search input */}
          <div className="sticky top-0 bg-white p-1.5 border-b border-gray-200">
            <input
              ref={searchRef}
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Type to filter…"
              className="w-full px-2 py-1 text-sm border border-gray-300 rounded
                         focus:outline-none focus:ring-2 focus:ring-blue-400
                         text-gray-800 bg-white"
            />
          </div>

          {/* Filtered options */}
          <ul className="max-h-52 overflow-y-auto">
            {options
              .filter((o) => o.label.toLowerCase().includes(filter.toLowerCase()))
              .map((opt) => (
                <li
                  key={opt.value}
                  onClick={() => { onChange(opt.value); setOpen(false); setFilter(""); }}
                  className={`px-2 py-1.5 text-sm cursor-pointer hover:bg-blue-50
                    ${opt.value === value ? "bg-blue-100 font-medium" : "text-gray-800"}`}
                >
                  {opt.label}
                </li>
              ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── Plugin row ──────────────────────────────────────────

/**
 * A single plugin implementation row.
 *
 * Shows a radio button, plugin display name, and a "Configure"
 * button that toggles the parameter form.
 *
 * ┌──────────────────────────────────────────────┐
 * │  ◉ ELSA Trainer                 ⚙ Configure │
 * └──────────────────────────────────────────────┘
 */
function PluginRow({
  impl,
  isSelected,
  onSelect,
  onToggleConfig,
  disabled,
}: {
  impl: ImplementationInfo;
  isSelected: boolean;
  onSelect: () => void;
  onToggleConfig: () => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center justify-between py-1">
      {/* Radio button + name */}
      <label className={`flex items-center gap-2 ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}>
        <input
          type="radio"
          checked={isSelected}
          onChange={onSelect}
          disabled={disabled}
          className="accent-blue-500"
        />
        <span className="text-sm text-gray-800">{impl.display_name}</span>
      </label>

      {/* Configure button (only if plugin has params) */}
      {impl.params.length > 0 && (
        <button
          onClick={onToggleConfig}
          disabled={disabled}
          className="flex items-center gap-1 text-xs text-blue-500
                     hover:text-blue-700 disabled:opacity-50
                     disabled:cursor-not-allowed transition-colors cursor-pointer"
        >
          <Settings className="h-3.5 w-3.5" />
          Configure
        </button>
      )}
    </div>
  );
}

// ── Run info block (locked label + deep links) ──────────

/**
 * Displays a locked run-name label and MLflow deep links.
 *
 * Shown on completed cards to identify which pipeline run
 * produced the result and provide direct navigation to MLflow.
 *
 * ┌──────────────────────────────────────────────┐
 * │  pipeline_run_2025-04-12_18-30...            │
 * │  see pipeline run | see step run             │
 * └──────────────────────────────────────────────┘
 */
function RunInfoBlock({
  pipelineRunId,
  stepRunId,
  pipelineRunName,
  mlflowInfo,
}: {
  pipelineRunId: string | null;
  stepRunId: string | null;
  pipelineRunName: string;
  mlflowInfo: MlflowInfo | null;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      {/* Locked run name label */}
      <div className="w-full px-3 py-2 text-sm border border-gray-200
                      rounded-md bg-gray-50 text-gray-600 font-mono truncate">
        {pipelineRunName}
      </div>

      {/* Deep links */}
      {mlflowInfo && (
        <div className="flex items-center gap-1.5 text-sm">
          {pipelineRunId && (
            <a
              href={mlflowRunUrl(mlflowInfo, pipelineRunId)}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:text-blue-700 hover:underline"
            >
              See pipeline run
            </a>
          )}
          {pipelineRunId && stepRunId && (
            <span className="text-gray-300">|</span>
          )}
          {stepRunId && (
            <a
              href={mlflowRunUrl(mlflowInfo, stepRunId)}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:text-blue-700 hover:underline"
            >
              See step run
            </a>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main component ──────────────────────────────────────

/**
 * PipelineCard — renders one category card.
 *
 * This is the main export. It composes the sub-components above
 * and connects them to the Zustand store.
 */
export default function PipelineCard({
  categoryKey,
  onRunUpTo,
  onExecuteStep,
}: PipelineCardProps) {
  // ── Read from store ─────────────────────────────────
  // Each selector picks one slice → component only re-renders
  // when that specific slice changes.

  const entry = usePipelineStore((s) => s.registry?.[categoryKey]);
  const card = usePipelineStore((s) => s.cards[categoryKey]);
  const selectPlugin = usePipelineStore((s) => s.selectPlugin);
  const setParam = usePipelineStore((s) => s.setParam);
  const toggleConfig = usePipelineStore((s) => s.toggleConfig);
  const pipelineRunning = usePipelineStore((s) => s.pipelineRunning);
  const setCardMode = usePipelineStore((s) => s.setCardMode);
  const pastRuns = usePipelineStore((s) => s.pastRuns);
  const loadPastRuns = usePipelineStore((s) => s.loadPastRuns);
  const loadFromPreviousRun = usePipelineStore((s) => s.loadFromPreviousRun);
  const currentRunId = usePipelineStore((s) => s.currentRunId);
  const mlflowInfo = usePipelineStore((s) => s.mlflowInfo);
  const context = usePipelineStore((s) => s.context);
  const allOneTimeDone = usePipelineStore((s) => {
    if (!s.registry) return false;
    return Object.entries(s.registry)
      .filter(([, e]) => e.category_info.type === "one_time")
      .every(([key]) => s.cards[key]?.status === "done");
  });
  const anyStepRunning = usePipelineStore((s) =>
    Object.values(s.cards).some((c) => c.status === "running")
  );

  const busy = pipelineRunning || anyStepRunning;

  // Guard: if registry hasn't loaded yet, render nothing.
  if (!entry || !card) return null;

  const { category_info, implementations } = entry;

  // Find the currently selected implementation object.
  const selectedImpl = implementations.find(
    (impl) => impl.plugin_name === card.selectedPlugin
  );

  // Is this a multi_run card?
  const isMultiRun = category_info.type === "multi_run";

  // Card mode from the store ("setup" or "load").
  const cardMode: CardMode = card.mode;

  // Fetch past runs when any card switches to "load" mode.
  useEffect(() => {
    if (cardMode === "load") {
      loadPastRuns();
    }
  }, [cardMode, loadPastRuns]);

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-5 flex flex-col gap-3">
      {/* ── Card header ──────────────────────────────── */}
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-gray-900">
          {category_info.display_name}
        </h3>
        <StatusIcon status={card.status} />
      </div>

      {/* ── Error text ────────────────────────────────── */}
      {card.status === "error" && (
        <p className="text-xs text-red-500">Step failed during execution.</p>
      )}

      {/* ── Mode toggle (one_time cards only, hidden when done) ── */}
      {!isMultiRun && card.status !== "done" && (
        <div className="flex rounded-md overflow-hidden border border-gray-300">
          <button
            onClick={() => setCardMode(categoryKey, "setup")}
            disabled={busy}
            className={`flex-1 py-1.5 text-xs font-medium transition-colors cursor-pointer
                        disabled:cursor-not-allowed ${
              cardMode === "setup"
                ? "bg-blue-500 text-white"
                : "bg-white text-gray-600 hover:bg-gray-50"
            }`}
          >
            Set up new
          </button>
          <button
            onClick={() => setCardMode(categoryKey, "load")}
            disabled={busy}
            className={`flex-1 py-1.5 text-xs font-medium transition-colors cursor-pointer
                        disabled:cursor-not-allowed border-l border-gray-300 ${
              cardMode === "load"
                ? "bg-blue-500 text-white"
                : "bg-white text-gray-600 hover:bg-gray-50"
            }`}
          >
            Load from previous run
          </button>
        </div>
      )}

      {/* ── Completed state (one_time): show run info ── */}
      {!isMultiRun && card.status === "done" && (
        <>
          <RunInfoBlock
            pipelineRunId={currentRunId}
            stepRunId={card.stepRunId}
            pipelineRunName={
              pastRuns.find((r) => r.run_id === currentRunId)?.run_name
              ?? currentRunId
              ?? "\u2014"
            }
            mlflowInfo={mlflowInfo}
          />
          <button
            onClick={() => setCardMode(categoryKey, "setup")}
            disabled={busy}
            className="w-full py-2 rounded-md text-sm font-medium
                       border border-gray-300 text-gray-700
                       hover:bg-gray-50 disabled:opacity-50
                       disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            Set up new
          </button>
        </>
      )}

      {/* ── Multi-run target info ────────────────────── */}
      {isMultiRun && !allOneTimeDone && (
        <p className="text-xs text-gray-400">
          Run or load a one-time pipeline first.
        </p>
      )}
      {isMultiRun && allOneTimeDone && currentRunId && (
        <RunInfoBlock
          pipelineRunId={currentRunId}
          stepRunId={card.stepRunId}
          pipelineRunName={
            pastRuns.find((r) => r.run_id === currentRunId)?.run_name
            ?? currentRunId
          }
          mlflowInfo={mlflowInfo}
        />
      )}

      {/* ── "Load from previous run" dropdown ──────────── */}
      {!isMultiRun && cardMode === "load" && card.status !== "done" && (
        <select
          disabled={busy}
          onChange={(e) => {
            const runId = e.target.value;
            if (runId) {
              loadFromPreviousRun(runId, categoryKey);
            }
          }}
          className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md
                     bg-white text-gray-700 focus:outline-none focus:ring-2
                     focus:ring-blue-400 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <option value="">Select a previous run...</option>
          {pastRuns.map((run) => (
            <option key={run.run_id} value={run.run_id}>
              {run.run_name} ({run.status})
            </option>
          ))}
        </select>
      )}

      {/* ── Plugin list ("Set up new" mode or multi_run) ── */}
      {(isMultiRun || (cardMode === "setup" && card.status !== "done")) && (
        <div className="flex flex-col gap-0.5">
          {implementations.map((impl) => (
            <PluginRow
              key={impl.plugin_name}
              impl={impl}
              isSelected={card.selectedPlugin === impl.plugin_name}
              onSelect={() => selectPlugin(categoryKey, impl.plugin_name)}
              onToggleConfig={() => {
                // Select the plugin first if not already selected.
                if (card.selectedPlugin !== impl.plugin_name) {
                  selectPlugin(categoryKey, impl.plugin_name);
                }
                toggleConfig(categoryKey);
              }}
              disabled={busy}
            />
          ))}
        </div>
      )}

      {/* ── Parameter form (expandable) ──────────────── */}
      {(isMultiRun || (cardMode === "setup" && card.status !== "done")) &&
        card.configOpen && selectedImpl && selectedImpl.params.length > 0 && (
        <div className="border border-gray-200 rounded-md p-3 bg-gray-50">
          {selectedImpl.params.map((param) => (
            <ParamRow
              key={param.name}
              param={param}
              value={
                card.params[param.name] ??
                (param.default != null ? String(param.default) : "")
              }
              onChange={(val) => setParam(categoryKey, param.name, val)}
              context={context}
            />
          ))}
        </div>
      )}

      {/* ── View Results button (multi_run + done + has_visual_results) ── */}
      {isMultiRun &&
        card.status === "done" &&
        category_info.has_visual_results &&
        selectedImpl?.display != null &&
        card.stepRunId != null &&
        card.selectedPlugin != null &&
        context?.dataset_loading?.run_id != null && (
          <a
            href={`/results/${categoryKey}?${new URLSearchParams({
              stepRunId: card.stepRunId,
              datasetRunId: context.dataset_loading.run_id,
              pluginName: card.selectedPlugin,
              params: JSON.stringify(
                Object.fromEntries(
                  (selectedImpl?.params ?? []).map((p) => [
                    p.name,
                    card.params[p.name] ?? (p.default != null ? String(p.default) : ""),
                  ])
                )
              ),
            })}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 w-full py-2
                       rounded-md text-sm font-medium text-white
                       bg-green-500 hover:bg-green-600
                       transition-colors cursor-pointer"
          >
            <Eye className="h-4 w-4" />
            View Results
          </a>
        )}

      {/* ── Action button ────────────────────────────── */}
      <div className="mt-auto pt-2">
        {isMultiRun ? (
          <button
            disabled={!card.selectedPlugin || busy || !allOneTimeDone}
            onClick={() => onExecuteStep?.(categoryKey)}
            className="w-full py-2 rounded-md text-sm font-medium text-white
                       bg-blue-500 hover:bg-blue-600 disabled:opacity-50
                       disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            {card.status === "running" ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Running...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                ▷ Execute step
              </span>
            )}
          </button>
        ) : (cardMode === "setup" && card.status !== "done") ? (
          <button
            disabled={busy}
            onClick={() => onRunUpTo?.(categoryKey)}
            className="w-full py-2 rounded-md text-sm font-medium
                       border border-gray-300 text-gray-700
                       hover:bg-gray-50 disabled:opacity-50
                       disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            {card.status === "running" ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Running...
              </span>
            ) : (
              "Run up to this step"
            )}
          </button>
        ) : null}
      </div>
    </div>
  );
}
