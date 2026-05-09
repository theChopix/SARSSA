/**
 * VisualResultsPanel — renders rows of ItemCards driven by a DisplaySpec.
 *
 * This panel is shown inside a PipelineCard after a steering or
 * inspection step completes. It reads the `DisplaySpec` from the
 * plugin registry to know which artifact keys to fetch and how to
 * label each row.
 *
 * For each row in `displaySpec.rows`:
 *   1. Fetches the artifact JSON (list of item IDs) from the step run.
 *   2. Calls `fetchEnrichedItems(datasetRunId, itemIds)`.
 *   3. Renders a labeled, horizontally scrollable row of ItemCards.
 *
 * ┌─────────────────────────────────────────────────────────┐
 * │  ── Interaction History ──────────────────────────────── │
 * │  [Card] [Card] [Card] [Card] [Card] →                  │
 * │                                                         │
 * │  ── Original Recommendations ────────────────────────── │
 * │  [Card] [Card] [Card] [Card] [Card] →                  │
 * │                                                         │
 * │  ── Steered Recommendations ─────────────────────────── │
 * │  [Card] [Card] [Card] [Card] [Card] →                  │
 * └─────────────────────────────────────────────────────────┘
 */

import { useEffect, useState } from "react";
import { Loader2, AlertCircle, ChevronDown } from "lucide-react";

import { fetchStepArtifact, fetchEnrichedItems } from "../api/items";
import { ItemCard } from "./ItemCard";
import type { ItemRowsDisplaySpec, DisplayRowSpec } from "../types/plugin";
import type { EnrichedItem } from "../types/items";

// ── Props ────────────────────────────────────────────────

interface VisualResultsPanelProps {
  displaySpec: ItemRowsDisplaySpec;
  stepRunId: string;
  datasetRunId: string;
}

// ── Per-row state ────────────────────────────────────────

interface RowState {
  items: EnrichedItem[];
  loading: boolean;
  error: string | null;
}

// ── Single row component ─────────────────────────────────

/**
 * Fetches item IDs from the step artifact, enriches them, and
 * renders a horizontal scrollable row of ItemCards.
 */
function ItemRow({
  row,
  stepRunId,
  datasetRunId,
}: {
  row: DisplayRowSpec;
  stepRunId: string;
  datasetRunId: string;
}) {
  const [state, setState] = useState<RowState>({
    items: [],
    loading: true,
    error: null,
  });
  const [expanded, setExpanded] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        // 1. Fetch the artifact (list of item IDs)
        const artifact = await fetchStepArtifact(
          stepRunId,
          `${row.key}.json`
        );

        if (cancelled) return;

        // The artifact should be an array of item IDs
        const itemIds = Array.isArray(artifact)
          ? artifact.map(String)
          : [];

        if (itemIds.length === 0) {
          setState({ items: [], loading: false, error: null });
          return;
        }

        // 2. Enrich the item IDs with metadata
        const { items } = await fetchEnrichedItems(datasetRunId, itemIds);

        if (cancelled) return;
        setState({ items, loading: false, error: null });
      } catch (err) {
        if (cancelled) return;
        setState({
          items: [],
          loading: false,
          error: err instanceof Error ? err.message : "Unknown error",
        });
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [row.key, stepRunId, datasetRunId]);

  return (
    <div className="space-y-2">
      {/* Collapsible row header */}
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center gap-2 text-left text-sm font-medium text-gray-700 hover:text-gray-900"
        aria-expanded={expanded}
      >
        <ChevronDown
          className={`h-4 w-4 text-gray-400 transition-transform ${
            expanded ? "" : "-rotate-90"
          }`}
        />
        <h4>{row.label}</h4>
      </button>

      {expanded && (
        <>
          {/* Loading skeleton */}
          {state.loading && (
            <div className="flex items-center gap-2 py-4 text-gray-400">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-xs">Loading items…</span>
            </div>
          )}

          {/* Error state */}
          {state.error && (
            <div className="flex items-center gap-2 py-2 text-red-500">
              <AlertCircle className="h-4 w-4" />
              <span className="text-xs">{state.error}</span>
            </div>
          )}

          {/* Item cards — horizontal scroll */}
          {!state.loading && !state.error && state.items.length > 0 && (
            <div className="flex gap-3 overflow-x-auto pb-2">
              {state.items.map((item) => (
                <ItemCard key={item.id} item={item} />
              ))}
            </div>
          )}

          {/* Empty state */}
          {!state.loading && !state.error && state.items.length === 0 && (
            <p className="text-xs text-gray-400 py-2">No items to display.</p>
          )}
        </>
      )}
    </div>
  );
}

// ── Main panel ───────────────────────────────────────────

export function VisualResultsPanel({
  displaySpec,
  stepRunId,
  datasetRunId,
}: VisualResultsPanelProps) {
  return (
    <div className="mt-4 space-y-4 border-t border-gray-100 pt-4">
      {displaySpec.rows.map((row) => (
        <ItemRow
          key={row.key}
          row={row}
          stepRunId={stepRunId}
          datasetRunId={datasetRunId}
        />
      ))}
    </div>
  );
}
