/**
 * ResultsPage — displays visual enrichment results for a completed step.
 *
 * Opened in a **new browser tab** from the "View Results" button on
 * a completed multi_run PipelineCard. Because each tab has its own
 * Zustand store instance, this page is self-sufficient:
 *
 *   1. Reads `categoryKey` from the URL path.
 *   2. Reads `stepRunId`, `datasetRunId`, and `pluginName` from
 *      query parameters.
 *   3. Loads the plugin registry on mount to derive the `DisplaySpec`.
 *   4. Renders the `VisualResultsPanel` with the resolved props.
 *
 * URL format:
 *   /results/:categoryKey?stepRunId=...&datasetRunId=...&pluginName=...
 */

import { useEffect, useMemo } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { Loader2, AlertCircle } from "lucide-react";

import { ArtifactPanel } from "../components/ArtifactPanel";
import { VisualResultsPanel } from "../components/VisualResultsPanel";
import { usePipelineStore } from "../store/pipelineStore";
import type { DisplaySpec } from "../types/plugin";

export function ResultsPage() {
  // ── URL params ──────────────────────────────────────────
  const { categoryKey } = useParams<{ categoryKey: string }>();
  const [searchParams] = useSearchParams();

  const stepRunId = searchParams.get("stepRunId");
  const datasetRunId = searchParams.get("datasetRunId");
  const pluginName = searchParams.get("pluginName");

  // ── Plugin input params (serialised by PipelineCard) ───
  const pluginParams: Record<string, string> | null = useMemo(() => {
    const raw = searchParams.get("params");
    if (!raw) return null;
    try {
      return JSON.parse(raw) as Record<string, string>;
    } catch {
      return null;
    }
  }, [searchParams]);

  // ── Load registry (new tab = empty store) ───────────────
  const registry = usePipelineStore((s) => s.registry);
  const loadRegistry = usePipelineStore((s) => s.loadRegistry);

  useEffect(() => {
    if (!registry) {
      loadRegistry();
    }
  }, [registry, loadRegistry]);

  // ── Derive DisplaySpec from registry ────────────────────
  const displaySpec: DisplaySpec | null = useMemo(() => {
    if (!registry || !categoryKey || !pluginName) return null;

    const entry = registry[categoryKey];
    if (!entry) return null;

    const impl = entry.implementations.find(
      (i) => i.plugin_name === pluginName
    );

    return impl?.display ?? null;
  }, [registry, categoryKey, pluginName]);

  // Category display name for the page title.
  const categoryDisplayName = useMemo(() => {
    if (!registry || !categoryKey) return categoryKey ?? "Unknown";
    return registry[categoryKey]?.category_info.display_name ?? categoryKey;
  }, [registry, categoryKey]);

  // ── Missing query params ────────────────────────────────
  if (!stepRunId || !datasetRunId || !pluginName || !categoryKey) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="flex items-center gap-2 text-red-500">
          <AlertCircle className="h-5 w-5" />
          <p className="text-sm">
            Missing required parameters. Please open this page from a
            completed pipeline card.
          </p>
        </div>
      </div>
    );
  }

  // ── Loading registry ────────────────────────────────────
  if (!registry) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
      </div>
    );
  }

  // ── DisplaySpec not found ───────────────────────────────
  if (!displaySpec) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="flex items-center gap-2 text-amber-500">
          <AlertCircle className="h-5 w-5" />
          <p className="text-sm">
            No display specification found for plugin "{pluginName}"
            in category "{categoryKey}".
          </p>
        </div>
      </div>
    );
  }

  // ── Render results ──────────────────────────────────────
  return (
    <div className="flex-1 px-8 py-8">
      <h2 className="text-xl font-bold text-gray-900">
        {categoryDisplayName} — Results
      </h2>

      {pluginParams && Object.keys(pluginParams).length > 0 && (
        <p className="mt-1 text-sm text-gray-500 font-mono">
          [{" "}
          {Object.entries(pluginParams).map(([key, value], idx, arr) => (
            <span key={key}>
              <span className="font-semibold text-gray-600">{key}</span>
              {": "}
              <span>{value}</span>
              {idx < arr.length - 1 && ",  "}
            </span>
          ))}
          {" ]"}
        </p>
      )}

      {displaySpec.type === "item_rows" && (
        <VisualResultsPanel
          displaySpec={displaySpec}
          stepRunId={stepRunId}
          datasetRunId={datasetRunId}
        />
      )}

      {displaySpec.type === "artifact" && (
        <ArtifactPanel
          files={displaySpec.files}
          stepRunId={stepRunId}
        />
      )}
    </div>
  );
}
