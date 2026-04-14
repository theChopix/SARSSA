/**
 * LaunchModal — confirmation dialog before starting a pipeline run.
 *
 * Shown when the user clicks "Run up to this step" or "Run full pipeline".
 * Lets the user add key-value tags and a free-text description that will
 * be stored as MLflow tags (prefixed with `sarssa.`) on the parent run.
 *
 * Auto-suggested tags are generated from the selected plugin names and
 * any non-default parameter values.
 *
 * Visual structure:
 *
 *   ┌────────────────────────────────────────────┐
 *   │  Launch Pipeline Run                       │
 *   │                                            │
 *   │  Tags                          [+ Add tag] │
 *   │  ┌──────────┐  ┌──────────────┐   [×]      │
 *   │  │ key      │  │ value        │            │
 *   │  └──────────┘  └──────────────┘            │
 *   │  ...                                       │
 *   │                                            │
 *   │  Description                               │
 *   │  ┌────────────────────────────────────┐    │
 *   │  │                                    │    │
 *   │  └────────────────────────────────────┘    │
 *   │                                            │
 *   │            [ Cancel ]  [ Launch ]          │
 *   └────────────────────────────────────────────┘
 */

import { useState, useEffect, useCallback } from "react";
import { Plus, X } from "lucide-react";

import { usePipelineStore } from "../store/pipelineStore";
import type { StepDefinition } from "../types/pipeline";

/** A single key-value tag entry in the modal form. */
interface TagEntry {
  key: string;
  value: string;
}

/**
 * Build auto-suggested tags from the pending steps and the registry.
 *
 * Generates entries like:
 *   - `plugin.dataset_loading` → `movieLens_loader`
 *   - `param.epochs` → `100`
 */
function buildSuggestedTags(steps: StepDefinition[]): TagEntry[] {
  const tags: TagEntry[] = [];

  for (const step of steps) {
    const parts = step.plugin.split(".");
    const category = parts[0];
    // Use the second segment as the short plugin name.
    const shortName = parts.length >= 2 ? parts[1] : parts[0];
    tags.push({ key: `plugin.${category}`, value: shortName });
  }

  return tags;
}

/**
 * LaunchModal component.
 *
 * Reads `pendingSteps` from the store. When non-null, renders a
 * centered modal overlay. On confirm, calls `confirmLaunch` with
 * the tags dict and description string.
 */
export default function LaunchModal() {
  const pendingSteps = usePipelineStore((s) => s.pendingSteps);
  const setPendingSteps = usePipelineStore((s) => s.setPendingSteps);
  const confirmLaunch = usePipelineStore((s) => s.confirmLaunch);

  const [tags, setTags] = useState<TagEntry[]>([]);
  const [description, setDescription] = useState("");

  // Reset form state and populate auto-suggestions when modal opens.
  useEffect(() => {
    if (pendingSteps) {
      setTags(buildSuggestedTags(pendingSteps));
      setDescription("");
    }
  }, [pendingSteps]);

  const handleAddTag = useCallback(() => {
    setTags((prev) => [...prev, { key: "", value: "" }]);
  }, []);

  const handleRemoveTag = useCallback((index: number) => {
    setTags((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleTagChange = useCallback(
    (index: number, field: "key" | "value", val: string) => {
      setTags((prev) =>
        prev.map((entry, i) =>
          i === index ? { ...entry, [field]: val } : entry
        )
      );
    },
    []
  );

  const handleConfirm = useCallback(() => {
    // Convert tag entries to a Record, skipping empty keys.
    const tagsRecord: Record<string, string> = {};
    for (const entry of tags) {
      const trimmedKey = entry.key.trim();
      if (trimmedKey) {
        tagsRecord[trimmedKey] = entry.value.trim();
      }
    }
    confirmLaunch(tagsRecord, description.trim());
  }, [tags, description, confirmLaunch]);

  const handleCancel = useCallback(() => {
    setPendingSteps(null);
  }, [setPendingSteps]);

  // Close on Escape key.
  useEffect(() => {
    if (!pendingSteps) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleCancel();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [pendingSteps, handleCancel]);

  if (!pendingSteps) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={handleCancel}
    >
      <div
        className="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4 p-6"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Header ─────────────────────────────── */}
        <h2 className="text-lg font-bold text-gray-900 mb-4">
          Launch Pipeline Run
        </h2>

        {/* ── Instructions ──────────────────────── */}
        <p className="text-sm text-gray-500 mb-4">
          Add optional tags and a description for this run.
        </p>

        {/* ── Tags section ───────────────────────── */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-gray-700">Tags</label>
            <button
              type="button"
              onClick={handleAddTag}
              className="flex items-center gap-1 text-xs text-blue-500
                         hover:text-blue-700 transition-colors cursor-pointer"
            >
              <Plus className="h-3 w-3" />
              Add tag
            </button>
          </div>

          {tags.length === 0 && (
            <p className="text-xs text-gray-400 italic">
              No tags. Click "Add tag" to annotate this run.
            </p>
          )}

          <div className="space-y-2 max-h-48 overflow-y-auto pr-4">
            {tags.map((entry, i) => (
              <div key={i} className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder="key"
                  value={entry.key}
                  onChange={(e) => handleTagChange(i, "key", e.target.value)}
                  className="flex-1 rounded-md border border-gray-300 px-2 py-1.5
                             text-sm text-gray-900 placeholder-gray-400
                             focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                             outline-none"
                />
                <input
                  type="text"
                  placeholder="value"
                  value={entry.value}
                  onChange={(e) => handleTagChange(i, "value", e.target.value)}
                  className="flex-1 rounded-md border border-gray-300 px-2 py-1.5
                             text-sm text-gray-900 placeholder-gray-400
                             focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                             outline-none"
                />
                <button
                  type="button"
                  onClick={() => handleRemoveTag(i)}
                  className="shrink-0 text-gray-400 hover:text-red-500
                             transition-colors cursor-pointer"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* ── Description section ────────────────── */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Optional notes about this run..."
            rows={3}
            className="w-full rounded-md border border-gray-300 px-3 py-2
                       text-sm text-gray-900 placeholder-gray-400
                       focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                       outline-none resize-none"
          />
        </div>

        {/* ── Action buttons ─────────────────────── */}
        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={handleCancel}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       text-gray-700 bg-gray-100
                       hover:bg-gray-200 transition-colors cursor-pointer"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       text-white bg-blue-500
                       hover:bg-blue-600 transition-colors cursor-pointer"
          >
            Launch
          </button>
        </div>
      </div>
    </div>
  );
}
