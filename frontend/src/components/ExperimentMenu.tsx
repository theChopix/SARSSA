/**
 * ExperimentMenu — header dropdown selecting the MLflow experiment new
 * runs log to, with a "Create new experiment…" modal. The selection
 * persists in localStorage and is locked while a run is in flight.
 */

import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, ChevronUp, FlaskConical, Plus } from "lucide-react";
import { toast } from "sonner";

import { usePipelineStore } from "../store/pipelineStore";

export function ExperimentMenu() {
  const experiments = usePipelineStore((s) => s.experiments);
  const selectedExperiment = usePipelineStore((s) => s.selectedExperiment);
  const loadExperiments = usePipelineStore((s) => s.loadExperiments);
  const selectExperiment = usePipelineStore((s) => s.selectExperiment);
  const addExperiment = usePipelineStore((s) => s.addExperiment);
  const pipelineRunning = usePipelineStore((s) => s.pipelineRunning);
  const anyStepRunning = usePipelineStore((s) =>
    Object.values(s.cards).some((c) => c.status === "running")
  );
  const busy = pipelineRunning || anyStepRunning;

  const [open, setOpen] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadExperiments();
  }, [loadExperiments]);

  // Close the dropdown on an outside click.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [open]);

  // Close the modal on Escape.
  useEffect(() => {
    if (!modalOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setModalOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [modalOpen]);

  const shared = experiments.find((e) => e.shared);
  // "" means the shared experiment — resolve the display name.
  const currentName = selectedExperiment || shared?.name || "experiment";

  const handleSelect = async (name: string, isShared: boolean) => {
    setOpen(false);
    // The shared experiment is stored as "" (the default).
    await selectExperiment(isShared ? "" : name);
  };

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      await addExperiment(name);
      setModalOpen(false);
      setNewName("");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to create experiment"
      );
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="relative" ref={containerRef}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        disabled={busy}
        title={
          busy
            ? "Experiment can't be switched while a run is in progress"
            : `New runs log to "${currentName}"`
        }
        className="flex items-center gap-1.5 rounded-full border border-gray-300 bg-gray-50
                   px-3 py-1 text-sm font-medium text-gray-700 hover:bg-gray-100
                   disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <FlaskConical className="h-4 w-4 text-blue-500" />
        <span className="max-w-48 truncate">{currentName}</span>
        {open ? (
          <ChevronUp className="h-4 w-4" />
        ) : (
          <ChevronDown className="h-4 w-4" />
        )}
      </button>

      {open && (
        <div className="absolute right-0 z-20 mt-2 w-80 rounded-lg border border-gray-200 bg-white shadow-lg">
          <p className="border-b border-gray-100 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            Experiments
          </p>
          <ul className="max-h-80 overflow-y-auto py-1">
            {experiments.map((experiment) => {
              const isSelected = experiment.shared
                ? selectedExperiment === ""
                : selectedExperiment === experiment.name;
              return (
                <li key={experiment.experiment_id}>
                  <button
                    type="button"
                    onClick={() => handleSelect(experiment.name, experiment.shared)}
                    className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-gray-50"
                  >
                    <Check
                      className={`h-4 w-4 shrink-0 ${
                        isSelected ? "text-blue-500" : "text-transparent"
                      }`}
                    />
                    <span
                      className="min-w-0 flex-1 truncate text-sm text-gray-800"
                      title={experiment.name}
                    >
                      {experiment.name}
                      {experiment.shared && (
                        <span className="ml-1.5 text-xs text-gray-400">
                          (shared)
                        </span>
                      )}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
          <button
            type="button"
            onClick={() => {
              setOpen(false);
              setModalOpen(true);
            }}
            className="flex w-full items-center gap-2 border-t border-gray-100 px-3 py-2
                       text-left text-sm text-blue-500 hover:bg-gray-50 hover:text-blue-700"
          >
            <Plus className="h-4 w-4" />
            Create new experiment…
          </button>
        </div>
      )}

      {modalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
          onClick={() => setModalOpen(false)}
        >
          <div
            className="bg-white rounded-xl shadow-xl w-full max-w-md mx-4 p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-bold text-gray-900 mb-4">
              Create new experiment
            </h2>
            <input
              type="text"
              autoFocus
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleCreate();
              }}
              maxLength={100}
              placeholder="e.g. my_experiments"
              className="w-full rounded-md border border-gray-300 px-3 py-2
                         text-sm text-gray-900 placeholder-gray-400
                         focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                         outline-none"
            />
            <p className="mt-2 text-xs text-gray-500">
              New pipeline runs will log to this MLflow experiment. Shared
              baseline runs stay available in "Load from previous run".
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setModalOpen(false)}
                className="rounded-md border border-gray-300 px-4 py-2 text-sm
                           font-medium text-gray-700 hover:bg-gray-50 cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleCreate}
                disabled={!newName.trim() || creating}
                className="rounded-md bg-blue-500 px-4 py-2 text-sm font-medium
                           text-white hover:bg-blue-600 disabled:opacity-50
                           disabled:cursor-not-allowed cursor-pointer"
              >
                {creating ? "Creating…" : "Create"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
