/**
 * RunningTasksMenu — a header pill listing in-flight pipeline runs.
 *
 * Polls `GET /pipelines/tasks` on a coarse interval (a lightweight list
 * poll, deliberately slower than the active run's 2s detailed poller)
 * and renders a count badge + dropdown. Selecting a row loads that run
 * into the main view via `store.loadRunningTask`. The pill hides itself
 * when nothing is running.
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronDown, ChevronUp, Clock, Loader2 } from "lucide-react";

import { usePipelineStore } from "../store/pipelineStore";
import type { TaskSummary } from "../types/pipeline";

/** How often to refresh the running-tasks list, in milliseconds. */
const POLL_INTERVAL_MS = 3000;

/** Row label: the user's pipeline name, else a short run id, else generic. */
function taskLabel(task: TaskSummary): string {
  if (task.pipeline_name) return task.pipeline_name;
  if (task.run_id) return `Run ${task.run_id.slice(0, 8)}`;
  return "Pipeline run";
}

export function RunningTasksMenu() {
  const runningTasks = usePipelineStore((s) => s.runningTasks);
  const loadRunningTasks = usePipelineStore((s) => s.loadRunningTasks);
  const loadRunningTask = usePipelineStore((s) => s.loadRunningTask);
  const currentTaskId = usePipelineStore((s) => s.currentTaskId);
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Poll the running-tasks list while mounted.
  useEffect(() => {
    loadRunningTasks();
    const id = setInterval(loadRunningTasks, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [loadRunningTasks]);

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

  if (runningTasks.length === 0) return null;

  const handleSelect = (task: TaskSummary) => {
    setOpen(false);
    navigate("/");
    loadRunningTask(task);
  };

  return (
    <div className="relative" ref={containerRef}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-sm font-medium text-blue-700 hover:bg-blue-100"
      >
        <Loader2 className="h-4 w-4 animate-spin" />
        Running ({runningTasks.length})
        {open ? (
          <ChevronUp className="h-4 w-4" />
        ) : (
          <ChevronDown className="h-4 w-4" />
        )}
      </button>

      {open && (
        <div className="absolute left-0 z-20 mt-2 w-96 rounded-lg border border-gray-200 bg-white shadow-lg">
          <p className="border-b border-gray-100 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            Running tasks
          </p>
          <ul className="max-h-80 overflow-y-auto py-1">
            {runningTasks.map((task) => (
              <li key={task.task_id}>
                <button
                  type="button"
                  onClick={() => handleSelect(task)}
                  className="flex w-full items-start gap-2 px-3 py-2 text-left hover:bg-gray-50"
                >
                  {task.status === "queued" ? (
                    <Clock className="mt-0.5 h-4 w-4 shrink-0 text-gray-400" />
                  ) : (
                    <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-blue-500" />
                  )}
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center justify-between gap-2">
                      <span
                        className="truncate text-sm font-medium text-gray-800"
                        title={taskLabel(task)}
                      >
                        {taskLabel(task)}
                        {task.task_id === currentTaskId && (
                          <span className="ml-1.5 text-xs font-normal text-blue-500">
                            (open)
                          </span>
                        )}
                      </span>
                      <span className="shrink-0 text-xs text-gray-500">
                        {task.status === "queued"
                          ? "queued"
                          : `step ${task.current_step_index + 1} / ${task.total_steps}`}
                      </span>
                    </span>
                    {task.current_step && (
                      <span className="block truncate text-xs text-gray-500">
                        {task.current_step}
                      </span>
                    )}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
