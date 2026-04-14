/**
 * API service for pipeline-related endpoints.
 *
 * Contains fetch wrappers for:
 *   - `GET  /pipelines/runs`                       → list past runs
 *   - `GET  /pipelines/runs/{run_id}/context`      → fetch run context
 *   - `POST /pipelines/runs/{run_id}/execute-step`  → execute single step
 *   - `POST /pipelines/run-async`                  → start pipeline in background
 *   - `GET  /pipelines/tasks/{task_id}`            → poll task status
 *
 * ┌──────────────────────────────────────────────────────────┐
 * │  async / await refresher                                 │
 * │                                                          │
 * │  Every function here is `async` — it returns a Promise.  │
 * │  When you call it, you use `await`:                      │
 * │                                                          │
 * │    const runs = await fetchPipelineRuns();                │
 * │                                                          │
 * │  This pauses execution until the HTTP response arrives,  │
 * │  but does NOT block the browser — the UI stays           │
 * │  responsive while waiting.                               │
 * └──────────────────────────────────────────────────────────┘
 */

import { API_BASE_URL } from "../constants";
import type {
  ExecuteStepResponse,
  MlflowInfo,
  PipelineContext,
  PipelineRun,
  StepDefinition,
  TaskStatusResponse,
} from "../types/pipeline";

// ── GET /pipelines/runs ─────────────────────────────────

/**
 * Fetch the list of all past pipeline runs.
 *
 * @returns Array of run summaries sorted by the backend (newest first).
 *
 * @example
 * ```ts
 * const runs = await fetchPipelineRuns();
 * runs.forEach((r) => console.log(r.run_name, r.status));
 * ```
 */
export async function fetchPipelineRuns(): Promise<PipelineRun[]> {
  const response = await fetch(`${API_BASE_URL}/pipelines/runs`);

  if (!response.ok) {
    throw new Error(
      `Failed to fetch pipeline runs: ${response.status} ${response.statusText}`
    );
  }

  return (await response.json()) as PipelineRun[];
}

// ── GET /pipelines/runs/{run_id}/context ────────────────

/**
 * Fetch the context.json artifact from a specific pipeline run.
 *
 * The context maps each category that ran to its nested MLflow run ID.
 * This is used by "Load from previous run" to restore pipeline state.
 *
 * @param runId - The MLflow run ID of the parent pipeline run.
 * @returns The context dictionary.
 *
 * @throws {Error} 404 if the run has no context artifact.
 */
export async function fetchRunContext(
  runId: string
): Promise<PipelineContext> {
  const response = await fetch(
    `${API_BASE_URL}/pipelines/runs/${runId}/context`
  );

  if (!response.ok) {
    throw new Error(
      `Failed to fetch run context: ${response.status} ${response.statusText}`
    );
  }

  return (await response.json()) as PipelineContext;
}

// ── POST /pipelines/runs/{run_id}/execute-step ──────────

/**
 * Execute a single plugin step on an existing pipeline run.
 *
 * Used for "multi_run" plugins (Inspection, Steering, etc.) that
 * can be triggered individually after the main pipeline finishes.
 *
 * @param runId - The parent pipeline run to attach to.
 * @param step  - Which plugin to run and with what parameters.
 * @returns The category and the new nested run ID.
 *
 * @example
 * ```ts
 * const result = await executeStep("abc123", {
 *   plugin: "steering.sae_steering.sae_steering",
 *   params: { alpha: 0.5 },
 * });
 * console.log(result.step_run_id); // "def456"
 * ```
 */
export async function executeStep(
  runId: string,
  step: StepDefinition
): Promise<ExecuteStepResponse> {
  const response = await fetch(
    `${API_BASE_URL}/pipelines/runs/${runId}/execute-step`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(step),
    }
  );

  if (!response.ok) {
    throw new Error(
      `Failed to execute step: ${response.status} ${response.statusText}`
    );
  }

  return (await response.json()) as ExecuteStepResponse;
}

// ── POST /pipelines/run-async ────────────────────────────

/**
 * Start a pipeline execution in the background.
 *
 * The backend spawns a worker thread and returns immediately
 * with a task ID. Use {@link getTaskStatus} to poll for progress.
 *
 * @param steps       - Array of steps to execute in order.
 * @param context     - Optional pre-populated context from a previous run.
 * @param tags        - Optional user-defined key-value tags for the MLflow run.
 * @param description - Optional free-text description for the MLflow run.
 * @returns Object containing the `task_id` to poll.
 *
 * @example
 * ```ts
 * const { task_id } = await startPipelineTask(
 *   [{ plugin: "dataset_loading.movieLens_loader.movieLens_loader", params: {} }],
 *   { dataset_loading: { run_id: "abc123" } },
 *   { dataset: "MovieLens", model: "ELSA" },
 *   "Baseline run with default params"
 * );
 * ```
 */
export async function startPipelineTask(
  steps: StepDefinition[],
  context: Record<string, unknown> = {},
  tags: Record<string, string> = {},
  description: string = ""
): Promise<{ task_id: string }> {
  const response = await fetch(`${API_BASE_URL}/pipelines/run-async`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ steps, context, tags, description }),
  });

  if (!response.ok) {
    throw new Error(
      `Failed to start pipeline task: ${response.status} ${response.statusText}`
    );
  }

  return (await response.json()) as { task_id: string };
}

// ── GET /pipelines/tasks/{task_id} ──────────────────────

/**
 * Poll the current status of a background pipeline task.
 *
 * Returns immediately with the task's current state. The caller
 * should poll this endpoint at a regular interval (e.g. every 2s)
 * until `status` is `"completed"` or `"error"`.
 *
 * @param taskId - The task ID returned by {@link startPipelineTask}.
 * @returns Current task status including progress and results.
 *
 * @example
 * ```ts
 * const status = await getTaskStatus("abc123");
 * if (status.status === "completed") {
 *   console.log("Done!", status.context);
 * }
 * ```
 */
export async function getTaskStatus(
  taskId: string
): Promise<TaskStatusResponse> {
  const response = await fetch(
    `${API_BASE_URL}/pipelines/tasks/${taskId}`
  );

  if (!response.ok) {
    throw new Error(
      `Failed to fetch task status: ${response.status} ${response.statusText}`
    );
  }

  return (await response.json()) as TaskStatusResponse;
}

// ── GET /pipelines/mlflow-info ───────────────────────────

/**
 * Fetch MLflow UI connection info for constructing deep links.
 *
 * Returns the MLflow UI base URL and the numeric experiment ID
 * resolved from the configured experiment name.
 *
 * @returns MLflow UI base URL and experiment ID.
 *
 * @example
 * ```ts
 * const info = await fetchMlflowInfo();
 * console.log(info.ui_base_url);    // "http://localhost:5000"
 * console.log(info.experiment_id);  // "1"
 * ```
 */
export async function fetchMlflowInfo(): Promise<MlflowInfo> {
  const response = await fetch(`${API_BASE_URL}/pipelines/mlflow-info`);

  if (!response.ok) {
    throw new Error(
      `Failed to fetch MLflow info: ${response.status} ${response.statusText}`
    );
  }

  return (await response.json()) as MlflowInfo;
}
