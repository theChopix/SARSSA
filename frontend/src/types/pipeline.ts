/**
 * TypeScript types for pipeline runs and execution.
 *
 * These mirror the backend models and API response shapes for
 * pipeline-related endpoints.
 */

// ── Pipeline run (from GET /pipelines/runs) ─────────────

/**
 * Summary of a single pipeline run returned by the backend.
 *
 * - `run_id`    – Unique MLflow run identifier.
 * - `run_name`  – Human-readable name (e.g. "pipeline_run_2025-04-07_20-30-00").
 * - `status`    – MLflow run status: "FINISHED", "RUNNING", "FAILED", etc.
 * - `start_time`– Unix timestamp (milliseconds) when the run started.
 */
export interface PipelineRun {
  run_id: string;
  run_name: string;
  status: string;
  start_time: number;
}

// ── Pipeline context (from GET /pipelines/runs/{id}/context) ──

/**
 * The context dictionary saved as `context.json` in each pipeline run.
 *
 * Maps category key → an object with at least `run_id`.
 * Example: `{ "dataset_loading": { "run_id": "abc123" } }`
 */
export type PipelineContext = Record<string, { run_id: string }>;

// ── Step definition (sent in POST bodies) ───────────────

/**
 * Defines a single plugin step to execute.
 *
 * Sent to `POST /pipelines/run-async` (as part of an array)
 * or `POST /pipelines/runs/{run_id}/execute-step` (single step).
 *
 * - `plugin` – Dotted module path (e.g. "steering.single.sae_steering.sae_steering").
 * - `params` – Key-value pairs forwarded to the plugin's `run()` method.
 */
export interface StepDefinition {
  plugin: string;
  params: Record<string, unknown>;
}

// ── Task status (from GET /pipelines/tasks/{task_id}) ───

/**
 * Response from `GET /pipelines/tasks/{task_id}`.
 *
 * Returned by the polling endpoint to track background pipeline progress.
 *
 * - `status`             – One of `"running"`, `"completed"`, `"error"`, `"cancelled"`.
 * - `current_step`       – Category key of the step currently executing.
 * - `current_step_index` – 0-based index into the requested steps.
 * - `total_steps`        – Total number of steps in the pipeline.
 * - `completed_steps`    – Steps that have finished so far.
 * - `context`            – Final pipeline context (set on completion).
 * - `error`              – Error message (set on failure).
 * - `messages`           – Ordered list of notification dicts pushed by
 *                          the executing plugin via `PluginNotifier`.
 *                          Each entry has `timestamp`, `level`, and `text`.
 */
export interface TaskStatusResponse {
  task_id: string;
  status: "running" | "completed" | "error" | "cancelled";
  run_id: string | null;
  current_step: string | null;
  current_step_index: number;
  total_steps: number;
  completed_steps: { category: string; run_id: string }[];
  context: PipelineContext | null;
  error: string | null;
  messages: { timestamp: number; level: string; text: string }[];
}

// ── Running-task summary (from GET /pipelines/tasks) ────

/**
 * Compact summary of an active (running) pipeline task.
 *
 * Mirrors the backend `TaskSummary` model and powers the header
 * "running tasks" menu. `steps_requested` lets the frontend rebuild a
 * run's card layout when loading a task it did not start itself;
 * `initial_context` carries the upstream steps that were loaded from a
 * previous run (and so are absent from `steps_requested`), letting a new
 * tab restore those cards as completed too.
 */
export interface TaskSummary {
  task_id: string;
  run_id: string | null;
  pipeline_name: string;
  status: "running" | "completed" | "error" | "cancelled";
  current_step: string | null;
  current_step_index: number;
  total_steps: number;
  steps_requested: StepDefinition[];
  initial_context: PipelineContext;
}

// ── Execute-step response ───────────────────────────────

/**
 * Response from `POST /pipelines/runs/{run_id}/execute-step`.
 *
 * - `category`    – Which category the executed plugin belongs to.
 * - `step_run_id` – MLflow run ID of the newly created nested run.
 */
export interface ExecuteStepResponse {
  category: string;
  step_run_id: string;
}

// ── Execute-step-async response ────────────────────────

/**
 * Response from `POST /pipelines/runs/{run_id}/execute-step-async`.
 *
 * The backend spawns a worker thread and returns immediately.
 * Poll `GET /pipelines/tasks/{task_id}` until the status is terminal.
 *
 * - `task_id` – Opaque ID to pass to {@link TaskStatusResponse} polling.
 */
export interface ExecuteStepAsyncResponse {
  task_id: string;
}

// ── MLflow info (from GET /pipelines/mlflow-info) ────────

/**
 * Response from `GET /pipelines/mlflow-info`.
 *
 * Provides the MLflow UI base URL and experiment ID needed to
 * construct deep links to experiment and run pages.
 *
 * - `ui_base_url`    – Base URL of the MLflow UI server (e.g. "http://localhost:5000").
 * - `experiment_id`  – Numeric experiment ID assigned by MLflow.
 */
export interface MlflowInfo {
  ui_base_url: string;
  experiment_id: string;
}
