/**
 * API service for pipeline-related endpoints.
 *
 * Contains fetch wrappers for:
 *   - `GET  /pipelines/runs`                    → list past runs
 *   - `GET  /pipelines/runs/{run_id}/context`   → fetch run context
 *   - `POST /pipelines/runs/{run_id}/execute-step` → execute single step
 *   - `POST /pipelines/run`                     → legacy batch run
 *
 * The SSE streaming endpoint (`POST /pipelines/run-stream`) is handled
 * separately via `subscribeToPipelineStream()` because it uses
 * EventSource-style streaming, not a simple request/response.
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
  PipelineContext,
  PipelineRun,
  StepDefinition,
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

// ── POST /pipelines/run-stream (SSE) ────────────────────

/**
 * Callbacks for SSE events emitted during a pipeline stream.
 *
 * Each callback is optional — you only need to provide the ones
 * you care about. The store will wire these up to update UI state.
 *
 * ┌────────────────────────────────────────────────────────┐
 * │  What is a callback?                                   │
 * │                                                        │
 * │  A callback is a function you pass to another function │
 * │  so it can "call you back" when something happens.     │
 * │  Think of it like: "when X event fires, run my code."  │
 * └────────────────────────────────────────────────────────┘
 */
export interface PipelineStreamCallbacks {
  /** Called when the parent MLflow run is created. */
  onRunStarted?: (data: { run_id: string }) => void;
  /** Called when a plugin step begins executing. */
  onStepStarted?: (data: { category: string; plugin: string }) => void;
  /** Called when a plugin step finishes successfully. */
  onStepCompleted?: (data: { category: string; run_id: string }) => void;
  /** Called when all steps are done and context is saved. */
  onRunCompleted?: (data: {
    run_id: string;
    context: PipelineContext;
  }) => void;
  /** Called if the stream encounters an error. */
  onError?: (error: Error) => void;
}

/**
 * Start a pipeline execution with real-time SSE streaming.
 *
 * This function opens a POST request to `/pipelines/run-stream` and
 * reads the response body as a **stream** of Server-Sent Events.
 *
 * Unlike the other functions in this file (which return a single
 * value), this one takes **callbacks** — functions that get called
 * each time an event arrives from the server.
 *
 * ┌────────────────────────────────────────────────────────┐
 * │  Why not just `await` the whole response?              │
 * │                                                        │
 * │  Because the pipeline takes minutes. We want to show   │
 * │  progress as each step completes, not wait until the   │
 * │  end. SSE gives us a live stream of events.            │
 * │                                                        │
 * │  The browser receives chunks of text as they arrive:   │
 * │    event: step_started\n                               │
 * │    data: {"category":"training_cfm"}\n\n               │
 * │                                                        │
 * │  We parse each chunk and call the matching callback.   │
 * └────────────────────────────────────────────────────────┘
 *
 * @param steps     - Array of steps to execute in order.
 * @param callbacks - Functions to call as events arrive.
 *
 * @example
 * ```ts
 * subscribeToPipelineStream(
 *   [{ plugin: "dataset_loading.lastFm1k_loader.lastFm1k_loader", params: {} }],
 *   {
 *     onStepStarted: (d) => console.log("Started:", d.category),
 *     onStepCompleted: (d) => console.log("Done:", d.category),
 *   }
 * );
 * ```
 */
export async function subscribeToPipelineStream(
  steps: StepDefinition[],
  callbacks: PipelineStreamCallbacks
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/pipelines/run-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ steps }),
  });

  if (!response.ok) {
    const message = `Pipeline stream failed: ${response.status} ${response.statusText}`;
    callbacks.onError?.(new Error(message));
    return;
  }

  // The response body is a ReadableStream of text chunks.
  // We read it line by line, parsing SSE events as they arrive.
  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError?.(new Error("No response body"));
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "";

  try {
    while (true) {
      // `read()` waits for the next chunk of data from the server.
      // `done` is true when the stream closes (pipeline finished).
      const { done, value } = await reader.read();
      if (done) break;

      // Decode the binary chunk to a string and append to buffer.
      buffer += decoder.decode(value, { stream: true });

      // SSE protocol: events are separated by blank lines (\n\n).
      // Each event has optional "event:" and "data:" lines.
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith("event:")) {
          // Remember the event type for the next data line.
          currentEvent = line.slice("event:".length).trim();
        } else if (line.startsWith("data:")) {
          const rawData = line.slice("data:".length).trim();
          if (!rawData) continue;

          try {
            const data = JSON.parse(rawData);
            // Dispatch to the correct callback based on event type.
            switch (currentEvent) {
              case "run_started":
                callbacks.onRunStarted?.(data);
                break;
              case "step_started":
                callbacks.onStepStarted?.(data);
                break;
              case "step_completed":
                callbacks.onStepCompleted?.(data);
                break;
              case "run_completed":
                callbacks.onRunCompleted?.(data);
                break;
            }
          } catch {
            // Ignore malformed JSON lines.
          }
          currentEvent = "";
        }
      }
    }
  } catch (error) {
    callbacks.onError?.(
      error instanceof Error ? error : new Error(String(error))
    );
  }
}
