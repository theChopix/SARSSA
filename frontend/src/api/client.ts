import type {
  PluginRegistry,
  PipelineRun,
  PipelineRequest,
  PipelineRunResponse,
  StepDefinition,
  ExecuteStepResponse,
} from "./types";

const API_BASE = "http://localhost:8000";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

/** GET /plugins/registry */
export function fetchPluginRegistry(): Promise<PluginRegistry> {
  return request<PluginRegistry>("/plugins/registry");
}

/** GET /pipelines/runs */
export function fetchPipelineRuns(): Promise<PipelineRun[]> {
  return request<PipelineRun[]>("/pipelines/runs");
}

/** POST /pipelines/run */
export function runPipeline(
  context: Record<string, unknown>,
  pipelineRequest: PipelineRequest,
): Promise<PipelineRunResponse> {
  return request<PipelineRunResponse>("/pipelines/run", {
    method: "POST",
    body: JSON.stringify({
      context,
      pipeline_request: pipelineRequest,
    }),
  });
}

/** SSE event from /pipelines/run-stream */
export interface SSEEvent {
  event: string;
  data: Record<string, unknown>;
}

/** POST /pipelines/run-stream — returns an async iterator of SSE events */
export async function* runPipelineStream(
  context: Record<string, unknown>,
  pipelineRequest: PipelineRequest,
): AsyncGenerator<SSEEvent> {
  const res = await fetch(`${API_BASE}/pipelines/run-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      context,
      pipeline_request: pipelineRequest,
    }),
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Parse SSE messages from the buffer
    const parts = buffer.split("\n\n");
    buffer = parts.pop()!; // keep incomplete chunk

    for (const part of parts) {
      let eventName = "message";
      let dataStr = "";
      for (const line of part.split("\n")) {
        if (line.startsWith("event: ")) {
          eventName = line.slice(7);
        } else if (line.startsWith("data: ")) {
          dataStr = line.slice(6);
        }
      }
      if (dataStr) {
        yield { event: eventName, data: JSON.parse(dataStr) };
      }
    }
  }
}

/** POST /pipelines/{runId}/execute-step */
export function executeStep(
  runId: string,
  step: StepDefinition,
): Promise<ExecuteStepResponse> {
  return request<ExecuteStepResponse>(
    `/pipelines/${runId}/execute-step`,
    {
      method: "POST",
      body: JSON.stringify(step),
    },
  );
}
