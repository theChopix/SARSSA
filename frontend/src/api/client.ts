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
