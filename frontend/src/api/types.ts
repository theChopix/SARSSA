// --- Plugin Registry types (GET /plugins/registry) ---

export interface PluginParam {
  name: string;
  type: "int" | "float" | "str" | "bool";
  default: string | number | boolean | null;
}

export interface PluginImplementation {
  plugin_path: string;
  display_name: string;
  group: string | null;
  params: PluginParam[];
}

export type CategoryType = "one_time" | "multi_run";

export interface PluginCategory {
  name: string;
  display_name: string;
  type: CategoryType;
  order: number;
  implementations: PluginImplementation[];
}

export interface PluginRegistry {
  categories: PluginCategory[];
}

// --- Pipeline Runs types (GET /pipelines/runs) ---

export interface PipelineStep {
  run_id: string;
  plugin_name: string;
  category: string;
  status: string;
}

export interface PipelineRun {
  run_id: string;
  run_name: string;
  start_time: number;
  end_time: number | null;
  status: string;
  context: Record<string, { run_id: string }> | null;
  steps: PipelineStep[];
}

// --- Pipeline Execution types (POST /pipelines/run) ---

export interface StepDefinition {
  plugin: string;
  params?: Record<string, unknown>;
}

export interface PipelineRequest {
  steps: StepDefinition[];
}

export interface PipelineRunResponse {
  message: string;
  result: Record<string, { run_id: string }>;
}

// --- Execute Step types (POST /pipelines/{run_id}/execute-step) ---

export interface ExecuteStepResponse {
  message: string;
  result: Record<string, { run_id: string }>;
}
