/**
 * TypeScript types for the plugin registry.
 *
 * These mirror the Pydantic models defined on the backend in
 * `src/app/models/plugin.py`. Keeping them in sync ensures that the
 * data flowing from `GET /plugins/registry` is correctly typed
 * throughout the frontend.
 *
 * ┌─────────────────────────────────────────────────────────┐
 * │  Backend (Python)         →   Frontend (TypeScript)     │
 * │  CategoryInfo             →   CategoryInfo              │
 * │  ParameterInfo            →   ParameterInfo             │
 * │  ImplementationInfo       →   ImplementationInfo        │
 * │  CategoryRegistryEntry    →   CategoryRegistryEntry     │
 * └─────────────────────────────────────────────────────────┘
 */

// ── Category metadata ───────────────────────────────────

/**
 * Describes how a category of plugins behaves in the pipeline.
 *
 * - `order`        – Execution order (0 = first).
 * - `type`         – "one_time" plugins run once during the pipeline;
 *                    "multi_run" plugins can be triggered repeatedly
 *                    from the UI after the pipeline finishes.
 * - `display_name` – Human-readable label shown in the UI.
 */
export interface CategoryInfo {
  order: number;
  type: "one_time" | "multi_run";
  display_name: string;
  has_visual_results: boolean;
}

// ── Plugin parameter metadata ───────────────────────────

/**
 * Describes a single parameter of a plugin's `run()` method.
 *
 * Extracted by the backend via Python's `inspect` module from
 * the plugin class's `run(self, context, **params)` signature.
 *
 * - `name`     – Parameter name (e.g. "learning_rate").
 * - `type`     – Python type as a string (e.g. "float", "int").
 * - `default`  – Default value, or `null` if the param is required.
 * - `required` – `true` when the user must provide a value.
 */
export interface ParameterInfo {
  name: string;
  type: string;
  default: unknown;
  required: boolean;
}

// ── Single plugin implementation ────────────────────────

/**
 * Represents one concrete plugin within a category.
 *
 * For example, within the "dataset_loading" category there might be
 * a "LastFM 1K Loader" implementation.
 *
 * - `plugin_name`  – Dotted module path used to invoke the plugin
 *                     (e.g. "dataset_loading.lastFm1k_loader.lastFm1k_loader").
 * - `display_name` – Human-readable name for the UI.
 * - `params`       – Array of parameters the user can configure.
 */
export interface ImplementationInfo {
  plugin_name: string;
  display_name: string;
  params: ParameterInfo[];
}

// ── Full category entry in the registry ─────────────────

/**
 * Everything the frontend needs to render one pipeline card.
 *
 * Returned as a value in the `GET /plugins/registry` response,
 * keyed by category name (e.g. "dataset_loading").
 *
 * - `category`        – Metadata about the category itself.
 * - `implementations` – All available plugins for this category.
 */
export interface CategoryRegistryEntry {
  category_info: CategoryInfo;
  implementations: ImplementationInfo[];
}

// ── The full registry response ──────────────────────────

/**
 * Shape of the JSON returned by `GET /plugins/registry`.
 *
 * A plain object mapping category key → entry.
 * Example keys: "dataset_loading", "training_cfm", "steering".
 */
export type PluginRegistry = Record<string, CategoryRegistryEntry>;
