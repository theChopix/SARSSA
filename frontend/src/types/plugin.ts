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

// ── Widget configuration ────────────────────────────────

/**
 * Extra configuration for non-default parameter widgets.
 *
 * Only the fields relevant to the chosen `widget` type are
 * populated; the rest are `null`.
 *
 * - `choices_endpoint` – URL path for fetching dynamic dropdown
 *                         options (used when `widget = "dropdown"`).
 * - `run_id_source`    – Pipeline context key whose `run_id` should
 *                         be passed as a query param when fetching
 *                         choices (e.g. `"neuron_labeling"`).
 * - `slider_min`       – Minimum value for slider widgets.
 * - `slider_max`       – Maximum value for slider widgets.
 * - `slider_step`      – Step increment for slider widgets.
 */
export interface WidgetConfig {
  choices_endpoint: string | null;
  run_id_source: string | null;
  slider_min: number | null;
  slider_max: number | null;
  slider_step: number | null;
}

// ── Plugin parameter metadata ───────────────────────────

/**
 * Describes a single parameter of a plugin's `run()` method.
 *
 * Extracted by the backend via Python's `inspect` module from
 * the plugin class's `run(self, context, **params)` signature.
 *
 * - `name`          – Parameter name (e.g. "learning_rate").
 * - `type`          – Python type as a string (e.g. "float", "int").
 * - `default`       – Default value, or `null` if the param is required.
 * - `required`      – `true` when the user must provide a value.
 * - `widget`        – Frontend widget type (`"text"` or `"dropdown"`).
 * - `widget_config` – Extra configuration for non-default widgets.
 */
export interface ParameterInfo {
  name: string;
  type: string;
  default: unknown;
  required: boolean;
  widget: string;
  widget_config: WidgetConfig | null;
}

// ── Display spec (visual output metadata) ───────────────
//
// DisplaySpec is a discriminated union on the `type` field.
// Each variant carries the data needed by a specific frontend
// rendering strategy.

/**
 * One row of visual items to render in the frontend.
 *
 * - `key`   – Key in the plugin's output artifacts (e.g. "interacted_items").
 * - `label` – Human-readable row label for the UI (e.g. "Interaction History").
 */
export interface DisplayRowSpec {
  key: string;
  label: string;
}

/**
 * Horizontal scrollable rows of enriched item cards.
 *
 * Each row corresponds to one output artifact containing a
 * list of item IDs that are enriched with metadata and
 * rendered as ItemCard components.
 */
export interface ItemRowsDisplaySpec {
  type: "item_rows";
  rows: DisplayRowSpec[];
}

/**
 * One renderable artifact file produced by a plugin.
 *
 * - `filename`     – Artifact filename (e.g. "dendrogram.svg").
 * - `label`        – Human-readable label for the UI.
 * - `content_type` – MIME type for rendering (e.g. "image/svg+xml", "text/html").
 */
export interface ArtifactFileSpec {
  filename: string;
  label: string;
  content_type: string;
}

/**
 * Standalone visual artifacts rendered inline.
 *
 * Each file entry is fetched from the step's MLflow artifacts
 * and rendered according to its content_type.
 */
export interface ArtifactDisplaySpec {
  type: "artifact";
  files: ArtifactFileSpec[];
}

/**
 * Discriminated union of all display spec variants.
 *
 * Switch on `displaySpec.type` to determine the rendering strategy.
 */
export type DisplaySpec = ItemRowsDisplaySpec | ArtifactDisplaySpec;

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
 * - `display`      – Optional display spec for visual output rendering.
 * - `kind`         – Plugin variant ("single" / "compare") derived from
 *                    the folder layout, or `null` when the category does
 *                    not opt into the single/compare distinction.
 */
export interface ImplementationInfo {
  plugin_name: string;
  display_name: string;
  params: ParameterInfo[];
  display: DisplaySpec | null;
  kind: "single" | "compare" | null;
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
