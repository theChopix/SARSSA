"""Pydantic models for plugin-related data."""

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Tag


class CategoryType(StrEnum):
    """Allowed execution types for a plugin category."""

    ONE_TIME = "one_time"
    MULTI_RUN = "multi_run"


class CategoryInfo(BaseModel):
    """Metadata for a single plugin category.

    Attributes:
        order: Execution order within the pipeline (0-indexed).
        type: Execution type of the category.
        display_name: Human-readable label for the UI.
        has_visual_results: Whether plugins in this category
            produce visual output that the frontend should
            render in a results panel.
    """

    order: int
    type: CategoryType
    display_name: str
    has_visual_results: bool = False


class WidgetConfig(BaseModel):
    """Configuration payload for a non-default parameter widget.

    Only the fields relevant to the chosen ``widget`` type are
    populated; every other field stays ``None`` (or ``None`` /
    empty list for collections).  The valid combinations are:

    - ``widget="dropdown"`` (regular dynamic dropdown):
      ``choices_endpoint`` + ``run_id_source``.  The frontend
      resolves the upstream step's run id from the current
      pipeline context and calls the param-choices endpoint with
      it.

    - ``widget="dropdown"`` (cascading variant): same fields as the
      regular dropdown, plus ``source_run_param`` pointing to
      another parameter on the same plugin.  When set, the
      frontend uses that param's current value (a parent pipeline
      run id) instead of resolving via ``run_id_source``, and
      refetches whenever the watched param changes.

    - ``widget="past_runs_dropdown"``: ``required_steps`` only.
      The frontend hits ``/pipelines/runs?required_steps=...``
      directly to list eligible past runs; no
      ``choices_endpoint`` / ``run_id_source`` is needed.

    - ``widget="slider"``: ``slider_min`` + ``slider_max`` +
      ``slider_step`` define the slider's bounds and increment.

    - ``widget="text"`` (default): ``WidgetConfig`` is omitted
      entirely (``ParameterInfo.widget_config`` is ``None``).

    Attributes:
        choices_endpoint: URL path for fetching dynamic dropdown
            options (used when ``widget="dropdown"``).
        run_id_source: Pipeline context key whose ``run_id`` the
            frontend should pass as query param when fetching
            choices (e.g. ``"neuron_labeling"``).
        source_run_param: Name of another parameter on the same
            plugin whose value is a parent pipeline run id.  When
            set, the frontend should pass that param's current
            value as ``run_id`` to ``choices_endpoint`` and refetch
            whenever it changes — overriding the
            ``run_id_source`` lookup against the current pipeline
            context.
        slider_min: Minimum value for slider widgets.
        slider_max: Maximum value for slider widgets.
        slider_step: Step increment for slider widgets.
        required_steps: Step keys that an eligible past run must
            contain in its ``context.json`` (used when
            ``widget="past_runs_dropdown"``).
    """

    choices_endpoint: str | None = None
    run_id_source: str | None = None
    source_run_param: str | None = None
    slider_min: float | None = None
    slider_max: float | None = None
    slider_step: float | None = None
    required_steps: list[str] | None = None


class ParameterInfo(BaseModel):
    """Schema for a single plugin parameter.

    Attributes:
        name: Parameter name as it appears in the run() signature.
        type: Python type name (e.g. ``int``, ``float``, ``str``).
        default: Default value, or None if the parameter is required.
        required: Whether the parameter must be provided by the user.
        widget: Frontend widget type. Defaults to ``"text"``.
        widget_config: Extra configuration for non-default widgets.
    """

    name: str
    type: str
    default: Any | None = None
    required: bool = True
    widget: str = "text"
    widget_config: WidgetConfig | None = None


class DisplayRowSpec(BaseModel):
    """One row of visual items to render in the frontend.

    Attributes:
        key: Key in the plugin's output artifacts.
        label: Human-readable row label for the UI.
    """

    key: str
    label: str


class ItemRowsDisplayModel(BaseModel):
    """Horizontal scrollable rows of enriched item cards.

    Attributes:
        type: Display layout discriminator (always ``"item_rows"``).
        rows: Ordered list of item-row specifications.
    """

    type: Literal["item_rows"] = "item_rows"
    rows: list[DisplayRowSpec] = []


class ArtifactFileModel(BaseModel):
    """One renderable artifact file produced by a plugin.

    Attributes:
        filename: Artifact filename (e.g. ``"dendrogram.svg"``).
        label: Human-readable label for the UI.
        content_type: MIME type for rendering.
    """

    filename: str
    label: str
    content_type: str


class ArtifactDisplayModel(BaseModel):
    """Standalone visual artifacts rendered inline.

    Attributes:
        type: Display layout discriminator (always ``"artifact"``).
        files: Ordered list of artifact file specifications.
    """

    type: Literal["artifact"] = "artifact"
    files: list[ArtifactFileModel] = []


DisplaySpec = Annotated[
    Annotated[ItemRowsDisplayModel, Tag("item_rows")]
    | Annotated[ArtifactDisplayModel, Tag("artifact")],
    Discriminator("type"),
]


class ImplementationInfo(BaseModel):
    """Schema for a discovered plugin implementation.

    Attributes:
        plugin_name: Dotted module path used by PluginManager.load()
            (e.g. ``dataset_loading.movieLens_loader.movieLens_loader``).
        display_name: Human-readable name derived from the module name.
        params: List of configurable parameters from the run() signature.
        display: Optional display specification describing how the
            frontend should render this plugin's output.
        kind: Plugin variant derived from the folder layout.
            ``"single"`` if the plugin lives directly under a
            ``single/`` subfolder of its category, ``"compare"`` if
            under ``compare/``.  ``None`` when the category does not
            opt into the single/compare distinction.
    """

    plugin_name: str
    display_name: str
    params: list[ParameterInfo]
    display: ItemRowsDisplayModel | ArtifactDisplayModel | None = None
    kind: Literal["single", "compare"] | None = None


class CategoryRegistryEntry(BaseModel):
    """Full registry entry for a single plugin category.

    Attributes:
        category_info: Static metadata for the category.
        implementations: Discovered plugin implementations.
    """

    category_info: CategoryInfo
    implementations: list[ImplementationInfo]
