import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import scipy.sparse as sp

from utils.mlflow_manager import MLflowRunLoader
from utils.plugin_notifier import NullNotifier, PluginNotifier


class MissingContextError(Exception):
    """Raised when a required pipeline step is missing from context.

    Args:
        message: Human-readable description of the missing dependency.
    """


@dataclass
class ArtifactSpec:
    """One artifact to load from a previous step's MLflow run.

    Attributes:
        step: Context key of the upstream pipeline step
            (e.g. ``"dataset_loading"``).
        filename: Artifact filename stored in MLflow
            (e.g. ``"train_csr.npz"``).
        attr: Attribute name set on the plugin instance after
            loading (e.g. ``"train_csr"``).
        loader: Loader strategy identifier. Supported values:
            ``"npz"``, ``"npy"``, ``"json"``, ``"model_dir"``,
            ``"pt"``.
        loader_kwargs: Extra keyword arguments forwarded to the
            loader function.
    """

    step: str
    filename: str
    attr: str
    loader: str
    loader_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParamSpec:
    """One parameter to read from a previous step's MLflow run.

    Attributes:
        step: Context key of the upstream pipeline step
            (e.g. ``"dataset_loading"``).
        param_name: MLflow parameter name
            (e.g. ``"num_users"``).
        attr: Attribute name set on the plugin instance after
            loading.
        dtype: Target type to cast the raw string value to.
    """

    step: str
    param_name: str
    attr: str
    dtype: type = str


@dataclass
class OutputArtifactSpec:
    """One artifact this plugin will produce.

    Attributes:
        attr: Attribute name on the plugin instance holding the
            value to save.
        filename: Target filename in MLflow
            (e.g. ``"neuron_labels.json"``).
        saver: Saver strategy identifier. Supported values:
            ``"json"``, ``"npz"``, ``"npy"``, ``"pt"``.
    """

    attr: str
    filename: str
    saver: str


@dataclass
class OutputParamSpec:
    """One parameter this plugin will log to MLflow.

    Attributes:
        key: MLflow parameter key.
        attr: Attribute name on the plugin instance holding the
            value to log.
    """

    key: str
    attr: str


@dataclass
class DisplayRowSpec:
    """One row of visual items to render in the frontend.

    Attributes:
        key: Key in the plugin's output artifacts (e.g.
            ``"interacted_items"``).
        label: Human-readable row label for the UI (e.g.
            ``"Interaction History"``).
    """

    key: str
    label: str


@dataclass
class ItemRowsDisplaySpec:
    """Horizontal scrollable rows of enriched item cards.

    Each row corresponds to one output artifact containing a
    list of item IDs that will be enriched with metadata and
    rendered as ``ItemCard`` components.

    Attributes:
        type: Display layout discriminator (always ``"item_rows"``).
        rows: Ordered list of item-row specifications.
    """

    type: str = "item_rows"
    rows: list[DisplayRowSpec] = field(default_factory=list)


@dataclass
class ArtifactFileSpec:
    """One renderable artifact file produced by a plugin.

    Attributes:
        filename: Artifact filename (e.g. ``"dendrogram.svg"``).
        label: Human-readable label for the UI (e.g.
            ``"Dendrogram"``).
        content_type: MIME type used for rendering (e.g.
            ``"image/svg+xml"``, ``"text/html"``).
    """

    filename: str
    label: str
    content_type: str


@dataclass
class ArtifactDisplaySpec:
    """Standalone visual artifacts rendered inline.

    Each file entry is fetched from the step's MLflow artifacts
    and rendered according to its ``content_type`` (e.g. an
    ``<img>`` for SVG, an ``<iframe>`` for HTML).

    Attributes:
        type: Display layout discriminator (always ``"artifact"``).
        files: Ordered list of artifact file specifications.
    """

    type: str = "artifact"
    files: list[ArtifactFileSpec] = field(default_factory=list)


DisplaySpec = ItemRowsDisplaySpec | ArtifactDisplaySpec


@dataclass
class ParamUIHint:
    """Base class for parameter UI rendering hints.

    Subclasses declare how a specific ``run()`` parameter should
    be rendered in the frontend (e.g. dropdown, slider).  Plugins
    attach these to ``PluginIOSpec.param_ui_hints`` so the registry
    builder can translate them into widget metadata for the API.

    Attributes:
        param_name: Name of the ``run()`` parameter this hint
            applies to.
    """

    param_name: str


@dataclass
class DynamicDropdownHint(ParamUIHint):
    """Render a parameter as a dropdown populated from an artifact.

    The backend loads the artifact from the specified upstream
    pipeline step, passes the raw data to the plugin's
    ``formatter`` static method, and returns a list of
    ``{"label": ..., "value": ...}`` options to the frontend.

    Attributes:
        artifact_step: Context key of the pipeline step that
            produces the source artifact.
        artifact_file: Filename of the artifact in MLflow.
        artifact_loader: Loader strategy identifier (e.g.
            ``"json"``, ``"npy"``).
        formatter: Name of a **static method** on the plugin
            class that transforms the loaded artifact data into
            ``list[dict[str, str]]`` with ``"label"`` and
            ``"value"`` keys.
    """

    artifact_step: str = ""
    artifact_file: str = ""
    artifact_loader: str = "json"
    formatter: str = ""


@dataclass
class PastRunsDropdownHint(ParamUIHint):
    """Render a parameter as a dropdown of past pipeline runs.

    Used by *compare* plugins to let the user pick a previously
    completed pipeline run to compare against.  The frontend fetches
    the list of eligible runs from the pipelines runs endpoint and
    sends ``required_steps`` so only runs whose ``context.json``
    contains all of those step keys are returned.

    Attributes:
        required_steps: Step keys that must be present in a past
            run's ``context.json`` for the run to be considered
            eligible.  Empty list means every top-level run is
            eligible.
    """

    required_steps: list[str] = field(default_factory=list)


@dataclass
class SliderHint(ParamUIHint):
    """Render a parameter as a range slider.

    The frontend displays an ``<input type="range">`` with the
    configured bounds and step size, along with the current numeric
    value.

    Attributes:
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.
        step: Increment between allowed values.
    """

    min_value: float = 0.0
    max_value: float = 1.0
    step: float = 0.01


@dataclass
class PluginIOSpec:
    """Full I/O contract for a plugin.

    Attributes:
        required_steps: Context keys that must be present before
            the plugin can run.
        input_artifacts: Artifacts to load from upstream steps.
        input_params: Parameters to read from upstream steps.
        output_artifacts: Artifacts this plugin produces.
        output_params: Parameters this plugin logs.
        display: Optional declarative display specification
            describing how the frontend should render this
            plugin's output.  ``None`` means no visual rendering
            (the default for most plugins).
        param_ui_hints: Optional list of UI rendering hints for
            ``run()`` parameters.  Each hint declares how a
            parameter should be presented in the frontend.
    """

    required_steps: list[str] = field(default_factory=list)
    input_artifacts: list[ArtifactSpec] = field(
        default_factory=list,
    )
    input_params: list[ParamSpec] = field(default_factory=list)
    output_artifacts: list[OutputArtifactSpec] = field(
        default_factory=list,
    )
    output_params: list[OutputParamSpec] = field(
        default_factory=list,
    )
    display: DisplaySpec | None = None
    param_ui_hints: list[ParamUIHint] = field(
        default_factory=list,
    )


class BasePlugin(ABC):
    """Base class for all pipeline plugins.

    Subclasses declare an ``io_spec`` describing their inputs and
    outputs.  The pipeline engine drives the lifecycle::

        plugin.load_context(context)   # validate + hydrate self.*
        plugin.run(**params)           # pure business logic
        plugin.update_context()        # log self.* outputs to MLflow

    Attributes:
        name: Optional human-readable display name. When set, the
            plugin registry uses this instead of the auto-derived
            name.
        io_spec: Declarative I/O contract for the plugin.
        notifier: Notifier for pushing messages to the UI during
            execution.  Defaults to :class:`~utils.plugin_notifier.NullNotifier`
            (silent no-op).  The pipeline engine replaces this with a
            real :class:`~utils.plugin_notifier.PluginNotifier` before
            calling ``run()``.
    """

    name: str | None = None
    io_spec: PluginIOSpec = PluginIOSpec()
    notifier: PluginNotifier = NullNotifier()

    def load_context(self, context: dict[str, Any]) -> None:
        """Validate required steps and hydrate inputs from MLflow.

        For each entry in ``io_spec.input_artifacts`` and
        ``io_spec.input_params``, downloads the value from the
        upstream MLflow run and sets it as ``self.<attr>``.

        Subclasses may override this to add custom validation
        (e.g. cross-step assertions) but should call
        ``super().load_context(context)`` first.

        Args:
            context: Pipeline context mapping step names to
                dicts containing at least ``{"run_id": ...}``.

        Raises:
            MissingContextError: If a required step or its
                ``run_id`` is missing from *context*.
        """
        self._context = context
        self._validate_required_steps(context)
        self._load_input_artifacts(context)
        self._load_input_params(context)

    def update_context(self) -> None:
        """Log all declared outputs to the active MLflow run.

        Reads ``self.<attr>`` for every entry in
        ``io_spec.output_params`` and ``io_spec.output_artifacts``
        and logs them via ``mlflow.log_params`` /
        ``mlflow.log_artifacts``.
        """
        self._log_output_params()
        self._log_output_artifacts()

    @abstractmethod
    def run(self, **params: Any) -> None:
        """Execute the plugin's core logic.

        After ``load_context`` has been called, all declared inputs
        are available on ``self.*``.  The method should populate
        ``self.*`` output attributes for ``update_context``.

        During migration, unmigrated plugins can access the pipeline
        context via ``self._context`` (set by ``load_context``).

        Args:
            **params: Plugin-specific keyword arguments.
        """

    # ── private helpers ──────────────────────────────────────────

    def _validate_required_steps(
        self,
        context: dict[str, Any],
    ) -> None:
        """Check that every required step is present in *context*.

        Args:
            context: Pipeline context dict.

        Raises:
            MissingContextError: If a step key is absent or
                has no ``run_id``.
        """
        for step in self.io_spec.required_steps:
            if step not in context:
                raise MissingContextError(
                    f"Plugin requires step '{step}' but it's missing from context"
                )
            if "run_id" not in context[step]:
                raise MissingContextError(f"Step '{step}' is present but has no 'run_id'")

    def _load_input_artifacts(
        self,
        context: dict[str, Any],
    ) -> None:
        """Download and set all declared input artifacts.

        Args:
            context: Pipeline context dict.

        Raises:
            MissingContextError: If an artifact cannot be retrieved
                from the upstream MLflow run.
        """
        for spec in self.io_spec.input_artifacts:
            run_id = context[spec.step]["run_id"]
            loader = MLflowRunLoader(run_id)
            try:
                value = self._load_artifact(loader, spec)
            except Exception as exc:
                raise MissingContextError(
                    f"Failed to load artifact '{spec.filename}' "
                    f"(loader='{spec.loader}') from step "
                    f"'{spec.step}' (run_id='{run_id}'): {exc}"
                ) from exc
            setattr(self, spec.attr, value)

    def _load_input_params(
        self,
        context: dict[str, Any],
    ) -> None:
        """Read and cast all declared input parameters.

        Args:
            context: Pipeline context dict.

        Raises:
            MissingContextError: If a parameter is missing from the
                upstream MLflow run or cannot be cast to the
                declared type.
        """
        for spec in self.io_spec.input_params:
            run_id = context[spec.step]["run_id"]
            loader = MLflowRunLoader(run_id)
            raw = loader.get_parameter(spec.param_name)
            if raw is None:
                raise MissingContextError(
                    f"Parameter '{spec.param_name}' not found in "
                    f"step '{spec.step}' (run_id='{run_id}')"
                )
            try:
                setattr(self, spec.attr, spec.dtype(raw))
            except (ValueError, TypeError) as exc:
                raise MissingContextError(
                    f"Cannot cast parameter '{spec.param_name}' "
                    f"value '{raw}' to {spec.dtype.__name__} from "
                    f"step '{spec.step}' (run_id='{run_id}'): {exc}"
                ) from exc

    def _load_artifact(
        self,
        loader: MLflowRunLoader,
        spec: ArtifactSpec,
    ) -> Any:
        """Dispatch artifact loading based on ``spec.loader``.

        Args:
            loader: MLflow run loader for the upstream step.
            spec: Artifact specification describing what to load.

        Returns:
            Any: The loaded artifact value.

        Raises:
            ValueError: If ``spec.loader`` is not a recognised
                strategy.
        """
        match spec.loader:
            case "npz":
                return loader.get_npz_artifact(spec.filename, **spec.loader_kwargs)
            case "npy":
                return loader.get_npy_artifact(spec.filename, **spec.loader_kwargs)
            case "json":
                return loader.get_json_artifact(spec.filename, **spec.loader_kwargs)
            case "base_model":
                from utils.torch.models.model_loader import load_base_model

                path = loader.download_artifact_dir(**spec.loader_kwargs)
                return load_base_model(
                    path,
                    device=spec.loader_kwargs.get("device", "cpu"),
                )
            case "sae_model":
                from utils.torch.models.model_loader import load_sae_model

                path = loader.download_artifact_dir(**spec.loader_kwargs)
                return load_sae_model(
                    path,
                    device=spec.loader_kwargs.get("device", "cpu"),
                )
            case "pt":
                import torch

                path = loader.download_artifact(spec.filename)
                return torch.load(
                    path,
                    map_location=spec.loader_kwargs.get("map_location", "cpu"),
                    weights_only=spec.loader_kwargs.get("weights_only", True),
                )
            case _:
                raise ValueError(f"Unknown loader type: {spec.loader}")

    def _log_output_params(self) -> None:
        """Log all declared output params to MLflow."""
        params: dict[str, Any] = {}
        for spec in self.io_spec.output_params:
            params[spec.key] = getattr(self, spec.attr)
        if params:
            mlflow.log_params(params)

    def _log_output_artifacts(self) -> None:
        """Save all declared output artifacts to MLflow."""
        if not self.io_spec.output_artifacts:
            return
        with tempfile.TemporaryDirectory() as tmp:
            for spec in self.io_spec.output_artifacts:
                self._save_artifact(tmp, spec)
            mlflow.log_artifacts(tmp)

    def _save_artifact(
        self,
        tmp_dir: str,
        spec: OutputArtifactSpec,
    ) -> None:
        """Dispatch artifact saving based on ``spec.saver``.

        Args:
            tmp_dir: Temporary directory to write the file into.
            spec: Output artifact specification.

        Raises:
            ValueError: If ``spec.saver`` is not a recognised
                strategy.
        """
        value = getattr(self, spec.attr)
        path = os.path.join(tmp_dir, spec.filename)
        match spec.saver:
            case "json":
                with open(path, "w") as f:
                    json.dump(value, f, indent=2, default=str)
            case "npz":
                sp.save_npz(path, value)
            case "npy":
                np.save(path, value)
            case "pt":
                import torch

                torch.save(value, path)
            case "model":
                import torch

                config = value.get_config()
                with open(
                    os.path.join(tmp_dir, "config.json"),
                    "w",
                ) as f:
                    json.dump(config, f, indent=2)
                torch.save(
                    {"state_dict": value.state_dict()},
                    os.path.join(tmp_dir, "model.pt"),
                )
            case _:
                raise ValueError(f"Unknown saver type: {spec.saver}")
