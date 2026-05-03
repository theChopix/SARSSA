"""Unit tests for plugins.compare_plugin_interface.

Covers ``BaseComparePlugin``: hint auto-merging, ``run()`` wrapping
that loads past_context, and the ``load_past_artifact`` helper.
All MLflow interactions are mocked.
"""

import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from plugins.compare_plugin_interface import BaseComparePlugin
from plugins.plugin_interface import (
    DynamicDropdownHint,
    MissingContextError,
    PastRunsDropdownHint,
    PluginIOSpec,
)


def _make_compare_subclass(
    *,
    extra_hints: list[Any] | None = None,
    required_steps: list[str] | None = None,
) -> type[BaseComparePlugin]:
    """Build a concrete BaseComparePlugin subclass for testing.

    Args:
        extra_hints: Optional UI hints to seed on the io_spec before
            __init_subclass__ runs.
        required_steps: Optional value for ``past_run_required_steps``.

    Returns:
        type[BaseComparePlugin]: A fresh subclass with a no-op
            ``run`` method recording its kwargs on ``self.calls``.
    """
    namespace: dict[str, Any] = {
        "io_spec": PluginIOSpec(param_ui_hints=list(extra_hints or [])),
        "past_run_required_steps": list(required_steps or []),
    }

    def run(
        self: BaseComparePlugin,
        past_run_id: str,
        neuron_id: str,
        k: int = 10,
    ) -> None:
        """Record the kwargs the wrapper forwarded.

        Args:
            self: The plugin instance.
            past_run_id: Past pipeline run id (kwarg-only via base).
            neuron_id: Concept neuron id.
            k: Top-k count.
        """
        self.calls = {  # type: ignore[attr-defined]
            "past_run_id": past_run_id,
            "neuron_id": neuron_id,
            "k": k,
        }

    namespace["run"] = run
    return type("CompareSubclass", (BaseComparePlugin,), namespace)


class TestBaseComparePluginHintMerging:
    """Tests for the auto-injected PastRunsDropdownHint."""

    def test_injects_hint_when_absent(self) -> None:
        """Verify a PastRunsDropdownHint is added for past_run_id."""
        cls = _make_compare_subclass(required_steps=["dataset_loading", "neuron_labeling"])
        hints = [h for h in cls.io_spec.param_ui_hints if isinstance(h, PastRunsDropdownHint)]
        assert len(hints) == 1
        assert hints[0].param_name == "past_run_id"
        assert hints[0].required_steps == ["dataset_loading", "neuron_labeling"]

    def test_preserves_subclass_own_hints(self) -> None:
        """Verify pre-existing hints survive the merge."""
        own = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            formatter="_fmt",
        )
        cls = _make_compare_subclass(extra_hints=[own])
        names = [h.param_name for h in cls.io_spec.param_ui_hints]
        assert "neuron_id" in names
        assert "past_run_id" in names

    def test_does_not_duplicate_existing_past_run_hint(self) -> None:
        """Verify a subclass that already declares the hint is left alone."""
        existing = PastRunsDropdownHint(param_name="past_run_id", required_steps=["x"])
        cls = _make_compare_subclass(extra_hints=[existing])
        past_hints = [h for h in cls.io_spec.param_ui_hints if isinstance(h, PastRunsDropdownHint)]
        assert len(past_hints) == 1
        assert past_hints[0].required_steps == ["x"]

    def test_subclass_io_spec_does_not_mutate_parent(self) -> None:
        """Verify mutation of subclass io_spec does not bleed into BasePlugin."""
        cls = _make_compare_subclass()
        from plugins.plugin_interface import BasePlugin

        assert cls.io_spec is not BasePlugin.io_spec


class TestBaseComparePluginRunWrapping:
    """Tests for run() wrapping that loads past_context."""

    def test_signature_preserved_for_discovery(self) -> None:
        """Verify inspect.signature still sees the original parameters."""
        cls = _make_compare_subclass()
        sig = inspect.signature(cls.run)
        names = list(sig.parameters)
        assert names == ["self", "past_run_id", "neuron_id", "k"]

    def test_wrapper_marker_set(self) -> None:
        """Verify the wrapped run carries the marker attribute."""
        cls = _make_compare_subclass()
        assert getattr(cls.run, "__compare_wrapped__", False) is True

    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_wrapper_loads_past_context_then_delegates(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify past_context is hydrated before the original body runs."""
        mock_loader = MagicMock()
        mock_loader.get_json_artifact.return_value = {
            "neuron_labeling": {"run_id": "nl_step"},
        }
        mock_loader_cls.return_value = mock_loader

        cls = _make_compare_subclass()
        plugin = cls()
        plugin.run(past_run_id="parent_xyz", neuron_id="42")

        assert plugin.past_context == {"neuron_labeling": {"run_id": "nl_step"}}
        assert plugin.calls == {"past_run_id": "parent_xyz", "neuron_id": "42", "k": 10}
        mock_loader_cls.assert_called_once_with("parent_xyz")
        mock_loader.get_json_artifact.assert_called_once_with("context.json")

    def test_missing_past_run_id_raises(self) -> None:
        """Verify omitting past_run_id surfaces MissingContextError."""
        cls = _make_compare_subclass()
        plugin = cls()
        with pytest.raises(MissingContextError, match="past_run_id"):
            plugin.run(neuron_id="42")  # type: ignore[call-arg]

    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_mlflow_failure_wraps_as_missing_context(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify any loader exception is converted to MissingContextError."""
        mock_loader = MagicMock()
        mock_loader.get_json_artifact.side_effect = FileNotFoundError("no file")
        mock_loader_cls.return_value = mock_loader

        cls = _make_compare_subclass()
        plugin = cls()
        with pytest.raises(MissingContextError, match="parent_xyz"):
            plugin.run(past_run_id="parent_xyz", neuron_id="42")

    def test_intermediate_abstract_subclass_skipped(self) -> None:
        """Verify run is not wrapped when still abstract (no marker)."""

        class IntermediateBase(BaseComparePlugin):
            """Subclass without overriding run; remains abstract."""

        assert getattr(IntermediateBase.run, "__compare_wrapped__", False) is False


class TestBaseComparePluginLoadPastArtifact:
    """Tests for load_past_artifact dispatching through past_context."""

    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_loads_json_artifact_from_past_step(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify json artifacts are loaded from the resolved past step."""
        mock_loader = MagicMock()
        mock_loader.get_json_artifact.return_value = {"7": "drama"}
        mock_loader_cls.return_value = mock_loader

        cls = _make_compare_subclass()
        plugin = cls()
        plugin.past_context = {"neuron_labeling": {"run_id": "past_nl"}}

        result = plugin.load_past_artifact(
            "neuron_labeling",
            "neuron_labels.json",
            "json",
        )

        assert result == {"7": "drama"}
        mock_loader_cls.assert_called_once_with("past_nl")
        mock_loader.get_json_artifact.assert_called_once_with("neuron_labels.json")

    def test_raises_when_past_context_not_loaded(self) -> None:
        """Verify error when past_context has not been hydrated yet."""
        cls = _make_compare_subclass()
        plugin = cls()
        with pytest.raises(MissingContextError, match="past_context"):
            plugin.load_past_artifact("neuron_labeling", "x.json", "json")

    def test_raises_when_step_missing_from_context(self) -> None:
        """Verify a missing step yields MissingContextError."""
        cls = _make_compare_subclass()
        plugin = cls()
        plugin.past_context = {"dataset_loading": {"run_id": "ds"}}
        with pytest.raises(MissingContextError, match="neuron_labeling"):
            plugin.load_past_artifact("neuron_labeling", "x.json", "json")

    def test_raises_when_step_entry_missing_run_id(self) -> None:
        """Verify a malformed step entry yields MissingContextError."""
        cls = _make_compare_subclass()
        plugin = cls()
        plugin.past_context = {"neuron_labeling": {"foo": "bar"}}
        with pytest.raises(MissingContextError):
            plugin.load_past_artifact("neuron_labeling", "x.json", "json")

    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_unknown_loader_raises_value_error(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify unsupported loader strategy bubbles ValueError."""
        mock_loader_cls.return_value = MagicMock()
        cls = _make_compare_subclass()
        plugin = cls()
        plugin.past_context = {"neuron_labeling": {"run_id": "past_nl"}}
        with pytest.raises(ValueError, match="Unknown loader type"):
            plugin.load_past_artifact("neuron_labeling", "x.csv", "csv")
