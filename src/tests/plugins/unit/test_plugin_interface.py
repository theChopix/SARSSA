"""Unit tests for plugin_interface spec dataclasses and MissingContextError."""

import pytest

from plugins.plugin_interface import (
    ArtifactSpec,
    MissingContextError,
    OutputArtifactSpec,
    OutputParamSpec,
    ParamSpec,
    PluginIOSpec,
)

# ── MissingContextError ─────────────────────────────────────────────


class TestMissingContextError:
    """Tests for the MissingContextError exception."""

    def test_is_exception_subclass(self) -> None:
        """Verify MissingContextError inherits from Exception."""
        assert issubclass(MissingContextError, Exception)

    def test_stores_message(self) -> None:
        """Verify the error message is accessible via args."""
        err = MissingContextError("step 'training_cfm' is missing")
        assert str(err) == "step 'training_cfm' is missing"

    def test_can_be_raised_and_caught(self) -> None:
        """Verify it can be raised and caught specifically."""
        with pytest.raises(MissingContextError, match="missing"):
            raise MissingContextError("missing")


# ── ArtifactSpec ─────────────────────────────────────────────────────


class TestArtifactSpec:
    """Tests for the ArtifactSpec dataclass."""

    def test_construction_all_fields(self) -> None:
        """Verify all fields are stored correctly."""
        spec = ArtifactSpec(
            step="dataset_loading",
            filename="train_csr.npz",
            attr="train_csr",
            loader="npz",
            loader_kwargs={"return_sparse": True},
        )
        assert spec.step == "dataset_loading"
        assert spec.filename == "train_csr.npz"
        assert spec.attr == "train_csr"
        assert spec.loader == "npz"
        assert spec.loader_kwargs == {"return_sparse": True}

    def test_loader_kwargs_defaults_to_empty_dict(self) -> None:
        """Verify loader_kwargs defaults to an empty dict."""
        spec = ArtifactSpec(
            step="s",
            filename="f.npz",
            attr="a",
            loader="npz",
        )
        assert spec.loader_kwargs == {}

    def test_default_loader_kwargs_not_shared(self) -> None:
        """Verify each instance gets its own default dict."""
        spec_a = ArtifactSpec("s", "f", "a", "npz")
        spec_b = ArtifactSpec("s", "f", "a", "npz")
        spec_a.loader_kwargs["key"] = "value"
        assert spec_b.loader_kwargs == {}


# ── ParamSpec ────────────────────────────────────────────────────────


class TestParamSpec:
    """Tests for the ParamSpec dataclass."""

    def test_construction_all_fields(self) -> None:
        """Verify all fields are stored correctly."""
        spec = ParamSpec(
            step="dataset_loading",
            param_name="num_users",
            attr="num_users",
            dtype=int,
        )
        assert spec.step == "dataset_loading"
        assert spec.param_name == "num_users"
        assert spec.attr == "num_users"
        assert spec.dtype is int

    def test_dtype_defaults_to_str(self) -> None:
        """Verify dtype defaults to str."""
        spec = ParamSpec(
            step="s",
            param_name="p",
            attr="a",
        )
        assert spec.dtype is str


# ── OutputArtifactSpec ───────────────────────────────────────────────


class TestOutputArtifactSpec:
    """Tests for the OutputArtifactSpec dataclass."""

    def test_construction_all_fields(self) -> None:
        """Verify all fields are stored correctly."""
        spec = OutputArtifactSpec(
            attr="train_csr",
            filename="train_csr.npz",
            saver="npz",
        )
        assert spec.attr == "train_csr"
        assert spec.filename == "train_csr.npz"
        assert spec.saver == "npz"


# ── OutputParamSpec ──────────────────────────────────────────────────


class TestOutputParamSpec:
    """Tests for the OutputParamSpec dataclass."""

    def test_construction_all_fields(self) -> None:
        """Verify all fields are stored correctly."""
        spec = OutputParamSpec(key="dataset_name", attr="dataset")
        assert spec.key == "dataset_name"
        assert spec.attr == "dataset"


# ── PluginIOSpec ─────────────────────────────────────────────────────


class TestPluginIOSpec:
    """Tests for the PluginIOSpec dataclass."""

    def test_all_fields_default_to_empty(self) -> None:
        """Verify all list fields default to empty lists."""
        spec = PluginIOSpec()
        assert spec.required_steps == []
        assert spec.input_artifacts == []
        assert spec.input_params == []
        assert spec.output_artifacts == []
        assert spec.output_params == []

    def test_construction_with_all_fields(self) -> None:
        """Verify full construction with populated lists."""
        artifact = ArtifactSpec("s", "f.npz", "a", "npz")
        param = ParamSpec("s", "p", "a", int)
        out_artifact = OutputArtifactSpec("a", "f.npz", "npz")
        out_param = OutputParamSpec("k", "a")

        spec = PluginIOSpec(
            required_steps=["dataset_loading"],
            input_artifacts=[artifact],
            input_params=[param],
            output_artifacts=[out_artifact],
            output_params=[out_param],
        )

        assert spec.required_steps == ["dataset_loading"]
        assert spec.input_artifacts == [artifact]
        assert spec.input_params == [param]
        assert spec.output_artifacts == [out_artifact]
        assert spec.output_params == [out_param]

    def test_default_lists_not_shared(self) -> None:
        """Verify each instance gets independent default lists."""
        spec_a = PluginIOSpec()
        spec_b = PluginIOSpec()
        spec_a.required_steps.append("step_x")
        assert spec_b.required_steps == []

    def test_multiple_required_steps(self) -> None:
        """Verify multiple required steps are stored."""
        spec = PluginIOSpec(
            required_steps=[
                "dataset_loading",
                "training_cfm",
                "training_sae",
            ],
        )
        assert len(spec.required_steps) == 3
        assert "training_cfm" in spec.required_steps
