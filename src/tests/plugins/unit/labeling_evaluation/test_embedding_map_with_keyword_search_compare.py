"""Unit tests for labeling_evaluation.compare.embedding_map_with_keyword_search.

Patches MLflowRunLoader (past side) and the embedding helpers
(``embed_labels``, ``compute_label_embedding_coords``) inside the
plugin module.  Verifies search_scope branching, sidebar layout per
mode, and the keyword param guard.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _build_plugin() -> Any:
    """Build a Plugin instance with current-side state preset.

    Returns:
        Plugin: Compare-mode keyword-search plugin instance.
    """
    from plugins.labeling_evaluation.compare.embedding_map_with_keyword_search.embedding_map_with_keyword_search import (  # noqa: E501
        Plugin,
    )

    plugin = Plugin()
    plugin.neuron_labels = {"0": "current_a", "1": "current_b"}
    plugin.current_neuron_ids = ["0", "1"]
    plugin.current_label_texts = ["current_a", "current_b"]
    return plugin


def _make_past_loader(
    *,
    context: dict[str, Any],
    past_neuron_labels: dict[str, str],
) -> MagicMock:
    """Build a MagicMock that mimics MLflowRunLoader for the past side.

    Args:
        context: Past-run context.json contents.
        past_neuron_labels: Past-run neuron_labels.json contents.

    Returns:
        MagicMock: Stand-in MLflowRunLoader.
    """

    def get_json(filename: str) -> Any:
        match filename:
            case "context.json":
                return context
            case "neuron_labels.json":
                return past_neuron_labels
            case other:
                raise AssertionError(f"unexpected json artifact: {other}")

    loader = MagicMock()
    loader.get_json_artifact.side_effect = get_json
    return loader


# Embeddings shared across most tests below: 2 current + 3 past + 1 keyword.
# Keyword direction is [1, 0]:
#   current_a → similarity 1.0 (best on current side)
#   current_b → similarity 0.0
#   past_a    → similarity 1.0 (best on past side)
#   past_b    → similarity 0.0
#   past_c    → similarity 0.5 (60° angle)
_SHARED_EMBEDDINGS = np.array(
    [
        [1.0, 0.0],  # current_a (idx 0)
        [0.0, 1.0],  # current_b (idx 1)
        [1.0, 0.0],  # past_a    (idx 2)
        [0.0, 1.0],  # past_b    (idx 3)
        [0.5, 0.866025],  # past_c    (idx 4)
        [1.0, 0.0],  # keyword   (idx 5)
    ]
)
# Coords are arbitrary; only used for plot positioning.
_SHARED_COORDS = np.array(
    [
        [10.0, 11.0],
        [20.0, 21.0],
        [30.0, 31.0],
        [40.0, 41.0],
        [50.0, 51.0],
        [60.0, 61.0],
    ]
)


class TestCompareKeywordSearchSeparate:
    """Tests for search_scope='separate'."""

    @patch(
        "plugins.labeling_evaluation.compare.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.compute_label_embedding_coords"
    )
    @patch(
        "plugins.labeling_evaluation.compare.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.embed_labels"
    )
    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_two_sidebars_one_per_side(
        self,
        mock_compare_loader_cls: MagicMock,
        mock_embed_labels: MagicMock,
        mock_compute_coords: MagicMock,
    ) -> None:
        """Verify per-side top-k yields exactly two sidebar sections.

        Args:
            mock_compare_loader_cls: Patched MLflowRunLoader class.
            mock_embed_labels: Patched cache helper.
            mock_compute_coords: Patched UMAP helper.
        """
        mock_compare_loader_cls.return_value = _make_past_loader(
            context={"neuron_labeling": {"run_id": "nl_past"}},
            past_neuron_labels={"7": "past_a", "8": "past_b", "9": "past_c"},
        )
        mock_embed_labels.return_value = _SHARED_EMBEDDINGS
        mock_compute_coords.return_value = _SHARED_COORDS

        plugin = _build_plugin()
        plugin.run(
            past_run_id="parent_xyz",
            keyword="search me",
            k=1,
            search_scope="separate",
        )

        sidebars = plugin._sidebars
        assert len(sidebars) == 2
        assert sidebars[0].title.startswith("Current")
        assert sidebars[1].title.startswith("Past")
        assert len(sidebars[0].items) == 1
        assert len(sidebars[1].items) == 1

        current_match = sidebars[0].items[0]
        past_match = sidebars[1].items[0]
        assert current_match.label == "[neuron 0] current_a"
        assert past_match.label == "[neuron 7] past_a"
        assert current_match.badge is None
        assert past_match.badge is None

        # Per-side similarity 1.0 for both top picks.
        assert current_match.similarity == pytest.approx(1.0)
        assert past_match.similarity == pytest.approx(1.0)

    @patch(
        "plugins.labeling_evaluation.compare.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.compute_label_embedding_coords"
    )
    @patch(
        "plugins.labeling_evaluation.compare.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.embed_labels"
    )
    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_top_k_matches_json_splits_per_side(
        self,
        mock_compare_loader_cls: MagicMock,
        mock_embed_labels: MagicMock,
        mock_compute_coords: MagicMock,
    ) -> None:
        """Verify the JSON artifact has per-side current/past keys.

        Args:
            mock_compare_loader_cls: Patched MLflowRunLoader class.
            mock_embed_labels: Patched cache helper.
            mock_compute_coords: Patched UMAP helper.
        """
        mock_compare_loader_cls.return_value = _make_past_loader(
            context={"neuron_labeling": {"run_id": "nl_past"}},
            past_neuron_labels={"7": "past_a", "8": "past_b", "9": "past_c"},
        )
        mock_embed_labels.return_value = _SHARED_EMBEDDINGS
        mock_compute_coords.return_value = _SHARED_COORDS

        plugin = _build_plugin()
        plugin.run(
            past_run_id="parent_xyz",
            keyword="search me",
            k=2,
            search_scope="separate",
        )

        records = json.loads(plugin.top_k_matches)
        assert "current" in records
        assert "past" in records
        assert len(records["current"]) == 2
        assert len(records["past"]) == 2

        assert plugin.search_scope_param == "separate"
        assert plugin.past_run_id_param == "parent_xyz"
        assert plugin.num_neurons_current_param == 2
        assert plugin.num_neurons_past_param == 3


class TestCompareKeywordSearchCombined:
    """Tests for search_scope='combined'."""

    @patch(
        "plugins.labeling_evaluation.compare.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.compute_label_embedding_coords"
    )
    @patch(
        "plugins.labeling_evaluation.compare.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.embed_labels"
    )
    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_one_sidebar_with_origin_badges(
        self,
        mock_compare_loader_cls: MagicMock,
        mock_embed_labels: MagicMock,
        mock_compute_coords: MagicMock,
    ) -> None:
        """Verify pooled top-k surfaces as one badged sidebar.

        Args:
            mock_compare_loader_cls: Patched MLflowRunLoader class.
            mock_embed_labels: Patched cache helper.
            mock_compute_coords: Patched UMAP helper.
        """
        mock_compare_loader_cls.return_value = _make_past_loader(
            context={"neuron_labeling": {"run_id": "nl_past"}},
            past_neuron_labels={"7": "past_a", "8": "past_b", "9": "past_c"},
        )
        mock_embed_labels.return_value = _SHARED_EMBEDDINGS
        mock_compute_coords.return_value = _SHARED_COORDS

        plugin = _build_plugin()
        plugin.run(
            past_run_id="parent_xyz",
            keyword="search me",
            k=3,
            search_scope="combined",
        )

        sidebars = plugin._sidebars
        assert len(sidebars) == 1
        items = sidebars[0].items
        assert len(items) == 3

        badges = {item.badge for item in items}
        assert badges == {"current", "past"}

        # Pooled top-3 ranks: current_a (1.0), past_a (1.0), past_c (0.5).
        # Sidebar items are sorted by similarity desc.
        sims = [item.similarity for item in items]
        assert sims == sorted(sims, reverse=True)
        assert sims[-1] == pytest.approx(0.5)

    @patch(
        "plugins.labeling_evaluation.compare.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.compute_label_embedding_coords"
    )
    @patch(
        "plugins.labeling_evaluation.compare.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.embed_labels"
    )
    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_combined_items_target_origin_specific_traces(
        self,
        mock_compare_loader_cls: MagicMock,
        mock_embed_labels: MagicMock,
        mock_compute_coords: MagicMock,
    ) -> None:
        """Verify each pooled-sidebar item points at its origin's highlight trace.

        Args:
            mock_compare_loader_cls: Patched MLflowRunLoader class.
            mock_embed_labels: Patched cache helper.
            mock_compute_coords: Patched UMAP helper.
        """
        mock_compare_loader_cls.return_value = _make_past_loader(
            context={"neuron_labeling": {"run_id": "nl_past"}},
            past_neuron_labels={"7": "past_a", "8": "past_b", "9": "past_c"},
        )
        mock_embed_labels.return_value = _SHARED_EMBEDDINGS
        mock_compute_coords.return_value = _SHARED_COORDS

        plugin = _build_plugin()
        plugin.run(
            past_run_id="parent_xyz",
            keyword="search me",
            k=3,
            search_scope="combined",
        )

        for item in plugin._sidebars[0].items:
            if item.badge == "current":
                assert item.trace_index == 2
            elif item.badge == "past":
                assert item.trace_index == 3


class TestCompareKeywordSearchValidation:
    """Tests for compare-mode input validation."""

    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_empty_keyword_raises(self, mock_loader_cls: MagicMock) -> None:  # noqa: ARG002
        """Verify blank keyword is rejected before any past-side load."""
        plugin = _build_plugin()
        with pytest.raises(ValueError, match="keyword must not be empty"):
            plugin.run(past_run_id="parent_xyz", keyword="")

    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_unknown_search_scope_raises(
        self,
        mock_loader_cls: MagicMock,  # noqa: ARG002
    ) -> None:
        """Verify an unrecognised search_scope is rejected with a clear error."""
        plugin = _build_plugin()
        with pytest.raises(ValueError, match="search_scope must be"):
            plugin.run(
                past_run_id="parent_xyz",
                keyword="x",
                search_scope="bogus",  # type: ignore[arg-type]
            )


class TestCompareKeywordSearchHints:
    """Tests for the auto-generated UI hint set."""

    def test_past_runs_dropdown_hint_present(self) -> None:
        """Verify BaseComparePlugin injected a PastRunsDropdownHint for past_run_id."""
        from plugins.labeling_evaluation.compare.embedding_map_with_keyword_search.embedding_map_with_keyword_search import (  # noqa: E501
            Plugin,
        )
        from plugins.plugin_interface import PastRunsDropdownHint

        hints = Plugin.io_spec.param_ui_hints
        past_hints = [
            h
            for h in hints
            if isinstance(h, PastRunsDropdownHint) and h.param_name == "past_run_id"
        ]
        assert len(past_hints) == 1
        assert past_hints[0].required_steps == ["neuron_labeling"]
