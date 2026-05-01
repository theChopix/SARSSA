"""Unit tests for app.core.item_enrichment."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.item_enrichment.item_enrichment import (
    enrich_items,
    load_item_metadata,
    load_step_artifact,
)

# ── load_item_metadata ─────────────────────────────────────────────


class TestLoadItemMetadata:
    """Tests for load_item_metadata."""

    def setup_method(self) -> None:
        """Clear the LRU cache before each test."""
        load_item_metadata.cache_clear()

    @patch("app.core.item_enrichment.item_enrichment.MLflowRunLoader")
    def test_returns_metadata_when_artifact_exists(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify metadata is returned when artifact exists."""
        mock_loader = MagicMock()
        mock_loader.artifact_exists.return_value = True
        mock_loader.get_json_artifact.return_value = {
            "1": {"title": "Movie A"},
        }
        mock_loader_cls.return_value = mock_loader

        result = load_item_metadata("run_123")

        assert result == {"1": {"title": "Movie A"}}
        mock_loader.get_json_artifact.assert_called_once_with(
            "item_metadata.json",
        )

    @patch("app.core.item_enrichment.item_enrichment.MLflowRunLoader")
    def test_returns_empty_dict_when_artifact_missing(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify empty dict when artifact does not exist."""
        mock_loader = MagicMock()
        mock_loader.artifact_exists.return_value = False
        mock_loader_cls.return_value = mock_loader

        result = load_item_metadata("run_missing")

        assert result == {}
        mock_loader.get_json_artifact.assert_not_called()

    @patch("app.core.item_enrichment.item_enrichment.MLflowRunLoader")
    def test_caches_result_for_same_run_id(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify same run_id only loads once (LRU cache)."""
        mock_loader = MagicMock()
        mock_loader.artifact_exists.return_value = True
        mock_loader.get_json_artifact.return_value = {"1": {"title": "X"}}
        mock_loader_cls.return_value = mock_loader

        load_item_metadata("run_cached")
        load_item_metadata("run_cached")

        assert mock_loader_cls.call_count == 1


# ── enrich_items ───────────────────────────────────────────────────


class TestEnrichItems:
    """Tests for enrich_items."""

    def setup_method(self) -> None:
        """Clear the LRU cache before each test."""
        load_item_metadata.cache_clear()

    @patch("app.core.item_enrichment.item_enrichment.load_item_metadata")
    def test_enriches_with_metadata(
        self,
        mock_load: MagicMock,
    ) -> None:
        """Verify items are enriched when metadata is available."""
        mock_load.return_value = {
            "10": {"title": "Movie A", "year": 2020},
            "20": {"title": "Movie B", "year": 2021},
        }

        items, available = enrich_items("run_1", ["10", "20"])

        assert available is True
        assert len(items) == 2
        assert items[0] == {"id": "10", "title": "Movie A", "year": 2020}
        assert items[1] == {"id": "20", "title": "Movie B", "year": 2021}

    @patch("app.core.item_enrichment.item_enrichment.load_item_metadata")
    def test_falls_back_when_metadata_missing(
        self,
        mock_load: MagicMock,
    ) -> None:
        """Verify fallback when metadata artifact is absent."""
        mock_load.return_value = {}

        items, available = enrich_items("run_no_meta", ["42", "99"])

        assert available is False
        assert items == [
            {"id": "42", "title": "42"},
            {"id": "99", "title": "99"},
        ]

    @patch("app.core.item_enrichment.item_enrichment.load_item_metadata")
    def test_partial_metadata(
        self,
        mock_load: MagicMock,
    ) -> None:
        """Verify mix of enriched and fallback items."""
        mock_load.return_value = {
            "10": {"title": "Known Movie"},
        }

        items, available = enrich_items("run_2", ["10", "999"])

        assert available is True
        assert items[0] == {"id": "10", "title": "Known Movie"}
        assert items[1] == {"id": "999", "title": "999"}

    @patch("app.core.item_enrichment.item_enrichment.load_item_metadata")
    def test_empty_ids_returns_empty_list(
        self,
        mock_load: MagicMock,
    ) -> None:
        """Verify empty input produces empty output."""
        mock_load.return_value = {"1": {"title": "X"}}

        items, available = enrich_items("run_3", [])

        assert items == []
        assert available is True

    @patch("app.core.item_enrichment.item_enrichment.load_item_metadata")
    def test_preserves_order(
        self,
        mock_load: MagicMock,
    ) -> None:
        """Verify output order matches input order."""
        mock_load.return_value = {
            "a": {"title": "A"},
            "b": {"title": "B"},
            "c": {"title": "C"},
        }

        items, _ = enrich_items("run_4", ["c", "a", "b"])

        assert [i["id"] for i in items] == ["c", "a", "b"]


# ── load_step_artifact ────────────────────────────────────────────


class TestLoadStepArtifact:
    """Tests for load_step_artifact."""

    @patch("app.core.item_enrichment.item_enrichment.MLflowRunLoader")
    def test_returns_artifact_content(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify artifact JSON content is returned."""
        mock_loader = MagicMock()
        mock_loader.artifact_exists.return_value = True
        mock_loader.get_json_artifact.return_value = ["42", "107"]
        mock_loader_cls.return_value = mock_loader

        result = load_step_artifact("run_1", "recs.json")

        assert result == ["42", "107"]
        mock_loader.get_json_artifact.assert_called_once_with("recs.json")

    @patch("app.core.item_enrichment.item_enrichment.MLflowRunLoader")
    def test_raises_when_artifact_missing(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify FileNotFoundError when artifact does not exist."""
        mock_loader = MagicMock()
        mock_loader.artifact_exists.return_value = False
        mock_loader_cls.return_value = mock_loader

        with pytest.raises(FileNotFoundError, match="recs.json"):
            load_step_artifact("run_bad", "recs.json")
