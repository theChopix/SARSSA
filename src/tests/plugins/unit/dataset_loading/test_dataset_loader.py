"""Unit tests for DatasetLoader base class.

Tests cover helpers, validation, filtering, CSR construction,
splitting, to_artifacts, and the full prepare pipeline.
"""

import os
from typing import Any

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from plugins.dataset_loading._dataset_loader import DatasetLoader

# ── Concrete stub for testing ───────────────────────────────────────


class _StubLoader(DatasetLoader):
    """Minimal concrete loader with low thresholds for small fixtures."""

    MIN_USER_INTERACTIONS: int = 2
    MIN_ITEM_INTERACTIONS: int = 2

    def __init__(
        self,
        data_dir: str = "/tmp/test_data",
        df: pl.DataFrame | None = None,
    ) -> None:
        super().__init__("StubDataset", data_dir)
        self._fixture_df = df

    def load_ratings(self) -> None:
        """Populate df_interactions from the fixture DataFrame."""
        if self._fixture_df is not None:
            self.df_interactions = self._fixture_df


def _make_interactions(
    rows: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build a minimal interactions DataFrame.

    Args:
        rows: List of (userId, itemId) tuples.

    Returns:
        pl.DataFrame with userId and itemId columns.
    """
    return pl.DataFrame(
        {"userId": [r[0] for r in rows], "itemId": [r[1] for r in rows]},
        schema={"userId": pl.String, "itemId": pl.String},
    )


def _make_filtered_loader(
    rows: list[tuple[str, str]],
) -> _StubLoader:
    """Create a stub loader with interactions already filtered + CSR built.

    Args:
        rows: Interaction tuples that should all survive filtering.

    Returns:
        Loader with filter_interactions, build_csr_matrix, and split done.
    """
    df = _make_interactions(rows)
    loader = _StubLoader(df=df)
    loader.df_interactions = df
    loader.filter_interactions()
    loader.build_csr_matrix()
    return loader


# ── Fixture data ────────────────────────────────────────────────────

# 3 users, 3 items — all have >=2 interactions so all survive filtering
DENSE_ROWS: list[tuple[str, str]] = [
    ("u1", "i1"),
    ("u1", "i2"),
    ("u1", "i3"),
    ("u2", "i1"),
    ("u2", "i2"),
    ("u2", "i3"),
    ("u3", "i1"),
    ("u3", "i2"),
    ("u3", "i3"),
]

# u1 has only 1 interaction → should be filtered out
SPARSE_USER_ROWS: list[tuple[str, str]] = [
    ("u1", "i1"),
    ("u2", "i1"),
    ("u2", "i2"),
    ("u3", "i1"),
    ("u3", "i2"),
]

# i3 has only 1 interaction → should be filtered out
SPARSE_ITEM_ROWS: list[tuple[str, str]] = [
    ("u1", "i1"),
    ("u1", "i2"),
    ("u1", "i3"),
    ("u2", "i1"),
    ("u2", "i2"),
]


# ── _resolve_path / _file_exists ────────────────────────────────────


class TestResolvePathAndFileExists:
    """Tests for path helper methods."""

    def test_resolve_path_joins_data_dir_and_filename(self) -> None:
        """Verify _resolve_path joins data_dir with filename."""
        loader = _StubLoader(data_dir="/data/movieLens")
        assert loader._resolve_path("ratings.csv") == os.path.join("/data/movieLens", "ratings.csv")

    def test_resolve_path_with_nested_filename(self) -> None:
        """Verify _resolve_path works with subdirectory filenames."""
        loader = _StubLoader(data_dir="/data")
        assert loader._resolve_path("raw/links.csv") == os.path.join("/data", "raw/links.csv")

    def test_file_exists_returns_true_for_existing_file(self, tmp_path: Any) -> None:
        """Verify _file_exists returns True for a real file."""
        (tmp_path / "test.csv").write_text("data")
        loader = _StubLoader(data_dir=str(tmp_path))
        assert loader._file_exists("test.csv") is True

    def test_file_exists_returns_false_for_missing_file(self, tmp_path: Any) -> None:
        """Verify _file_exists returns False for a non-existent file."""
        loader = _StubLoader(data_dir=str(tmp_path))
        assert loader._file_exists("nonexistent.csv") is False


# ── validate_interactions ───────────────────────────────────────────


class TestValidateInteractions:
    """Tests for DatasetLoader.validate_interactions."""

    def test_passes_with_valid_dataframe(self) -> None:
        """Verify no error for a DataFrame with userId and itemId."""
        loader = _StubLoader()
        loader.df_interactions = _make_interactions([("u1", "i1")])
        loader.validate_interactions()

    def test_raises_when_df_is_none(self) -> None:
        """Verify ValueError when df_interactions is None."""
        loader = _StubLoader()
        with pytest.raises(ValueError, match="pl.DataFrame"):
            loader.validate_interactions()

    def test_raises_when_column_missing(self) -> None:
        """Verify ValueError when required column is absent."""
        loader = _StubLoader()
        loader.df_interactions = pl.DataFrame({"userId": ["u1"], "wrongCol": ["x"]})
        with pytest.raises(ValueError, match="pl.DataFrame"):
            loader.validate_interactions()

    def test_raises_when_both_columns_missing(self) -> None:
        """Verify ValueError when neither column is present."""
        loader = _StubLoader()
        loader.df_interactions = pl.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="pl.DataFrame"):
            loader.validate_interactions()

    def test_raises_for_non_dataframe(self) -> None:
        """Verify ValueError when df_interactions is not a DataFrame."""
        loader = _StubLoader()
        loader.df_interactions = {"userId": [], "itemId": []}  # type: ignore[assignment]
        with pytest.raises(ValueError, match="pl.DataFrame"):
            loader.validate_interactions()


# ── filter_interactions ─────────────────────────────────────────────


class TestFilterInteractions:
    """Tests for DatasetLoader.filter_interactions."""

    def test_keeps_all_when_above_threshold(self) -> None:
        """Verify no rows removed when all users/items meet threshold."""
        loader = _StubLoader()
        loader.df_interactions = _make_interactions(DENSE_ROWS)
        loader.filter_interactions()

        assert len(loader.df_interactions) == len(DENSE_ROWS)

    def test_removes_sparse_users(self) -> None:
        """Verify users with <MIN_USER_INTERACTIONS are removed."""
        loader = _StubLoader()
        loader.df_interactions = _make_interactions(SPARSE_USER_ROWS)
        loader.filter_interactions()

        remaining_users = loader.df_interactions["userId"].unique().to_list()
        assert "u1" not in remaining_users
        assert "u2" in remaining_users
        assert "u3" in remaining_users

    def test_removes_sparse_items(self) -> None:
        """Verify items with <MIN_ITEM_INTERACTIONS are removed."""
        loader = _StubLoader()
        loader.df_interactions = _make_interactions(SPARSE_ITEM_ROWS)
        loader.filter_interactions()

        remaining_items = loader.df_interactions["itemId"].unique().to_list()
        assert "i3" not in remaining_items
        assert "i1" in remaining_items
        assert "i2" in remaining_items

    def test_output_has_categorical_columns(self) -> None:
        """Verify filtered DataFrame has categorical userId/itemId."""
        loader = _StubLoader()
        loader.df_interactions = _make_interactions(DENSE_ROWS)
        loader.filter_interactions()

        assert loader.df_interactions["userId"].dtype == pl.Categorical
        assert loader.df_interactions["itemId"].dtype == pl.Categorical


# ── build_csr_matrix ────────────────────────────────────────────────


class TestBuildCsrMatrix:
    """Tests for DatasetLoader.build_csr_matrix."""

    def test_shape_matches_users_and_items(self) -> None:
        """Verify CSR matrix shape is (num_users, num_items)."""
        loader = _make_filtered_loader(DENSE_ROWS)
        assert loader.csr_interactions.shape == (3, 3)

    def test_nnz_matches_interactions(self) -> None:
        """Verify number of non-zeros matches interaction count."""
        loader = _make_filtered_loader(DENSE_ROWS)
        assert loader.csr_interactions.nnz == len(DENSE_ROWS)

    def test_values_are_ones(self) -> None:
        """Verify all non-zero values are 1.0."""
        loader = _make_filtered_loader(DENSE_ROWS)
        np.testing.assert_array_equal(
            loader.csr_interactions.data,
            np.ones(len(DENSE_ROWS), dtype=np.float32),
        )

    def test_users_and_items_arrays_populated(self) -> None:
        """Verify users and items arrays have correct length."""
        loader = _make_filtered_loader(DENSE_ROWS)
        assert len(loader.users) == 3
        assert len(loader.items) == 3

    def test_matrix_is_csr(self) -> None:
        """Verify the matrix is a CSR sparse matrix."""
        loader = _make_filtered_loader(DENSE_ROWS)
        assert sp.issparse(loader.csr_interactions)
        assert isinstance(loader.csr_interactions, sp.csr_matrix)


# ── split ───────────────────────────────────────────────────────────


class TestSplit:
    """Tests for DatasetLoader.split."""

    def test_split_sizes(self) -> None:
        """Verify train/val/test sizes sum to total users."""
        loader = _make_filtered_loader(DENSE_ROWS)
        loader.split(val_ratio=0.1, test_ratio=0.1, seed=42)

        total = len(loader.train_users) + len(loader.valid_users) + len(loader.test_users)
        assert total == len(loader.users)

    def test_no_user_overlap(self) -> None:
        """Verify no user appears in multiple splits."""
        loader = _make_filtered_loader(DENSE_ROWS)
        loader.split(val_ratio=0.1, test_ratio=0.1, seed=42)

        assert loader.train_users is not None
        assert loader.valid_users is not None
        assert loader.test_users is not None
        all_split_users = np.concatenate(
            [loader.train_users, loader.valid_users, loader.test_users]
        )
        assert len(all_split_users) == len(set(all_split_users))

    def test_csr_rows_match_indices(self) -> None:
        """Verify split CSR matrices have correct number of rows."""
        loader = _make_filtered_loader(DENSE_ROWS)
        loader.split(val_ratio=0.1, test_ratio=0.1, seed=42)

        assert loader.train_csr.shape[0] == len(loader.train_idx)
        assert loader.valid_csr.shape[0] == len(loader.valid_idx)
        assert loader.test_csr.shape[0] == len(loader.test_idx)

    def test_deterministic_with_same_seed(self) -> None:
        """Verify same seed produces same split."""
        loader1 = _make_filtered_loader(DENSE_ROWS)
        loader1.split(val_ratio=0.1, test_ratio=0.1, seed=123)

        loader2 = _make_filtered_loader(DENSE_ROWS)
        loader2.split(val_ratio=0.1, test_ratio=0.1, seed=123)

        np.testing.assert_array_equal(loader1.train_idx, loader2.train_idx)
        np.testing.assert_array_equal(loader1.valid_idx, loader2.valid_idx)
        np.testing.assert_array_equal(loader1.test_idx, loader2.test_idx)

    def test_different_seed_gives_different_split(self) -> None:
        """Verify different seeds produce different splits."""
        # Use more users so permutation is likely to differ
        rows = [(f"u{i}", f"i{j}") for i in range(10) for j in range(10)]
        loader1 = _make_filtered_loader(rows)
        loader1.split(val_ratio=0.2, test_ratio=0.2, seed=1)

        loader2 = _make_filtered_loader(rows)
        loader2.split(val_ratio=0.2, test_ratio=0.2, seed=99)

        # With 10 users it's astronomically unlikely to get same permutation
        assert not np.array_equal(loader1.train_idx, loader2.train_idx)

    def test_split_columns_match(self) -> None:
        """Verify CSR split columns equal total items."""
        loader = _make_filtered_loader(DENSE_ROWS)
        loader.split(val_ratio=0.1, test_ratio=0.1, seed=42)

        assert loader.train_csr.shape[1] == len(loader.items)
        assert loader.valid_csr.shape[1] == len(loader.items)
        assert loader.test_csr.shape[1] == len(loader.items)


# ── load_optional_data ──────────────────────────────────────────────


class TestLoadOptionalData:
    """Tests for DatasetLoader.load_optional_data default."""

    def test_default_is_noop(self) -> None:
        """Verify default load_optional_data leaves optional attrs as None."""
        loader = _StubLoader()
        loader.load_optional_data()

        assert loader.df_tags is None
        assert loader.descriptions is None
        assert loader.metadata is None


# ── get_item_metadata ──────────────────────────────────────────────


class TestGetItemMetadata:
    """Tests for DatasetLoader.get_item_metadata."""

    def test_default_returns_empty_dict(self) -> None:
        """Verify base class returns empty dict."""
        loader = _StubLoader()
        assert loader.get_item_metadata() == {}

    def test_default_returns_empty_after_prepare(self) -> None:
        """Verify base class still returns empty dict after full prepare."""
        loader = _StubLoader(df=_make_interactions(DENSE_ROWS))
        loader.prepare(val_ratio=0.1, test_ratio=0.1, seed=42)
        assert loader.get_item_metadata() == {}

    def test_subclass_with_metadata_filters_to_items(self) -> None:
        """Verify a subclass with metadata returns only matching items."""

        class _MetadataLoader(_StubLoader):
            def load_optional_data(self) -> None:
                self.metadata = {
                    "i1": {"title": "Item 1"},
                    "i2": {"title": "Item 2"},
                    "i999": {"title": "Not in dataset"},
                }

            def get_item_metadata(self) -> dict[str, dict]:
                if self.metadata is None or self.items is None:
                    return {}
                item_set = set(self.items)
                return {k: v for k, v in self.metadata.items() if k in item_set}

        loader = _MetadataLoader(df=_make_interactions(DENSE_ROWS))
        loader.prepare(val_ratio=0.1, test_ratio=0.1, seed=42)
        meta = loader.get_item_metadata()

        assert "i1" in meta
        assert "i2" in meta
        assert "i999" not in meta
        assert meta["i1"]["title"] == "Item 1"

    def test_subclass_with_no_metadata_returns_empty(self) -> None:
        """Verify subclass returns empty dict when metadata is None."""

        class _NoMetaLoader(_StubLoader):
            def get_item_metadata(self) -> dict[str, dict]:
                if self.metadata is None or self.items is None:
                    return {}
                return {}

        loader = _NoMetaLoader(df=_make_interactions(DENSE_ROWS))
        loader.prepare(val_ratio=0.1, test_ratio=0.1, seed=42)
        assert loader.get_item_metadata() == {}


# ── to_artifacts ────────────────────────────────────────────────────


class TestToArtifacts:
    """Tests for DatasetLoader.to_artifacts."""

    def _prepared_loader(self) -> _StubLoader:
        """Return a fully prepared loader for artifact tests."""
        loader = _StubLoader(df=_make_interactions(DENSE_ROWS))
        loader.prepare(val_ratio=0.1, test_ratio=0.1, seed=42)
        return loader

    def test_has_tags_false_by_default(self) -> None:
        """Verify has_tags is False when no tags loaded."""
        loader = self._prepared_loader()
        arts = loader.to_artifacts()
        assert arts["has_tags"] is False

    def test_has_tags_true_when_tags_set(self) -> None:
        """Verify has_tags is True when df_tags is populated."""
        loader = self._prepared_loader()
        loader.df_tags = pl.DataFrame({"itemId": ["i1"], "tag": ["action"]})
        arts = loader.to_artifacts()
        assert arts["has_tags"] is True


# ── prepare (integration) ──────────────────────────────────────────


class TestPrepare:
    """Integration tests for DatasetLoader.prepare pipeline."""

    def test_raises_on_invalid_interactions(self) -> None:
        """Verify prepare raises when load_ratings produces bad data."""

        class _BadLoader(DatasetLoader):
            MIN_USER_INTERACTIONS = 2
            MIN_ITEM_INTERACTIONS = 2

            def load_ratings(self) -> None:
                self.df_interactions = pl.DataFrame({"wrongCol": ["a"]})

        loader = _BadLoader("BadDataset", "/tmp")
        with pytest.raises(ValueError, match="pl.DataFrame"):
            loader.prepare(val_ratio=0.1, test_ratio=0.1, seed=42)
