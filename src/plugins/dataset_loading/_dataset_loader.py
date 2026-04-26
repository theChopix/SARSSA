import logging
import os
import time
from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp


class DatasetLoader:
    # Constants (subclasses override)
    MIN_USER_INTERACTIONS: int = 20
    MIN_ITEM_INTERACTIONS: int = 200

    def __init__(self, name: str, data_dir: str) -> None:
        self.name = name
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)

        # Core interaction data
        self.df_interactions: pl.DataFrame | None = None
        self.csr_interactions: sp.csr_matrix | None = None
        self.users: np.ndarray | None = None
        self.items: np.ndarray | None = None

        # Train/val/test splits
        self.train_csr: sp.csr_matrix | None = None
        self.valid_csr: sp.csr_matrix | None = None
        self.test_csr: sp.csr_matrix | None = None
        self.train_users: np.ndarray | None = None
        self.valid_users: np.ndarray | None = None
        self.test_users: np.ndarray | None = None
        self.train_idx: np.ndarray | None = None
        self.valid_idx: np.ndarray | None = None
        self.test_idx: np.ndarray | None = None

        # Optional data (populated by load_optional_data)
        self.df_tags: pl.DataFrame | None = None
        self.descriptions: dict[str, dict] | None = None
        self.metadata: dict[str, dict] | None = None

        # Prepare params (stored for to_artifacts)
        self._seed: int | None = None
        self._val_ratio: float | None = None
        self._test_ratio: float | None = None

    def _resolve_path(self, filename: str) -> str:
        """Resolve a canonical filename relative to data_dir."""
        return os.path.join(self.data_dir, filename)

    def _file_exists(self, filename: str) -> bool:
        """Check whether a canonical file exists in data_dir."""
        return os.path.exists(self._resolve_path(filename))

    def prepare(self, val_ratio: float, test_ratio: float, seed: int) -> None:
        """Prepare all data for processing."""
        self._seed = seed
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio

        self.logger.info("Preparing dataset: %s", self.name)
        start = time.time()

        self.logger.info("Loading dataset...")
        self.load_ratings()
        load_end = time.time()
        self.logger.info(f"Dataset loaded in {load_end - start:.2f}s")

        self.validate_interactions()

        self.logger.info("Filtering dataset...")
        self.filter_interactions()
        filter_end = time.time()
        self.logger.info(f"Dataset filtered in {filter_end - load_end:.2f}s")

        self.logger.info(
            "Final interactions: %d, users: %d, items: %d",
            len(self.df_interactions),
            len(self.df_interactions["userId"].unique()),
            len(self.df_interactions["itemId"].unique()),
        )

        self.logger.info("Creating csr_matrix...")
        self.build_csr_matrix()
        csr_end = time.time()
        self.logger.info(f"csr_matrix created in {csr_end - filter_end:.2f}s")

        self.logger.info("Splitting dataset...")
        self.split(val_ratio, test_ratio, seed=seed)
        split_end = time.time()
        self.logger.info(f"Dataset split in {split_end - csr_end:.2f}s")

        self.logger.info("-" * 20)
        self.logger.info(f"Dataset prepared in {split_end - start:.2f}s")

        self.load_optional_data()

    @abstractmethod
    def load_ratings(self) -> None:
        """Load interactions into self.df_interactions.

        Must produce a pl.DataFrame with columns ``userId`` and ``itemId``.
        Subclasses should raise FileNotFoundError if the data file is missing.
        """
        ...

    def validate_interactions(self) -> None:
        """Validate that df_interactions is correctly loaded."""
        if not isinstance(self.df_interactions, pl.DataFrame) or any(
            col not in self.df_interactions.columns for col in ["userId", "itemId"]
        ):
            raise ValueError(
                "df_interactions must be a pl.DataFrame with columns userId and itemId"
            )

    def filter_interactions(self) -> None:
        """Filter users and items that have too few interactions."""
        df = self.df_interactions
        self.logger.info(
            "Initial interactions: %d, users: %d, items: %d",
            len(df),
            len(df["userId"].unique()),
            len(df["itemId"].unique()),
        )

        # convert columns to categorical
        df = df.cast({"userId": pl.String, "itemId": pl.String}).cast(
            {"userId": pl.Categorical, "itemId": pl.Categorical}
        )

        # filter users with too few interactions
        df = df.filter(
            df["userId"].is_in(
                df["userId"]
                .value_counts()
                .filter(pl.col("count") >= self.MIN_USER_INTERACTIONS)["userId"]
            ),
        )
        # filter items with too few interactions
        df = df.filter(
            df["itemId"].is_in(
                df["itemId"]
                .value_counts()
                .filter(pl.col("count") >= self.MIN_ITEM_INTERACTIONS)["itemId"]
            ),
        )

        # reset categories
        df = df.cast({"userId": pl.String, "itemId": pl.String}).cast(
            {"userId": pl.Categorical, "itemId": pl.Categorical}
        )

        self.logger.info(
            "Filtered interactions: %d, users: %d, items: %d",
            len(df),
            len(df["userId"].unique()),
            len(df["itemId"].unique()),
        )

        self.df_interactions = df

    def build_csr_matrix(self) -> None:
        """Create a csr_matrix from the interactions DataFrame."""
        self.users = self.df_interactions["userId"].cat.get_categories().to_numpy()
        self.items = self.df_interactions["itemId"].cat.get_categories().to_numpy()

        self.csr_interactions = sp.csr_matrix(
            (
                np.ones(len(self.df_interactions), dtype=np.float32),
                (
                    self.df_interactions["userId"].to_physical().to_numpy(),
                    self.df_interactions["itemId"].to_physical().to_numpy(),
                ),
            ),
            shape=(len(self.users), len(self.items)),
        )

    def split(self, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> None:
        """Split the dataset into train, validation and test sets."""
        np.random.seed(seed)
        p = np.random.permutation(len(self.users))
        val_count, test_count = int(len(self.users) * val_ratio), int(len(self.users) * test_ratio)
        train_idx, val_idx, test_idx = (
            p[val_count + test_count :],
            p[:val_count],
            p[val_count : val_count + test_count],
        )

        self.train_users, self.train_idx, self.train_csr = (
            self.users[train_idx],
            train_idx,
            self.csr_interactions[train_idx],
        )
        self.valid_users, self.valid_idx, self.valid_csr = (
            self.users[val_idx],
            val_idx,
            self.csr_interactions[val_idx],
        )
        self.test_users, self.test_idx, self.test_csr = (
            self.users[test_idx],
            test_idx,
            self.csr_interactions[test_idx],
        )

    def load_optional_data(self) -> None:
        """Hook for subclasses to load tags, descriptions, metadata, etc.

        Called at the end of :meth:`prepare`. Default implementation is a no-op.
        """

    def get_item_metadata(self) -> dict[str, dict[str, Any]]:
        """Return display-ready metadata for items in the dataset.

        Subclasses that have item metadata (title, year, genres,
        image_url, etc.) should override this method and return a
        dict keyed by item ID.  The default implementation returns
        an empty dict, meaning no visual metadata is available.

        Returns:
            dict[str, dict[str, Any]]: Mapping of item ID to
                metadata fields.
        """
        return {}

    def to_artifacts(self) -> dict[str, Any]:
        """Return all outputs as a flat dict for plugin attribute population."""
        return {
            # Artifacts
            "users": self.users,
            "items": self.items,
            "full_csr": self.csr_interactions,
            "train_csr": self.train_csr,
            "valid_csr": self.valid_csr,
            "test_csr": self.test_csr,
            "train_idx": self.train_idx,
            "valid_idx": self.valid_idx,
            "test_idx": self.test_idx,
            "train_users": self.train_users,
            "valid_users": self.valid_users,
            "test_users": self.test_users,
            # Params
            "dataset_name": self.name,
            "seed": self._seed,
            "val_ratio": self._val_ratio,
            "test_ratio": self._test_ratio,
            "num_users": len(self.users),
            "num_items": len(self.items),
            "num_interactions": self.csr_interactions.nnz,
            "num_train_users": len(self.train_users),
            "num_valid_users": len(self.valid_users),
            "num_test_users": len(self.test_users),
            "min_user_interactions": self.MIN_USER_INTERACTIONS,
            "min_item_interactions": self.MIN_ITEM_INTERACTIONS,
            "has_tags": self.df_tags is not None,
        }
