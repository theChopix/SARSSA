import json
import tempfile

import mlflow
import numpy as np
import polars as pl
import scipy.sparse as sp

from plugins.dataset_loading._dataset_loader import DatasetLoader
from plugins.plugin_interface import (
    BasePlugin,
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
)
from utils.plugin_logger import get_logger

logger = get_logger(__name__)


class MovieLensLoader(DatasetLoader):
    MIN_USER_INTERACTIONS: int = 50
    MIN_ITEM_INTERACTIONS: int = 20

    RATINGS_FILE: str = "ratings.csv"
    TAGS_FILE: str = "tags.csv"
    DESCRIPTIONS_FILE: str = "descriptions.json"
    METADATA_FILE: str = "metadata.json"

    def __init__(self, data_dir: str = "../data/movieLens") -> None:
        super().__init__("MovieLens", data_dir)

    def load_ratings(self) -> None:
        if not self._file_exists(self.RATINGS_FILE):
            raise FileNotFoundError(
                f"Ratings file not found: {self._resolve_path(self.RATINGS_FILE)}. "
                "Download the dataset first."
            )
        self.df_interactions = (
            pl.scan_csv(self._resolve_path(self.RATINGS_FILE), has_header=True)
            .select(["userId", "movieId", "rating"])
            .rename({"movieId": "itemId"})
            .cast({"userId": pl.String, "itemId": pl.String, "rating": pl.Float64})
            .filter(pl.col("rating") >= 4.0)
            .select(["userId", "itemId"])
            .unique()
            .sort(["userId", "itemId"])
            .collect()
        )

    def load_optional_data(self) -> None:
        # Tags
        if self._file_exists(self.TAGS_FILE):
            self.logger.info("Loading tags...")
            self.df_tags = (
                pl.scan_csv(self._resolve_path(self.TAGS_FILE), has_header=True)
                .select(["movieId", "tag"])
                .rename({"movieId": "itemId"})
                .cast({"itemId": pl.String, "tag": pl.String})
                .with_columns(pl.col("tag").str.to_lowercase().str.strip_chars().alias("tag"))
                .filter(pl.col("itemId").is_in(self.items))
                .unique()
                .collect()
            )
            self.logger.info("Loaded %d tag entries", len(self.df_tags))

        # Descriptions
        if self._file_exists(self.DESCRIPTIONS_FILE):
            with open(self._resolve_path(self.DESCRIPTIONS_FILE)) as f:
                self.descriptions = json.load(f)
            self.logger.info("Loaded descriptions for %d items", len(self.descriptions))

        # Metadata
        if self._file_exists(self.METADATA_FILE):
            with open(self._resolve_path(self.METADATA_FILE)) as f:
                self.metadata = json.load(f)
            self.logger.info("Loaded metadata for %d items", len(self.metadata))

    def get_item_metadata(self) -> dict[str, dict]:
        """Return metadata filtered to items in the current dataset.

        Uses the ``metadata.json`` loaded in :meth:`load_optional_data`.
        Only items present in ``self.items`` (the filtered item set)
        are included.

        Returns:
            dict[str, dict]: Mapping of item ID to metadata fields
                (``title``, ``year``, ``genres``, ``image_url``).
                Empty dict if no metadata was loaded.
        """
        if self.metadata is None or self.items is None:
            return {}
        item_set = set(self.items)
        return {item_id: meta for item_id, meta in self.metadata.items() if item_id in item_set}

    def tag_ids(self):
        return self.df_tags["tag"].unique().sort().to_list()

    def tag_item_matrix(self):
        tag_ids = self.tag_ids()
        tag_to_idx = {t: i for i, t in enumerate(tag_ids)}
        item_to_idx = {i: idx for idx, i in enumerate(self.items)}

        rows, cols = [], []

        for row in self.df_tags.iter_rows(named=True):
            rows.append(tag_to_idx[row["tag"]])
            cols.append(item_to_idx[row["itemId"]])

        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix((data, (rows, cols)), shape=(len(tag_ids), len(self.items)))


class Plugin(BasePlugin):
    name = "MovieLens Loader"

    io_spec = PluginIOSpec(
        output_artifacts=[
            OutputArtifactSpec("users", "users.npy", "npy"),
            OutputArtifactSpec("items", "items.npy", "npy"),
            OutputArtifactSpec("full_csr", "full_csr.npz", "npz"),
            OutputArtifactSpec("train_csr", "train_csr.npz", "npz"),
            OutputArtifactSpec("valid_csr", "valid_csr.npz", "npz"),
            OutputArtifactSpec("test_csr", "test_csr.npz", "npz"),
            OutputArtifactSpec("train_idx", "train_idx.npy", "npy"),
            OutputArtifactSpec("valid_idx", "valid_idx.npy", "npy"),
            OutputArtifactSpec("test_idx", "test_idx.npy", "npy"),
            OutputArtifactSpec("train_users", "train_users.npy", "npy"),
            OutputArtifactSpec("valid_users", "valid_users.npy", "npy"),
            OutputArtifactSpec("test_users", "test_users.npy", "npy"),
        ],
        output_params=[
            OutputParamSpec("dataset_name", "dataset_name"),
            OutputParamSpec("seed", "seed"),
            OutputParamSpec("val_ratio", "val_ratio"),
            OutputParamSpec("test_ratio", "test_ratio"),
            OutputParamSpec("min_user_interactions", "min_user_interactions"),
            OutputParamSpec("min_item_interactions", "min_item_interactions"),
            OutputParamSpec("num_users", "num_users"),
            OutputParamSpec("num_items", "num_items"),
            OutputParamSpec("num_interactions", "num_interactions"),
            OutputParamSpec("num_train_users", "num_train_users"),
            OutputParamSpec("num_valid_users", "num_valid_users"),
            OutputParamSpec("num_test_users", "num_test_users"),
            OutputParamSpec("has_tags", "has_tags"),
        ],
    )

    def run(
        self,
        seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        """Load and prepare the MovieLens dataset.

        Creates train/val/test splits and populates output attributes
        for ``update_context()`` to log to MLflow. Tag artifacts are
        stored on ``self`` for the ``update_context`` override.

        Args:
            seed: Random seed for reproducibility.
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
        """
        logger.info("=" * 50)
        logger.info("Starting MovieLens dataset loading")
        logger.info("=" * 50)

        dataset_loader = MovieLensLoader()

        logger.info(
            f"Preparing dataset with seed={seed}, val_ratio={val_ratio}, test_ratio={test_ratio}"
        )
        dataset_loader.prepare(
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # Populate output params
        self.dataset_name = "MovieLens"
        self.seed = seed
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_user_interactions = dataset_loader.MIN_USER_INTERACTIONS
        self.min_item_interactions = dataset_loader.MIN_ITEM_INTERACTIONS
        self.num_users = len(dataset_loader.users)
        self.num_items = len(dataset_loader.items)
        self.num_interactions = dataset_loader.csr_interactions.nnz
        self.num_train_users = len(dataset_loader.train_users)
        self.num_valid_users = len(dataset_loader.valid_users)
        self.num_test_users = len(dataset_loader.test_users)
        self.has_tags = dataset_loader.df_tags is not None

        # Populate output artifacts
        self.users = dataset_loader.users
        self.items = dataset_loader.items
        self.full_csr = dataset_loader.csr_interactions
        self.train_csr = dataset_loader.train_csr
        self.valid_csr = dataset_loader.valid_csr
        self.test_csr = dataset_loader.test_csr
        self.train_idx = dataset_loader.train_idx
        self.valid_idx = dataset_loader.valid_idx
        self.test_idx = dataset_loader.test_idx
        self.train_users = dataset_loader.train_users
        self.valid_users = dataset_loader.valid_users
        self.test_users = dataset_loader.test_users

        # Tag outputs (conditional, handled in update_context)
        self._tag_ids: list[str] | None = None
        self._tag_item_matrix: sp.csr_matrix | None = None
        if dataset_loader.df_tags is not None:
            self._tag_ids = dataset_loader.tag_ids()
            self._tag_item_matrix = dataset_loader.tag_item_matrix()

        # Item metadata (conditional, handled in update_context)
        self._item_metadata = dataset_loader.get_item_metadata()

        logger.info("=" * 50)
        logger.info("MovieLens dataset loading completed")
        logger.info(f"Users: {self.num_users}, Items: {self.num_items}")
        logger.info(
            f"Train: {self.num_train_users}, "
            f"Valid: {self.num_valid_users}, "
            f"Test: {self.num_test_users}"
        )
        logger.info("=" * 50)

    def update_context(self) -> None:
        """Log outputs to MLflow, including conditional tag artifacts.

        Calls the base ``update_context()`` for common outputs, then
        manually logs tag-specific artifacts and parameters if the
        dataset contains tags.
        """
        super().update_context()

        if self._tag_ids is not None and self._tag_item_matrix is not None:
            logger.info("Saving tag metadata...")
            with tempfile.TemporaryDirectory() as tmp:
                with open(f"{tmp}/tag_ids.json", "w") as f:
                    json.dump(self._tag_ids, f, indent=2)
                sp.save_npz(f"{tmp}/tag_item_matrix.npz", self._tag_item_matrix)
                mlflow.log_artifacts(tmp)
            mlflow.log_param("num_tags", len(self._tag_ids))

        if self._item_metadata:
            logger.info("Saving item metadata (%d items)...", len(self._item_metadata))
            with tempfile.TemporaryDirectory() as tmp:
                with open(f"{tmp}/item_metadata.json", "w") as f:
                    json.dump(self._item_metadata, f)
                mlflow.log_artifacts(tmp)
