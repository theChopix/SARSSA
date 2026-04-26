import polars as pl

from plugins.dataset_loading._dataset_loader import DatasetLoader
from plugins.plugin_interface import (
    BasePlugin,
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
)
from utils.plugin_logger import get_logger

logger = get_logger(__name__)


class LastFm1kLoader(DatasetLoader):
    MIN_USER_INTERACTIONS: int = 5
    MIN_ITEM_INTERACTIONS: int = 10

    RATINGS_FILE: str = "ratings.tsv"

    def __init__(self, data_dir: str = "../data/lastFm1k") -> None:
        super().__init__("LastFM1k", data_dir)

    def load_ratings(self) -> None:
        if not self._file_exists(self.RATINGS_FILE):
            raise FileNotFoundError(
                f"Ratings file not found: {self._resolve_path(self.RATINGS_FILE)}. "
                "Download the dataset first."
            )
        skiprows = [
            2120260 - 1,
            2446318 - 1,
            11141081 - 1,
            11152099 - 1,
            11152402 - 1,
            11882087 - 1,
            12902539 - 1,
            12935044 - 1,
            17589539 - 1,
        ]

        self.df_interactions = (
            pl.scan_csv(
                self._resolve_path(self.RATINGS_FILE),
                separator="\t",
                has_header=False,
                quote_char=None,
            )
            .rename({"column_1": "userId", "column_3": "itemId"})
            .select(["userId", "itemId"])
            .with_row_index()
            .filter(~pl.col("index").is_in(skiprows))
            .drop("index")  # skip damaged rows
            .cast({"userId": pl.String, "itemId": pl.String})
            .unique()
            .sort(["userId", "itemId"])
            .collect()
        )


class Plugin(BasePlugin):
    name = "LastFM-1K Loader"

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
        """Load and prepare the LastFM1k dataset.

        Creates train/val/test splits and populates output attributes
        for ``update_context()`` to log to MLflow.

        Args:
            seed: Random seed for reproducibility.
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
        """
        logger.info("=" * 50)
        logger.info("Starting LastFM1k dataset loading")
        logger.info("=" * 50)

        dataset_loader = LastFm1kLoader()

        logger.info(
            f"Preparing dataset with seed={seed}, val_ratio={val_ratio}, test_ratio={test_ratio}"
        )
        dataset_loader.prepare(
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # Populate output params
        self.dataset_name = "LastFM1k"
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

        logger.info("=" * 50)
        logger.info("LastFM1k dataset loading completed")
        logger.info(f"Users: {self.num_users}, Items: {self.num_items}")
        logger.info(
            f"Train: {self.num_train_users}, "
            f"Valid: {self.num_valid_users}, "
            f"Test: {self.num_test_users}"
        )
        logger.info("=" * 50)
