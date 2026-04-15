import tempfile

import mlflow
import numpy as np
import polars as pl
import scipy.sparse as sp

from plugins.dataset_loading._dataset_loader import DatasetLoader
from plugins.plugin_interface import BasePlugin
from utils.plugin_logger import get_logger

logger = get_logger(__name__)


class LastFm1kLoader(DatasetLoader):
    MIN_USER_INTERACTIONS: int = 5
    MIN_ITEM_INTERACTIONS: int = 10

    def __init__(self, ratings_file_path: str = "../data/LastFm1k.tsv"):
        super().__init__("LastFM1k", ratings_file_path)

    def _load_ratings(self, ratings_file_path: str) -> None:
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
            pl.scan_csv(ratings_file_path, separator="\t", has_header=False, quote_char=None)
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

    def _load_tags(self, tags_file_path: str, items: np.ndarray) -> None:
        raise NotImplementedError("This dataset does not provide tag metadata.")


class Plugin(BasePlugin):
    name = "LastFM-1K Loader"

    def run(
        self,
        context: dict,  # noqa: ARG002
        seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        """
        Dataset loading plugin for LastFM1k.

        Loads and prepares the LastFM1k dataset, creating train/val/test splits
        and storing all necessary artifacts to MLflow for downstream plugins.

        Args:
            context: Plugin context dictionary
            seed: Random seed for reproducibility
            val_ratio: Validation set ratio (default 0.1)
            test_ratio: Test set ratio (default 0.1)
        """
        logger.info("=" * 50)
        logger.info("Starting LastFM1k dataset loading")
        logger.info("=" * 50)

        # Initialize dataset loader
        dataset_loader = LastFm1kLoader()

        # Prepare dataset with train/val/test splits
        logger.info(
            f"Preparing dataset with seed={seed}, val_ratio={val_ratio}, test_ratio={test_ratio}"
        )
        dataset_loader.prepare(
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # Log parameters
        mlflow.log_params(
            {
                "dataset_name": "LastFM1k",
                "seed": seed,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "min_user_interactions": dataset_loader.MIN_USER_INTERACTIONS,
                "min_item_interactions": dataset_loader.MIN_ITEM_INTERACTIONS,
                "num_users": len(dataset_loader.users),
                "num_items": len(dataset_loader.items),
                "num_interactions": dataset_loader.csr_interactions.nnz,
                "num_train_users": len(dataset_loader.train_users),
                "num_valid_users": len(dataset_loader.valid_users),
                "num_test_users": len(dataset_loader.test_users),
                "has_tags": dataset_loader.has_tags(),
            }
        )

        # Store artifacts
        with tempfile.TemporaryDirectory() as tmp:
            logger.info("Saving dataset artifacts...")

            # Core data
            np.save(f"{tmp}/users.npy", dataset_loader.users)
            np.save(f"{tmp}/items.npy", dataset_loader.items)

            # Full dataset (for neuron_labeling)
            sp.save_npz(f"{tmp}/full_csr.npz", dataset_loader.csr_interactions)

            # Split datasets (for training plugins)
            sp.save_npz(f"{tmp}/train_csr.npz", dataset_loader.train_csr)
            sp.save_npz(f"{tmp}/valid_csr.npz", dataset_loader.valid_csr)
            sp.save_npz(f"{tmp}/test_csr.npz", dataset_loader.test_csr)

            # Split indices (for reproducibility)
            np.save(f"{tmp}/train_idx.npy", dataset_loader.train_idx)
            np.save(f"{tmp}/valid_idx.npy", dataset_loader.valid_idx)
            np.save(f"{tmp}/test_idx.npy", dataset_loader.test_idx)

            np.save(f"{tmp}/train_users.npy", dataset_loader.train_users)
            np.save(f"{tmp}/valid_users.npy", dataset_loader.valid_users)
            np.save(f"{tmp}/test_users.npy", dataset_loader.test_users)

            # Log all artifacts
            mlflow.log_artifacts(tmp)
            logger.info("All artifacts saved to MLflow")

        logger.info("=" * 50)
        logger.info("LastFM1k dataset loading completed")
        logger.info(f"Users: {len(dataset_loader.users)}, Items: {len(dataset_loader.items)}")
        logger.info(
            f"Train: {len(dataset_loader.train_users)}, Valid: {len(dataset_loader.valid_users)}, Test: {len(dataset_loader.test_users)}"
        )
        logger.info("=" * 50)
