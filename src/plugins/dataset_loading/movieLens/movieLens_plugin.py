import tempfile
import json
import numpy as np
import scipy.sparse as sp
import mlflow

from plugins.plugin_interface import BasePlugin
from plugins.dataset_loading.movieLens.movieLens_loader import MovieLensLoader
from utils.plugin_logger import get_logger

logger = get_logger(__name__)


class Plugin(BasePlugin):
    def run(
        self,
        context: dict,
        seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        """
        Dataset loading plugin for MovieLens.
        
        Loads and prepares the MovieLens dataset, creating train/val/test splits
        and storing all necessary artifacts to MLflow for downstream plugins.
        
        Args:
            context: Plugin context dictionary
            seed: Random seed for reproducibility
            val_ratio: Validation set ratio (default 0.1)
            test_ratio: Test set ratio (default 0.1)
        """
        logger.info("="*50)
        logger.info("Starting MovieLens dataset loading")
        logger.info("="*50)
        
        # Initialize dataset loader
        dataset_loader = MovieLensLoader()
        
        # Prepare dataset with train/val/test splits
        logger.info(f"Preparing dataset with seed={seed}, val_ratio={val_ratio}, test_ratio={test_ratio}")
        dataset_loader.prepare(
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        
        # Log parameters
        mlflow.log_params({
            "dataset_name": "MovieLens",
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
        })
        
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
            
            # Tags (if available)
            if dataset_loader.has_tags():
                logger.info("Saving tag metadata...")
                tag_ids = dataset_loader.tag_ids()
                tag_item_matrix = dataset_loader.tag_item_matrix()
                
                with open(f"{tmp}/tag_ids.json", "w") as f:
                    json.dump(tag_ids, f, indent=2)
                
                sp.save_npz(f"{tmp}/tag_item_matrix.npz", tag_item_matrix)
                
                mlflow.log_param("num_tags", len(tag_ids))
            
            # Log all artifacts
            mlflow.log_artifacts(tmp)
            logger.info("All artifacts saved to MLflow")
        
        logger.info("="*50)
        logger.info("MovieLens dataset loading completed")
        logger.info(f"Users: {len(dataset_loader.users)}, Items: {len(dataset_loader.items)}")
        logger.info(f"Train: {len(dataset_loader.train_users)}, Valid: {len(dataset_loader.valid_users)}, Test: {len(dataset_loader.test_users)}")
        logger.info("="*50)
        
        return context
