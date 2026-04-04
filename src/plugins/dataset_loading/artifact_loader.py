"""
Utility for loading dataset artifacts from MLflow.

This module provides a helper class to load dataset artifacts that were
stored by the dataset_loading plugins (MovieLens or LastFM1k).
"""

import json
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import scipy.sparse as sp


@dataclass
class DatasetArtifacts:
    """Container for loaded dataset artifacts."""

    # Core data
    users: np.ndarray
    items: np.ndarray
    full_csr: sp.csr_matrix

    # Split data
    train_csr: sp.csr_matrix
    valid_csr: sp.csr_matrix
    test_csr: sp.csr_matrix

    # Split indices
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray

    # Split users
    train_users: np.ndarray
    valid_users: np.ndarray
    test_users: np.ndarray

    # Metadata
    metadata: dict[str, Any]

    # Optional: Tags (only for MovieLens)
    tag_ids: list | None = None
    tag_item_matrix: sp.csr_matrix | None = None

    @property
    def dataset_name(self) -> str:
        """Get dataset name from metadata."""
        return self.metadata.get("dataset_name", "Unknown")

    @property
    def num_users(self) -> int:
        """Get number of users."""
        return len(self.users)

    @property
    def num_items(self) -> int:
        """Get number of items."""
        return len(self.items)

    @property
    def has_tags(self) -> bool:
        """Check if tags are available."""
        return self.tag_ids is not None and self.tag_item_matrix is not None


class DatasetArtifactLoader:
    """
    Helper class to load dataset artifacts from MLflow.

    Usage:
        # From context
        loader = DatasetArtifactLoader.from_context(context)
        artifacts = loader.load()

        # From run_id
        loader = DatasetArtifactLoader(run_id="abc123")
        artifacts = loader.load()

        # Load only specific artifacts
        train_csr = loader.load_train_csr()
        full_csr = loader.load_full_csr()
    """

    def __init__(self, run_id: str, artifact_path: str = "dataset"):
        """
        Initialize the artifact loader.

        Args:
            run_id: MLflow run ID containing the dataset artifacts
            artifact_path: Path to artifacts within the run (default: "dataset")
        """
        self.run_id = run_id
        self.artifact_path = artifact_path
        self._artifact_uri = None

    @classmethod
    def from_context(cls, context: dict) -> "DatasetArtifactLoader":
        """
        Create loader from plugin context.

        Args:
            context: Plugin context dictionary containing dataset info

        Returns:
            DatasetArtifactLoader instance

        Raises:
            ValueError: If context doesn't contain dataset information
        """
        if "dataset" not in context:
            raise ValueError(
                "Context does not contain 'dataset' key. "
                "Make sure dataset_loading plugin was run first."
            )

        dataset_info = context["dataset"]
        return cls(
            run_id=dataset_info["run_id"],
            artifact_path=dataset_info.get("artifact_path", "dataset"),
        )

    @property
    def artifact_uri(self) -> str:
        """Get the artifact URI, computing it once and caching."""
        if self._artifact_uri is None:
            run = mlflow.get_run(self.run_id)
            base_uri = run.info.artifact_uri
            # Convert to local path if needed
            if base_uri.startswith("file://"):
                base_uri = base_uri[7:]
            elif "mlruns" in base_uri:
                base_uri = "./" + base_uri[base_uri.find("mlruns") :]
            self._artifact_uri = f"{base_uri}/{self.artifact_path}"
        return self._artifact_uri

    def load_metadata(self) -> dict[str, Any]:
        """Load metadata.json."""
        with open(f"{self.artifact_uri}/metadata.json") as f:
            return json.load(f)

    def load_users(self) -> np.ndarray:
        """Load users array."""
        return np.load(f"{self.artifact_uri}/users.npy")

    def load_items(self) -> np.ndarray:
        """Load items array."""
        return np.load(f"{self.artifact_uri}/items.npy")

    def load_full_csr(self) -> sp.csr_matrix:
        """Load full interaction matrix (for neuron_labeling)."""
        return sp.load_npz(f"{self.artifact_uri}/full_csr.npz")

    def load_train_csr(self) -> sp.csr_matrix:
        """Load training interaction matrix."""
        return sp.load_npz(f"{self.artifact_uri}/train_csr.npz")

    def load_valid_csr(self) -> sp.csr_matrix:
        """Load validation interaction matrix."""
        return sp.load_npz(f"{self.artifact_uri}/valid_csr.npz")

    def load_test_csr(self) -> sp.csr_matrix:
        """Load test interaction matrix."""
        return sp.load_npz(f"{self.artifact_uri}/test_csr.npz")

    def load_train_idx(self) -> np.ndarray:
        """Load training indices."""
        return np.load(f"{self.artifact_uri}/train_idx.npy")

    def load_valid_idx(self) -> np.ndarray:
        """Load validation indices."""
        return np.load(f"{self.artifact_uri}/valid_idx.npy")

    def load_test_idx(self) -> np.ndarray:
        """Load test indices."""
        return np.load(f"{self.artifact_uri}/test_idx.npy")

    def load_train_users(self) -> np.ndarray:
        """Load training users."""
        return np.load(f"{self.artifact_uri}/train_users.npy")

    def load_valid_users(self) -> np.ndarray:
        """Load validation users."""
        return np.load(f"{self.artifact_uri}/valid_users.npy")

    def load_test_users(self) -> np.ndarray:
        """Load test users."""
        return np.load(f"{self.artifact_uri}/test_users.npy")

    def load_tag_ids(self) -> list | None:
        """Load tag IDs (only available for MovieLens)."""
        try:
            with open(f"{self.artifact_uri}/tag_ids.json") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def load_tag_item_matrix(self) -> sp.csr_matrix | None:
        """Load tag-item matrix (only available for MovieLens)."""
        try:
            return sp.load_npz(f"{self.artifact_uri}/tag_item_matrix.npz")
        except FileNotFoundError:
            return None

    def load(self) -> DatasetArtifacts:
        """
        Load all dataset artifacts.

        Returns:
            DatasetArtifacts object containing all loaded data
        """
        return DatasetArtifacts(
            users=self.load_users(),
            items=self.load_items(),
            full_csr=self.load_full_csr(),
            train_csr=self.load_train_csr(),
            valid_csr=self.load_valid_csr(),
            test_csr=self.load_test_csr(),
            train_idx=self.load_train_idx(),
            valid_idx=self.load_valid_idx(),
            test_idx=self.load_test_idx(),
            train_users=self.load_train_users(),
            valid_users=self.load_valid_users(),
            test_users=self.load_test_users(),
            metadata=self.load_metadata(),
            tag_ids=self.load_tag_ids(),
            tag_item_matrix=self.load_tag_item_matrix(),
        )

    def load_for_training(
        self,
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray]:
        """
        Load only the artifacts needed for training plugins.

        Returns:
            Tuple of (train_csr, valid_csr, test_csr, items, users)
        """
        return (
            self.load_train_csr(),
            self.load_valid_csr(),
            self.load_test_csr(),
            self.load_items(),
            self.load_users(),
        )

    def load_for_neuron_labeling(
        self,
    ) -> tuple[sp.csr_matrix, np.ndarray, list | None, sp.csr_matrix | None]:
        """
        Load only the artifacts needed for neuron labeling.

        Returns:
            Tuple of (full_csr, items, tag_ids, tag_item_matrix)
        """
        return (
            self.load_full_csr(),
            self.load_items(),
            self.load_tag_ids(),
            self.load_tag_item_matrix(),
        )
