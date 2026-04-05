"""SAE-based steering plugin for recommender systems.

Given a user (by index), a concept tag, and a steering strength alpha,
returns the user's interaction history, the base-model's top-K recommendations,
and the SAE neuron-boosted top-K recommendations.
"""

import json
import tempfile

import mlflow
import numpy as np
import scipy.sparse as sp
import torch

from plugins.plugin_interface import BasePlugin
from utils.mlflow_manager import MLflowRunLoader
from utils.plugin_logger import get_logger
from utils.torch.models.model_loader import load_base_model, load_sae_model
from utils.torch.models.steered_model import SteeredModel
from utils.torch.runtime import set_device

logger = get_logger(__name__)


class Plugin(BasePlugin):
    """SAE steering plugin for interactive recommendation inspection.

    Expects prior ``dataset_loading``, ``training_cfm``, ``training_sae``,
    and ``neuron_labeling`` steps in the pipeline context.
    """

    def _load_artifacts(self, context: dict, device):
        """Load all required artifacts from previous pipeline steps."""
        dataset_run_id = context["dataset_loading"]["run_id"]
        dataset_loader = MLflowRunLoader(dataset_run_id)
        logger.info(f"Loading dataset artifacts from run {dataset_run_id}")

        full_npz = dataset_loader.get_npz_artifact("full_csr.npz")
        self.full_csr: sp.csr_matrix = (
            sp.csr_matrix(full_npz) if not isinstance(full_npz, sp.csr_matrix) else full_npz
        )
        self.users: np.ndarray = dataset_loader.get_npy_artifact("users.npy", allow_pickle=True)
        self.items: np.ndarray = dataset_loader.get_npy_artifact("items.npy", allow_pickle=True)
        logger.info(f"Dataset: {self.full_csr.shape[0]} users, {self.full_csr.shape[1]} items")

        base_run_id = context["training_cfm"]["run_id"]
        base_loader = MLflowRunLoader(base_run_id)
        logger.info("Loading base model")
        self.base_model = load_base_model(base_loader.get_artifact_path(), device)
        logger.info("Base model loaded successfully")

        sae_run_id = context["training_sae"]["run_id"]
        sae_loader = MLflowRunLoader(sae_run_id)
        logger.info("Loading SAE model")
        self.sae = load_sae_model(sae_loader.get_artifact_path(), device)
        logger.info("SAE model loaded successfully")

        labeling_run_id = context["neuron_labeling"]["run_id"]
        labeling_loader = MLflowRunLoader(labeling_run_id)
        self.top_neuron_per_tag: dict = labeling_loader.get_json_artifact("top_neuron_per_tag.json")
        logger.info(f"Loaded {len(self.top_neuron_per_tag)} top-neuron-per-tag mappings")

    def run(
        self,
        context: dict,
        user_id: int,
        tag: str,
        alpha: float = 0.3,
        k: int = 10,
    ):
        """Steer recommendations for a single user toward a concept.

        Args:
            context: Pipeline context with run IDs from previous steps.
                Required keys: ``dataset_loading``, ``training_cfm``,
                ``training_sae``, ``neuron_labeling``.
            user_id: Index of the user in the dataset (0-based).
            tag: Concept tag to steer toward (must exist in top_neuron_per_tag).
            alpha: Steering strength in [0, 1].
            k: Number of recommendations to return.

        Returns:
            dict with keys ``interacted_items``, ``original_recommendations``,
            and ``steered_recommendations`` (all lists of item IDs).
        """
        device = set_device()
        logger.info(f"Using device: {device}")

        self._load_artifacts(context, device)

        if user_id < 0 or user_id >= self.full_csr.shape[0]:
            raise ValueError(f"user_id {user_id} out of range [0, {self.full_csr.shape[0] - 1}]")
        if tag not in self.top_neuron_per_tag:
            raise ValueError(f"Tag '{tag}' not found in top_neuron_per_tag mapping")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        neuron_id = self.top_neuron_per_tag[tag]
        logger.info(f"Tag '{tag}' → neuron {neuron_id}")

        # User interaction vector — shape (1, num_items)
        interaction_vec = torch.tensor(
            self.full_csr[user_id].toarray(), dtype=torch.float32, device=device
        )

        # Interacted items
        interacted_indices = self.full_csr[user_id].indices.tolist()
        interacted_items = self.items[interacted_indices].tolist()

        # Original recommendations (base model only, no SAE)
        self.base_model.eval()
        _, orig_indices = self.base_model.recommend(interaction_vec, k=k, mask_interactions=True)
        original_recommendations = self.items[orig_indices[0]].tolist()

        # Steered recommendations (base model + SAE + neuron boost)
        steered_model = SteeredModel(self.base_model, self.sae, alpha=alpha)
        steered_model.eval()
        _, steered_indices = steered_model.recommend(
            interaction_vec, neuron_ids=[neuron_id], k=k, mask_interactions=True
        )
        steered_recommendations = self.items[steered_indices[0]].tolist()

        logger.info(
            f"User {user_id} | tag='{tag}' | neuron={neuron_id} | alpha={alpha} | "
            f"interacted={len(interacted_items)} items"
        )

        mlflow.log_params(
            {
                "user_id": user_id,
                "user_original_id": str(self.users[user_id]),
                "tag": tag,
                "neuron_id": neuron_id,
                "alpha": alpha,
                "k": k,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/interacted_items.json", "w") as f:
                json.dump(interacted_items, f, indent=2, default=str)
            with open(f"{tmp}/original_recommendations.json", "w") as f:
                json.dump(original_recommendations, f, indent=2, default=str)
            with open(f"{tmp}/steered_recommendations.json", "w") as f:
                json.dump(steered_recommendations, f, indent=2, default=str)
            mlflow.log_artifacts(tmp)

        logger.info("Steering complete — results logged to MLflow")
