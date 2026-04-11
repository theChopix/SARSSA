"""SAE-based inspection plugin for recommender systems.

Given a concept tag, finds the SAE neuron associated with it and returns
the top-K items for which that neuron activates the most (sorted by
activation strength, descending).
"""

import json
import tempfile

import mlflow
import numpy as np
import torch

from plugins.plugin_interface import BasePlugin
from utils.mlflow_manager import MLflowRunLoader
from utils.plugin_logger import get_logger

logger = get_logger(__name__)


class Plugin(BasePlugin):
    """SAE inspection plugin for exploring neuron–item relationships.

    Expects prior ``dataset_loading`` and ``neuron_labeling`` steps in
    the pipeline context.
    """

    def _load_artifacts(self, context: dict):
        """Load all required artifacts from previous pipeline steps."""
        dataset_run_id = context["dataset_loading"]["run_id"]
        dataset_loader = MLflowRunLoader(dataset_run_id)
        logger.info(f"Loading dataset artifacts from run {dataset_run_id}")

        self.items: np.ndarray = dataset_loader.get_npy_artifact("items.npy", allow_pickle=True)
        logger.info(f"Loaded {len(self.items)} items")

        labeling_run_id = context["neuron_labeling"]["run_id"]
        labeling_loader = MLflowRunLoader(labeling_run_id)

        self.top_neuron_per_tag: dict = labeling_loader.get_json_artifact("top_neuron_per_tag.json")
        logger.info(f"Loaded {len(self.top_neuron_per_tag)} top-neuron-per-tag mappings")

        item_acts_path = labeling_loader.download_artifact("item_acts.pt")
        self.item_acts: torch.Tensor = torch.load(
            item_acts_path, map_location="cpu", weights_only=True
        )
        logger.info(f"Loaded item activations: {self.item_acts.shape}")

    def run(
        self,
        context: dict,
        tag: str,
        k: int = 10,
    ):
        """Inspect which items activate a concept neuron the most.

        Args:
            context: Pipeline context with run IDs from previous steps.
                Required keys: ``dataset_loading``, ``neuron_labeling``.
            tag: Concept tag to inspect (must exist in top_neuron_per_tag).
            k: Number of top items to return.

        Returns:
            dict with keys ``neuron_id``, ``top_k_item_ids``, and
            ``top_k_activations`` (activation values for the returned items).
        """
        self._load_artifacts(context)

        if tag not in self.top_neuron_per_tag:
            raise ValueError(f"Tag '{tag}' not found in top_neuron_per_tag mapping")

        neuron_id = self.top_neuron_per_tag[tag]
        logger.info(f"Tag '{tag}' → neuron {neuron_id}")

        # Get activations for the target neuron across all items
        neuron_activations = self.item_acts[:, neuron_id]  # (num_items,)

        # Top-k items by activation (descending)
        k = min(k, len(neuron_activations))
        topk_values, topk_indices = torch.topk(neuron_activations, k)

        top_k_item_ids = self.items[topk_indices.numpy()].tolist()
        top_k_activations = topk_values.numpy().tolist()

        logger.info(f"Tag '{tag}' | neuron={neuron_id} | top-{k} items retrieved")

        mlflow.log_params(
            {
                "tag": tag,
                "neuron_id": neuron_id,
                "k": k,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/top_k_item_ids.json", "w") as f:
                json.dump(top_k_item_ids, f, indent=2, default=str)
            with open(f"{tmp}/top_k_activations.json", "w") as f:
                json.dump(top_k_activations, f, indent=2, default=str)
            mlflow.log_artifacts(tmp)

        logger.info("Inspection complete — results logged to MLflow")
