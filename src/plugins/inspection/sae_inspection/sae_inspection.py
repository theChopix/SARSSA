"""SAE-based inspection plugin for recommender systems.

Given a concept tag, finds the SAE neuron associated with it and returns
the top-K items for which that neuron activates the most (sorted by
activation strength, descending).
"""

import torch

from plugins.plugin_interface import (
    ArtifactSpec,
    BasePlugin,
    DisplayRowSpec,
    ItemRowsDisplaySpec,
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
)
from utils.plugin_logger import get_logger

logger = get_logger(__name__)


class Plugin(BasePlugin):
    """SAE inspection plugin for exploring neuron–item relationships.

    Expects prior ``dataset_loading`` and ``neuron_labeling`` steps in
    the pipeline context.
    """

    name = "SAE Inspection"

    io_spec = PluginIOSpec(
        required_steps=["dataset_loading", "neuron_labeling"],
        input_artifacts=[
            ArtifactSpec(
                "dataset_loading",
                "items.npy",
                "items",
                "npy",
                loader_kwargs={"allow_pickle": True},
            ),
            ArtifactSpec(
                "neuron_labeling",
                "top_neuron_per_tag.json",
                "top_neuron_per_tag",
                "json",
            ),
            ArtifactSpec("neuron_labeling", "item_acts.pt", "item_acts", "pt"),
        ],
        output_artifacts=[
            OutputArtifactSpec("top_k_item_ids", "top_k_item_ids.json", "json"),
            OutputArtifactSpec(
                "top_k_activations",
                "top_k_activations.json",
                "json",
            ),
        ],
        output_params=[
            OutputParamSpec("tag", "tag_param"),
            OutputParamSpec("neuron_id", "neuron_id"),
            OutputParamSpec("k", "k_param"),
        ],
        display=ItemRowsDisplaySpec(
            type="item_rows",
            rows=[
                DisplayRowSpec(
                    "top_k_item_ids",
                    "Top Items for Concept",
                ),
            ],
        ),
    )

    def run(
        self,
        tag: str,
        k: int = 10,
    ) -> None:
        """Inspect which items activate a concept neuron the most.

        Args:
            tag: Concept tag to inspect (must exist in top_neuron_per_tag).
            k: Number of top items to return.
        """
        if tag not in self.top_neuron_per_tag:
            raise ValueError(f"Tag '{tag}' not found in top_neuron_per_tag mapping")

        self.neuron_id = self.top_neuron_per_tag[tag]
        logger.info(f"Tag '{tag}' → neuron {self.neuron_id}")

        # Get activations for the target neuron across all items
        neuron_activations = self.item_acts[:, self.neuron_id]  # (num_items,)

        # Top-k items by activation (descending)
        k = min(k, len(neuron_activations))
        topk_values, topk_indices = torch.topk(neuron_activations, k)

        self.top_k_item_ids = self.items[topk_indices.numpy()].tolist()
        self.top_k_activations = topk_values.numpy().tolist()

        # output params
        self.tag_param = tag
        self.k_param = k

        logger.info(f"Tag '{tag}' | neuron={self.neuron_id} | top-{k} items retrieved")
