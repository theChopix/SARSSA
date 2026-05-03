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
    DynamicDropdownHint,
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
                "neuron_labels.json",
                "neuron_labels",
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
            OutputParamSpec("neuron_id", "neuron_id_param"),
            OutputParamSpec("label", "label_param"),
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
        param_ui_hints=[
            DynamicDropdownHint(
                param_name="neuron_id",
                artifact_step="neuron_labeling",
                artifact_file="neuron_labels.json",
                artifact_loader="json",
                formatter="_format_neuron_choices",
            ),
        ],
    )

    @staticmethod
    def _format_neuron_choices(
        data: dict[str, str],
    ) -> list[dict[str, str]]:
        """Format neuron_labels.json into dropdown options.

        Args:
            data: Mapping of neuron ID to label string.

        Returns:
            list[dict[str, str]]: Options with ``"label"``
                and ``"value"`` keys.
        """
        return [
            {
                "label": f"{label} [neuron id {nid}]",
                "value": nid,
            }
            for nid, label in data.items()
        ]

    def run(
        self,
        neuron_id: str,
        k: int = 10,
    ) -> None:
        """Inspect which items activate a concept neuron the most.

        Args:
            neuron_id: SAE neuron ID to inspect (as string
                from dropdown selection).
            k: Number of top items to return.
        """
        if neuron_id not in self.neuron_labels:
            raise ValueError(f"Neuron ID '{neuron_id}' not found in neuron_labels mapping")

        self.neuron_id = int(neuron_id)
        self.label = self.neuron_labels[neuron_id]
        logger.info(f"Neuron {self.neuron_id} ('{self.label}')")

        # Get activations for the target neuron across all items
        neuron_activations = self.item_acts[:, self.neuron_id]  # (num_items,)

        # Top-k items by activation (descending)
        k = min(k, len(neuron_activations))
        topk_values, topk_indices = torch.topk(neuron_activations, k)

        self.top_k_item_ids = self.items[topk_indices.numpy()].tolist()
        self.top_k_activations = topk_values.numpy().tolist()

        # output params
        self.neuron_id_param = neuron_id
        self.label_param = self.label
        self.k_param = k

        logger.info(f"Neuron {self.neuron_id} ('{self.label}') | top-{k} items retrieved")
