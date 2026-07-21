"""SAE-based inspection plugin for recommender systems.

Given a concept tag, finds the SAE neuron associated with it and returns
the top-K items for which that neuron activates the most (sorted by
activation strength, descending).
"""

from typing import Annotated

from plugins.inspection._top_k import compute_top_k_for_neuron
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
    description = (
        "Pick a labeled autoencoder neuron — a concept — and see the top items that "
        "activate it most strongly, i.e. the items that most embody that concept. "
        "Reuses precomputed activations for a fast top-K lookup without re-running any "
        "model."
    )

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
            ArtifactSpec("neuron_labeling", "item_acts.npz", "item_acts", "npz"),
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
            OutputParamSpec("label", "label_param"),
            OutputParamSpec("k_used", "k_param"),
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
        data: dict[str, dict],
    ) -> list[dict[str, object]]:
        """Format neuron_labels.json into dropdown options.

        Args:
            data: Mapping of neuron id (string) to a ``{"label"}`` entry,
                optionally carrying a numeric ``"confidence"``.

        Returns:
            list[dict[str, object]]: Options with ``"label"``, ``"value"``, an
                ``"emphasis"`` (the label text to bold) and, when the entry has
                a confidence, a numeric ``"tint"`` in [-1, 1] the frontend maps
                to a background colour; the label shows the score too.
        """
        choices: list[dict[str, object]] = []
        for nid, entry in data.items():
            name = f"{entry.get('label')}"
            confidence = entry.get("confidence")
            if confidence is None:
                text = f"{name} [neuron id {nid}]"
            else:
                text = f"{name} [neuron id {nid}] · conf {confidence:.3f}"
            choices.append({"label": text, "value": nid, "tint": confidence, "emphasis": name})
        return choices

    def run(
        self,
        neuron_id: Annotated[
            str,
            "SAE concept neuron (from the neuron-labeling step) to inspect; "
            "returns the items it activates on most strongly. Options read '<label> [neuron id <id>]'; a "
            "'conf <c>' suffix and colour tint, when present, show "
            "the label confidence (-1 to 1).",
        ],
        k: Annotated[
            int,
            "Number of top-activating items to return, ranked by activation "
            "strength (highest first).",
        ] = 10,
    ) -> None:
        """Inspect which items activate a concept neuron the most.

        Args:
            neuron_id: SAE neuron ID to inspect (as string
                from dropdown selection).
            k: Number of top items to return.
        """
        result = compute_top_k_for_neuron(
            neuron_id=neuron_id,
            neuron_labels=self.neuron_labels,
            items=self.items,
            item_acts=self.item_acts,
            k=k,
        )

        self.neuron_id = result["neuron_id"]
        self.label = result["label"]
        self.top_k_item_ids = result["top_k_item_ids"]
        self.top_k_activations = result["top_k_activations"]
        self.label_param = self.label
        self.k_param = result["k"]

        logger.info(f"Neuron {self.neuron_id} ('{self.label}') | top-{result['k']} items retrieved")
