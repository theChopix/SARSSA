"""SAE-based inspection plugin comparing the current run with a past run.

For a chosen pair of SAE neurons (one from the current pipeline run and
one from a user-selected past run) the plugin returns the top-K items
each neuron activates on, side-by-side.  Both halves share the same
analytical core via :func:`plugins.inspection._top_k.compute_top_k_for_neuron`.
"""

from plugins.compare_plugin_interface import BaseComparePlugin
from plugins.inspection._top_k import compute_top_k_for_neuron
from plugins.plugin_interface import (
    ArtifactSpec,
    DisplayRowSpec,
    DynamicDropdownHint,
    ItemRowsDisplaySpec,
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
)
from utils.plugin_logger import get_logger

logger = get_logger(__name__)


class Plugin(BaseComparePlugin):
    """SAE inspection compare plugin.

    Loads the same input artifacts as the single variant for the
    current pipeline (``items.npy``, ``neuron_labels.json``,
    ``item_acts.pt``) and pulls the past-run counterparts via
    :meth:`BaseComparePlugin.load_past_artifact`.  Returns two rows
    of top-K items — one per side — for the frontend to render
    side-by-side.
    """

    name = "SAE Inspection (compare)"

    past_run_required_steps = ["dataset_loading", "neuron_labeling"]

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
            OutputArtifactSpec(
                "current_top_k_item_ids",
                "current_top_k_item_ids.json",
                "json",
            ),
            OutputArtifactSpec(
                "current_top_k_activations",
                "current_top_k_activations.json",
                "json",
            ),
            OutputArtifactSpec(
                "past_top_k_item_ids",
                "past_top_k_item_ids.json",
                "json",
            ),
            OutputArtifactSpec(
                "past_top_k_activations",
                "past_top_k_activations.json",
                "json",
            ),
        ],
        output_params=[
            OutputParamSpec("neuron_id", "neuron_id_param"),
            OutputParamSpec("label", "label_param"),
            OutputParamSpec("past_neuron_id", "past_neuron_id_param"),
            OutputParamSpec("past_label", "past_label_param"),
            OutputParamSpec("past_run_id", "past_run_id_param"),
            OutputParamSpec("k", "k_param"),
        ],
        display=ItemRowsDisplaySpec(
            type="item_rows",
            rows=[
                DisplayRowSpec("current_top_k_item_ids", "Current Run"),
                DisplayRowSpec("past_top_k_item_ids", "Past Run"),
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
            DynamicDropdownHint(
                param_name="past_neuron_id",
                artifact_step="neuron_labeling",
                artifact_file="neuron_labels.json",
                artifact_loader="json",
                formatter="_format_neuron_choices",
                source_run_param="past_run_id",
            ),
        ],
    )

    @staticmethod
    def _format_neuron_choices(
        data: dict[str, str],
    ) -> list[dict[str, str]]:
        """Format ``neuron_labels.json`` into dropdown options.

        Args:
            data: Mapping of neuron id (string) to label string.

        Returns:
            list[dict[str, str]]: Options with ``"label"`` and
                ``"value"`` keys.
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
        past_run_id: str,
        neuron_id: str,
        past_neuron_id: str,
        k: int = 10,
    ) -> None:
        """Compute top-k items for the current and past neurons.

        Args:
            past_run_id: MLflow run id of the past pipeline run to
                compare against.  Loaded automatically by
                :class:`BaseComparePlugin` into ``self.past_context``.
            neuron_id: Concept neuron id from the current run's
                ``neuron_labels.json``.
            past_neuron_id: Concept neuron id from the past run's
                ``neuron_labels.json``.
            k: Number of top items to return for each side.
        """
        current = compute_top_k_for_neuron(
            neuron_id=neuron_id,
            neuron_labels=self.neuron_labels,
            items=self.items,
            item_acts=self.item_acts,
            k=k,
        )

        past_neuron_labels = self.load_past_artifact(
            "neuron_labeling",
            "neuron_labels.json",
            "json",
        )
        past_items = self.load_past_artifact(
            "dataset_loading",
            "items.npy",
            "npy",
            allow_pickle=True,
        )
        past_item_acts = self.load_past_artifact(
            "neuron_labeling",
            "item_acts.pt",
            "pt",
        )
        past = compute_top_k_for_neuron(
            neuron_id=past_neuron_id,
            neuron_labels=past_neuron_labels,
            items=past_items,
            item_acts=past_item_acts,
            k=k,
        )

        self.current_top_k_item_ids = current["top_k_item_ids"]
        self.current_top_k_activations = current["top_k_activations"]
        self.past_top_k_item_ids = past["top_k_item_ids"]
        self.past_top_k_activations = past["top_k_activations"]

        self.neuron_id_param = neuron_id
        self.label_param = current["label"]
        self.past_neuron_id_param = past_neuron_id
        self.past_label_param = past["label"]
        self.past_run_id_param = past_run_id
        self.k_param = current["k"]

        logger.info(
            "Compare: current neuron %s ('%s') vs past neuron %s ('%s') from run %s; "
            "top-%d items per side",
            current["neuron_id"],
            current["label"],
            past["neuron_id"],
            past["label"],
            past_run_id,
            current["k"],
        )
