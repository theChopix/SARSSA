"""SAE-based steering plugin comparing the current run with a past run.

For a chosen pair of SAE neurons (one from the current pipeline run
and one from a user-selected past run) the plugin computes interaction
history, base recommendations, and steered recommendations on each
side and surfaces them as six interleaved rows.  Both halves share
the same analytical core via
:func:`plugins.steering._steer.compute_steered_recommendations`.
"""

from typing import Annotated

from plugins.compare_plugin_interface import BaseComparePlugin
from plugins.plugin_interface import (
    ArtifactSpec,
    DisplayRowSpec,
    DynamicDropdownHint,
    ItemRowsDisplaySpec,
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
    SliderHint,
)
from plugins.steering._steer import compute_steered_recommendations
from utils.plugin_logger import get_logger
from utils.torch.runtime import set_device

logger = get_logger(__name__)


class Plugin(BaseComparePlugin):
    """SAE steering compare plugin.

    Loads the same input artifacts as the single variant for the
    current pipeline (``full_csr.npz``, ``users.npy``, ``items.npy``,
    base / SAE models, ``neuron_labels.json``) and pulls the past-run
    counterparts via :meth:`BaseComparePlugin.load_past_artifact`.
    Returns six rows for the frontend — current and past halves of
    interaction history, original recommendations, and steered
    recommendations — interleaved so each pair sits next to its
    counterpart in the UI.
    """

    name = "SAE Steering (compare)"

    past_run_required_steps = [
        "dataset_loading",
        "training_cfm",
        "training_sae",
        "neuron_labeling",
    ]

    io_spec = PluginIOSpec(
        required_steps=[
            "dataset_loading",
            "training_cfm",
            "training_sae",
            "neuron_labeling",
        ],
        input_artifacts=[
            ArtifactSpec("dataset_loading", "full_csr.npz", "full_csr", "npz"),
            ArtifactSpec(
                "dataset_loading",
                "users.npy",
                "users",
                "npy",
                loader_kwargs={"allow_pickle": True},
            ),
            ArtifactSpec(
                "dataset_loading",
                "items.npy",
                "items",
                "npy",
                loader_kwargs={"allow_pickle": True},
            ),
            ArtifactSpec("training_cfm", "", "base_model", "base_model"),
            ArtifactSpec("training_sae", "", "sae", "sae_model"),
            ArtifactSpec(
                "neuron_labeling",
                "neuron_labels.json",
                "neuron_labels",
                "json",
            ),
        ],
        output_artifacts=[
            OutputArtifactSpec(
                "current_interacted_items",
                "current_interacted_items.json",
                "json",
            ),
            OutputArtifactSpec(
                "past_interacted_items",
                "past_interacted_items.json",
                "json",
            ),
            OutputArtifactSpec(
                "current_original_recommendations",
                "current_original_recommendations.json",
                "json",
            ),
            OutputArtifactSpec(
                "past_original_recommendations",
                "past_original_recommendations.json",
                "json",
            ),
            OutputArtifactSpec(
                "current_steered_recommendations",
                "current_steered_recommendations.json",
                "json",
            ),
            OutputArtifactSpec(
                "past_steered_recommendations",
                "past_steered_recommendations.json",
                "json",
            ),
        ],
        output_params=[
            OutputParamSpec("user_id", "user_id_param"),
            OutputParamSpec("user_original_id", "user_original_id_param"),
            OutputParamSpec("past_user_original_id", "past_user_original_id_param"),
            OutputParamSpec("neuron_id", "neuron_id_param"),
            OutputParamSpec("label", "label_param"),
            OutputParamSpec("past_neuron_id", "past_neuron_id_param"),
            OutputParamSpec("past_label", "past_label_param"),
            OutputParamSpec("past_run_id", "past_run_id_param"),
            OutputParamSpec("alpha", "alpha_param"),
            OutputParamSpec("k", "k_param"),
        ],
        display=ItemRowsDisplaySpec(
            type="item_rows",
            rows=[
                DisplayRowSpec(
                    "current_interacted_items",
                    "Interaction History - Current Run",
                ),
                DisplayRowSpec(
                    "past_interacted_items",
                    "Interaction History - Past Run",
                ),
                DisplayRowSpec(
                    "current_original_recommendations",
                    "Original Recommendations - Current Run",
                ),
                DisplayRowSpec(
                    "past_original_recommendations",
                    "Original Recommendations - Past Run",
                ),
                DisplayRowSpec(
                    "current_steered_recommendations",
                    "Steered Recommendations - Current Run",
                ),
                DisplayRowSpec(
                    "past_steered_recommendations",
                    "Steered Recommendations - Past Run",
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
            DynamicDropdownHint(
                param_name="past_neuron_id",
                artifact_step="neuron_labeling",
                artifact_file="neuron_labels.json",
                artifact_loader="json",
                formatter="_format_neuron_choices",
                source_run_param="past_run_id",
            ),
            SliderHint(
                param_name="alpha",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
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
        past_run_id: Annotated[
            str,
            "A previously completed pipeline run to compare against; its "
            "dataset and SAE artifacts form the past-side counterpart.",
        ],
        user_id: Annotated[
            int,
            "0-based index of the user whose recommendations to steer; "
            "returns their history plus original and steered "
            "recommendations.",
        ],
        neuron_id: Annotated[
            str,
            "SAE concept neuron (from the neuron-labeling step) whose "
            "direction is amplified in the user's embedding to steer their "
            "recommendations.",
        ],
        past_neuron_id: Annotated[
            str,
            "Concept neuron from the past run to steer toward on the past "
            "side, independent of the current run's neuron_id.",
        ],
        alpha: Annotated[
            float,
            "Steering strength in [0, 1]. 0 leaves recommendations "
            "unchanged; higher pushes them more strongly toward the concept "
            "at the cost of fit to the user's history.",
        ] = 0.3,
        k: Annotated[
            int,
            "Number of recommendations to return for the original and steered lists.",
        ] = 10,
    ) -> None:
        """Steer recommendations for one user on both the current and past runs.

        Args:
            past_run_id: MLflow run id of the past pipeline run to
                compare against.  Loaded automatically by
                :class:`BaseComparePlugin` into ``self.past_context``.
            user_id: Index of the user in both datasets (0-based).
                Same index is applied to the current and past
                ``full_csr`` matrices.
            neuron_id: SAE neuron id from the current run's
                ``neuron_labels.json`` to steer toward on the current
                side.
            past_neuron_id: SAE neuron id from the past run's
                ``neuron_labels.json`` to steer toward on the past
                side.
            alpha: Steering strength in ``[0, 1]``; applied to both
                sides.
            k: Number of recommendations to return on each side.
        """
        device = set_device()
        logger.info(f"Using device: {device}")

        current = compute_steered_recommendations(
            full_csr=self.full_csr,
            items=self.items,
            users=self.users,
            base_model=self.base_model,
            sae=self.sae,
            neuron_labels=self.neuron_labels,
            user_id=user_id,
            neuron_id=neuron_id,
            alpha=alpha,
            k=k,
            device=device,
        )

        past_full_csr = self.load_past_artifact(
            "dataset_loading",
            "full_csr.npz",
            "npz",
        )
        past_users = self.load_past_artifact(
            "dataset_loading",
            "users.npy",
            "npy",
            allow_pickle=True,
        )
        past_items = self.load_past_artifact(
            "dataset_loading",
            "items.npy",
            "npy",
            allow_pickle=True,
        )
        past_base_model = self.load_past_artifact(
            "training_cfm",
            "",
            "base_model",
        )
        past_sae = self.load_past_artifact(
            "training_sae",
            "",
            "sae_model",
        )
        past_neuron_labels = self.load_past_artifact(
            "neuron_labeling",
            "neuron_labels.json",
            "json",
        )
        past = compute_steered_recommendations(
            full_csr=past_full_csr,
            items=past_items,
            users=past_users,
            base_model=past_base_model,
            sae=past_sae,
            neuron_labels=past_neuron_labels,
            user_id=user_id,
            neuron_id=past_neuron_id,
            alpha=alpha,
            k=k,
            device=device,
        )

        self.current_interacted_items = current["interacted_items"]
        self.past_interacted_items = past["interacted_items"]
        self.current_original_recommendations = current["original_recommendations"]
        self.past_original_recommendations = past["original_recommendations"]
        self.current_steered_recommendations = current["steered_recommendations"]
        self.past_steered_recommendations = past["steered_recommendations"]

        self.user_id_param = user_id
        self.user_original_id_param = current["user_original_id"]
        self.past_user_original_id_param = past["user_original_id"]
        self.neuron_id_param = neuron_id
        self.label_param = current["label"]
        self.past_neuron_id_param = past_neuron_id
        self.past_label_param = past["label"]
        self.past_run_id_param = past_run_id
        self.alpha_param = alpha
        self.k_param = k

        logger.info(
            "Compare: user %s | current neuron %s ('%s') vs past neuron %s ('%s') "
            "from run %s | alpha=%s | k=%d",
            user_id,
            current["neuron_id"],
            current["label"],
            past["neuron_id"],
            past["label"],
            past_run_id,
            alpha,
            k,
        )
