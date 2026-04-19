"""SAE-based steering plugin for recommender systems.

Given a user (by index), a concept tag, and a steering strength alpha,
returns the user's interaction history, the base-model's top-K recommendations,
and the SAE neuron-boosted top-K recommendations.
"""

import torch

from plugins.plugin_interface import (
    ArtifactSpec,
    BasePlugin,
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
)
from utils.plugin_logger import get_logger
from utils.torch.models.steered_model import SteeredModel
from utils.torch.runtime import set_device

logger = get_logger(__name__)


class Plugin(BasePlugin):
    """SAE steering plugin for interactive recommendation inspection.

    Expects prior ``dataset_loading``, ``training_cfm``, ``training_sae``,
    and ``neuron_labeling`` steps in the pipeline context.
    """

    name = "SAE Steering"

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
                "top_neuron_per_tag.json",
                "top_neuron_per_tag",
                "json",
            ),
        ],
        output_artifacts=[
            OutputArtifactSpec(
                "interacted_items",
                "interacted_items.json",
                "json",
            ),
            OutputArtifactSpec(
                "original_recommendations",
                "original_recommendations.json",
                "json",
            ),
            OutputArtifactSpec(
                "steered_recommendations",
                "steered_recommendations.json",
                "json",
            ),
        ],
        output_params=[
            OutputParamSpec("user_id", "user_id_param"),
            OutputParamSpec("user_original_id", "user_original_id"),
            OutputParamSpec("tag", "tag_param"),
            OutputParamSpec("neuron_id", "neuron_id"),
            OutputParamSpec("alpha", "alpha_param"),
            OutputParamSpec("k", "k_param"),
        ],
    )

    def run(
        self,
        user_id: int,
        tag: str,
        alpha: float = 0.3,
        k: int = 10,
    ) -> None:
        """Steer recommendations for a single user toward a concept.

        Args:
            user_id: Index of the user in the dataset (0-based).
            tag: Concept tag to steer toward (must exist in top_neuron_per_tag).
            alpha: Steering strength in [0, 1].
            k: Number of recommendations to return.
        """
        device = set_device()
        logger.info(f"Using device: {device}")

        self.base_model.to(device)
        self.sae.to(device)

        if user_id < 0 or user_id >= self.full_csr.shape[0]:
            raise ValueError(f"user_id {user_id} out of range [0, {self.full_csr.shape[0] - 1}]")
        if tag not in self.top_neuron_per_tag:
            raise ValueError(f"Tag '{tag}' not found in top_neuron_per_tag mapping")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.neuron_id = self.top_neuron_per_tag[tag]
        logger.info(f"Tag '{tag}' → neuron {self.neuron_id}")

        # User interaction vector — shape (1, num_items)
        interaction_vec = torch.tensor(
            self.full_csr[user_id].toarray(), dtype=torch.float32, device=device
        )

        # Interacted items
        interacted_indices = self.full_csr[user_id].indices.tolist()
        self.interacted_items = self.items[interacted_indices].tolist()

        # Original recommendations (base model only, no SAE)
        self.base_model.eval()
        _, orig_indices = self.base_model.recommend(interaction_vec, k=k, mask_interactions=True)
        self.original_recommendations = self.items[orig_indices[0]].tolist()

        # Steered recommendations (base model + SAE + neuron boost)
        steered_model = SteeredModel(self.base_model, self.sae, alpha=alpha)
        steered_model.eval()
        _, steered_indices = steered_model.recommend(
            interaction_vec, neuron_ids=[self.neuron_id], k=k, mask_interactions=True
        )
        self.steered_recommendations = self.items[steered_indices[0]].tolist()

        # output params
        self.user_id_param = user_id
        self.user_original_id = str(self.users[user_id])
        self.tag_param = tag
        self.alpha_param = alpha
        self.k_param = k

        logger.info(
            f"User {user_id} | tag='{tag}' | neuron={self.neuron_id} | alpha={alpha} | "
            f"interacted={len(self.interacted_items)} items"
        )
