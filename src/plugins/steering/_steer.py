"""Shared steering computation for SAE steering plugins.

Both ``steering.single.sae_steering`` and
``steering.compare.sae_steering`` invoke this helper so the analytical
core lives in one place and the compare plugin can apply it to two
independent contexts (current pipeline and past run).
"""

from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix

from utils.torch.models.steered_model import SteeredModel


def compute_steered_recommendations(
    full_csr: csr_matrix,
    items: np.ndarray,
    users: np.ndarray,
    base_model: Any,
    sae: Any,
    neuron_labels: dict[str, str],
    user_id: int,
    neuron_id: str,
    alpha: float,
    k: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Compute interaction history, base recs, and steered recs.

    Sends the user's interaction vector through the base model to get
    the original top-K recommendations, then through a steered version
    of the same model (boosted along the requested SAE neuron) to get
    the steered top-K recommendations.  Items the user already
    interacted with are masked out of both recommendation sets.

    Args:
        full_csr: User x item interaction matrix.
        items: Array of item ids indexable by item column index.
        users: Array of user ids indexable by user row index.
        base_model: Trained recommender model exposing ``recommend``.
        sae: Trained sparse autoencoder paired with *base_model*.
        neuron_labels: Mapping from neuron id (string) to label.
        user_id: Index of the user in *full_csr* (0-based).
        neuron_id: SAE neuron id as a string; must match a key in
            *neuron_labels*.
        alpha: Steering strength in ``[0, 1]``.
        k: Number of recommendations to return per side.
        device: Torch device the models should run on.

    Returns:
        dict[str, Any]: A payload with keys ``"user_id"`` (int),
            ``"user_original_id"`` (string), ``"neuron_id"`` (int),
            ``"label"`` (human-readable label), ``"interacted_items"``
            (list of item ids the user has already interacted with),
            ``"original_recommendations"`` (list of item ids from the
            base model), and ``"steered_recommendations"`` (list of
            item ids from the steered model).

    Raises:
        ValueError: If *user_id* is out of range, *neuron_id* is not
            a key in *neuron_labels*, or *alpha* is outside ``[0, 1]``.
    """
    if user_id < 0 or user_id >= full_csr.shape[0]:
        raise ValueError(f"user_id {user_id} out of range [0, {full_csr.shape[0] - 1}]")
    if neuron_id not in neuron_labels:
        raise ValueError(f"Neuron ID '{neuron_id}' not found in neuron_labels mapping")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    int_neuron_id = int(neuron_id)
    label = neuron_labels[neuron_id]

    base_model.to(device)
    sae.to(device)

    interaction_vec = torch.tensor(full_csr[user_id].toarray(), dtype=torch.float32, device=device)

    interacted_indices = full_csr[user_id].indices.tolist()
    interacted_items = items[interacted_indices].tolist()

    base_model.eval()
    _, orig_indices = base_model.recommend(interaction_vec, k=k, mask_interactions=True)
    original_recommendations = items[orig_indices[0]].tolist()

    steered_model = SteeredModel(base_model, sae, alpha=alpha)
    steered_model.eval()
    _, steered_indices = steered_model.recommend(
        interaction_vec, neuron_ids=[int_neuron_id], k=k, mask_interactions=True
    )
    steered_recommendations = items[steered_indices[0]].tolist()

    return {
        "user_id": user_id,
        "user_original_id": str(users[user_id]),
        "neuron_id": int_neuron_id,
        "label": label,
        "interacted_items": interacted_items,
        "original_recommendations": original_recommendations,
        "steered_recommendations": steered_recommendations,
    }
