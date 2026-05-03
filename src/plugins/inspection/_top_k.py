"""Shared top-k computation for SAE inspection plugins.

Both ``inspection.single.sae_inspection`` and
``inspection.compare.sae_inspection`` invoke this helper so the
analytical core lives in one place and the compare plugin can apply
it to two independent contexts (current pipeline and past run).
"""

from typing import Any

import numpy as np
import torch


def compute_top_k_for_neuron(
    neuron_id: str,
    neuron_labels: dict[str, str],
    items: np.ndarray,
    item_acts: torch.Tensor,
    k: int,
) -> dict[str, Any]:
    """Compute the items most strongly activating a concept neuron.

    Args:
        neuron_id: SAE neuron id as a string; must match a key in
            *neuron_labels*.
        neuron_labels: Mapping from neuron id (string) to label.
        items: Array of item ids indexable by row index of
            *item_acts*.
        item_acts: Activation tensor with shape
            ``(num_items, num_neurons)``.
        k: Requested top-k size; clamped to ``len(item_acts)`` when
            larger.

    Returns:
        dict[str, Any]: A payload with keys ``"neuron_id"`` (int
            index of the neuron), ``"label"`` (human-readable label),
            ``"top_k_item_ids"`` (list of item ids sorted by
            activation descending), ``"top_k_activations"`` (list of
            activation values aligned to ``top_k_item_ids``), and
            ``"k"`` (actual k used after clamping).

    Raises:
        ValueError: If *neuron_id* is not a key in *neuron_labels*
            or if *k* is not positive.
    """
    if neuron_id not in neuron_labels:
        raise ValueError(f"Neuron ID '{neuron_id}' not found in neuron_labels mapping")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    int_id = int(neuron_id)
    label = neuron_labels[neuron_id]

    neuron_activations = item_acts[:, int_id]
    actual_k = min(k, len(neuron_activations))
    topk_values, topk_indices = torch.topk(neuron_activations, actual_k)

    return {
        "neuron_id": int_id,
        "label": label,
        "top_k_item_ids": items[topk_indices.numpy()].tolist(),
        "top_k_activations": topk_values.numpy().tolist(),
        "k": actual_k,
    }
