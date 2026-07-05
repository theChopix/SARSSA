"""Confidence scoring shared by the TAG-based neuron-labeling plugins.

A label's confidence is the point-biserial correlation ``r`` in ``[-1, 1]``
between a neuron's activation and the binary presence of its assigned tag.
"""

import numpy as np
import scipy.sparse as sp


def point_biserial_matrix(activations: np.ndarray, attributes: sp.spmatrix) -> np.ndarray:
    """Point-biserial correlation for every (tag, neuron) pair.

    Pearson correlation between each neuron's continuous activation and each
    tag's binary presence indicator, evaluated for all pairs at once via the
    sum form ``r = (n·Σay − Σa·Σy) / √((n·Σa² − Σa²)(n·Σy² − Σy²))``. Since
    the attribute is binary, ``Σy² = Σy``. Pairs with zero variance (a dead
    neuron or an all/no-item tag) are set to 0.

    Args:
        activations: Dense ``(num_items, num_neurons)`` activation matrix.
        attributes: Binary ``(num_items, num_tags)`` attribute matrix.

    Returns:
        np.ndarray: Correlation matrix of shape ``(num_tags, num_neurons)``.
    """
    a = activations.astype(np.float64)
    n = a.shape[0]

    cross = attributes.T @ a  # (tags x neurons) = Σ(a·y)
    if sp.issparse(cross):
        cross = cross.toarray()

    sum_a = a.sum(axis=0)  # Σa  (neurons,)
    sum_a2 = (a * a).sum(axis=0)  # Σa² (neurons,)
    n1 = np.asarray(attributes.sum(axis=0)).ravel()  # Σy = items per tag (tags,)

    num = n * cross - n1[:, None] * sum_a[None, :]
    var_a = n * sum_a2 - sum_a**2  # (neurons,)
    var_y = n * n1 - n1**2  # (tags,), using Σy² = n1
    denom = np.sqrt(var_y[:, None] * var_a[None, :])

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = num / denom
    corr[~np.isfinite(corr)] = 0.0
    return corr


def labels_with_confidence(
    neuron_labels: dict[int, str | None],
    label_tag_index: dict[int, int | None],
    corr: np.ndarray,
) -> tuple[dict[int, dict[str, object]], float]:
    """Pair each neuron label with the confidence of its assignment.

    Args:
        neuron_labels: Neuron id → assigned tag label (``None`` if unlabelled).
        label_tag_index: Neuron id → row index of the assigned tag in ``corr``
            (``None`` if unlabelled). Must share keys with ``neuron_labels``.
        corr: Point-biserial matrix of shape ``(num_tags, num_neurons)`` from
            :func:`point_biserial_matrix`.

    Returns:
        tuple:
            - Neuron id → ``{"label": str | None, "confidence": float | None}``.
              Unlabelled neurons carry ``None`` for both fields.
            - Mean confidence over labelled neurons (``0.0`` when none).
    """
    result: dict[int, dict[str, object]] = {}
    scores: list[float] = []
    for neuron, label in neuron_labels.items():
        tag_index = label_tag_index.get(neuron)
        if label is None or tag_index is None:
            result[neuron] = {"label": None, "confidence": None}
            continue
        score = float(corr[tag_index, neuron])
        result[neuron] = {"label": label, "confidence": score}
        scores.append(score)

    mean_confidence = float(np.mean(scores)) if scores else 0.0
    return result, mean_confidence
