"""Shared nearest-neighbour distance computation for compare plugins.

Both ``labeling_evaluation.compare.nearest_label_distance_bars`` and
``labeling_evaluation.compare.nearest_label_distance_histogram``
invoke this helper so the analytical core — embedding both label
sets and computing the per-current-label cosine distance to the
nearest past-run label — lives in one place.

The embedding step delegates to
:func:`plugins.labeling_evaluation._embedding_cache.embed_labels`,
which memoizes per ``(labels, provider, model)`` so the two
plugins running in the same worker process share work.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from plugins.labeling_evaluation._embedding_cache import embed_labels


@dataclass(frozen=True)
class NearestLabelDistances:
    """Result of :func:`compute_nearest_distances`.

    Attributes:
        distances: ``(len(current_label_texts),)`` cosine distance
            from each current label's embedding to its nearest
            past-label embedding.  Indexed in input order.
        nearest_past_indices: ``(len(current_label_texts),)`` index
            into ``past_label_texts`` of the nearest past label
            for each current label.
    """

    distances: np.ndarray
    nearest_past_indices: np.ndarray


def compute_nearest_distances(
    current_label_texts: list[str],
    past_label_texts: list[str],
    embedding_provider: str,
    embedding_model: str,
) -> NearestLabelDistances:
    """Embed both label sets and find each current label's nearest past neighbour.

    Args:
        current_label_texts: Labels from the current pipeline run.
        past_label_texts: Labels from the past pipeline run being
            compared against.
        embedding_provider: Embedder provider name resolved by the
            registry (e.g. ``"openai"``).
        embedding_model: Provider-specific model identifier (e.g.
            ``"text-embedding-3-small"`` for OpenAI).

    Returns:
        NearestLabelDistances: Per-current-label cosine distance to
            the nearest past-label embedding, plus the matching
            past-label index.

    Raises:
        ValueError: If either label list is empty (propagated from
            :func:`embed_labels`).
    """
    current_embeddings = embed_labels(current_label_texts, embedding_provider, embedding_model)
    past_embeddings = embed_labels(past_label_texts, embedding_provider, embedding_model)

    # cdist returns shape (n_current, n_past); cosine distance is
    # 1 - cos_sim, so taking argmin/min over axis=1 picks the nearest
    # past label for each current label.
    distance_matrix = cdist(current_embeddings, past_embeddings, metric="cosine")
    nearest_past_indices = distance_matrix.argmin(axis=1)
    distances = distance_matrix[np.arange(len(current_label_texts)), nearest_past_indices]

    return NearestLabelDistances(
        distances=distances,
        nearest_past_indices=nearest_past_indices,
    )
