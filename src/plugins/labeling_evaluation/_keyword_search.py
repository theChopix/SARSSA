"""Top-k cosine-similarity search of label embeddings against a keyword.

Used by the ``embedding_map_with_keyword_search`` single + compare
plugins to surface the labels whose embeddings are closest to a
user-entered keyword.  Operates on the raw high-dimensional
embeddings; UMAP coordinates are layout-only and would distort
the ranking.
"""

import numpy as np


def find_top_k_nearest(
    keyword_embedding: np.ndarray,
    label_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank *label_embeddings* by cosine similarity to *keyword_embedding*.

    Args:
        keyword_embedding: ``(d,)`` embedding of the keyword.
        label_embeddings: ``(N, d)`` embeddings of the labels to
            rank.
        k: Maximum number of labels to return.  Clamped to
            ``len(label_embeddings)`` so callers can pass arbitrary
            integers without worrying about the corpus size.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(indices, similarities)``
            both of shape ``(min(k, N),)`` and aligned, sorted by
            cosine similarity in descending order.

    Raises:
        ValueError: If *k* is non-positive or *label_embeddings* is
            empty (can't return any neighbours).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if label_embeddings.shape[0] == 0:
        raise ValueError("label_embeddings must not be empty")

    keyword_norm = keyword_embedding / np.linalg.norm(keyword_embedding)
    label_norms = label_embeddings / np.linalg.norm(label_embeddings, axis=1, keepdims=True)

    similarities = label_norms @ keyword_norm

    effective_k = min(k, label_embeddings.shape[0])
    top_k_indices = np.argsort(-similarities)[:effective_k]
    return top_k_indices, similarities[top_k_indices]
