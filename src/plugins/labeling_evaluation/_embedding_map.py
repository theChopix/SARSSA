"""Shared embedding + UMAP computation for embedding_map plugins.

Both ``labeling_evaluation.single.embedding_map`` and
``labeling_evaluation.compare.embedding_map`` invoke this helper so
the analytical core (text embedding via OpenAI, then 2-D UMAP
projection) lives in one place.  The compare variant feeds the
*concatenation* of two label sets through the helper so both sides
share the same UMAP coordinate space.
"""

import numpy as np
import umap

from utils.embedder.openai_embedder import OpenAIEmbeddingLLM


def compute_label_embedding_coords(
    label_texts: list[str],
    embedding_model: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: int,
) -> np.ndarray:
    """Embed *label_texts* and project them to a 2-D UMAP space.

    Args:
        label_texts: Strings to embed.  Order is preserved through
            both the embedding and the UMAP fit so the returned
            coordinates align row-wise with the input.
        embedding_model: OpenAI embedding model identifier
            (e.g. ``"text-embedding-3-small"``).
        umap_n_neighbors: ``n_neighbors`` knob forwarded to UMAP.
        umap_min_dist: ``min_dist`` knob forwarded to UMAP.
        umap_metric: Distance metric forwarded to UMAP.
        umap_random_state: Seed forwarded to UMAP for reproducibility.

    Returns:
        np.ndarray: ``(len(label_texts), 2)`` array of UMAP-reduced
            coordinates.

    Raises:
        ValueError: If *label_texts* is empty.
    """
    if not label_texts:
        raise ValueError("label_texts must not be empty")

    embedder = OpenAIEmbeddingLLM(model=embedding_model)
    embeddings = np.array([embedder.generate_embedding(t) for t in label_texts])

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_random_state,
    )
    return reducer.fit_transform(embeddings)
