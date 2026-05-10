"""Process-local memoization for label embedding matrices.

Both ``labeling_evaluation`` plugin variants (``embedding_map`` and
``dendrogram``) embed the same label list when they run in the same
pipeline. This module memoizes the resulting embedding matrix per
``(labels, provider, model)`` so repeat calls collapse to a dict
lookup, both within a single pipeline run and across runs in the
same worker process.

The cache lives in process memory (an :func:`functools.lru_cache`)
on purpose: storing numpy arrays under keys in the pipeline
``context`` dict would break ``mlflow.log_dict(context, ...)`` at
``app.core.pipeline_engine`` end-of-run, since ``np.ndarray`` is
not JSON-serializable.
"""

from functools import lru_cache

import numpy as np

from utils.embedder.registry import create_embedder

_CACHE_MAXSIZE = 4


@lru_cache(maxsize=_CACHE_MAXSIZE)
def _embed_label_tuple(
    labels: tuple[str, ...],
    provider: str,
    model: str,
) -> np.ndarray:
    """Embed *labels* via the registered provider; bounded LRU-cached.

    The cache key is ``(labels, provider, model)``. The provider
    must participate in the key — otherwise an OpenAI result could
    be silently returned for a Gemini request (or vice-versa).

    Args:
        labels: Label strings to embed, as a tuple so the value is
            hashable for ``lru_cache``.
        provider: Embedder provider name resolved by
            :func:`utils.embedder.registry.create_embedder`.
        model: Model identifier forwarded to the provider.

    Returns:
        np.ndarray: ``(len(labels), embedding_dim)`` array of
            embeddings, in input order.
    """
    embedder = create_embedder(provider, model)
    return np.array(embedder.generate_embeddings(list(labels)))


def embed_labels(
    label_texts: list[str],
    provider: str,
    model: str,
) -> np.ndarray:
    """Return embeddings for *label_texts*; memoized per (labels, provider, model).

    Pass arguments **positionally** at call sites — ``lru_cache``
    keys on positional identity, so passing any of them as kwargs
    would defeat the cache.

    Args:
        label_texts: Strings to embed. The same list passed twice
            in a row hits the cache; a reordered list does not.
        provider: Embedder provider name (e.g. ``"openai"``).
        model: Model identifier (e.g. ``"text-embedding-3-small"``).

    Returns:
        np.ndarray: ``(len(label_texts), embedding_dim)`` array of
            embeddings, in input order.

    Raises:
        ValueError: If *label_texts* is empty (refused up front so
            no zero-row entry is cached) or if *provider* is not a
            registered embedder provider (propagated from
            :func:`utils.embedder.registry.create_embedder`).
    """
    if not label_texts:
        raise ValueError("label_texts must not be empty")
    return _embed_label_tuple(tuple(label_texts), provider, model)
