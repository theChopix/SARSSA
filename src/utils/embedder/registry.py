"""Provider registry and factory for embedding LLMs.

Maps a string provider name (e.g. ``"openai"``) to its concrete
:class:`EmbeddingLLM` subclass. Adding a new provider is a
two-line change: import the class and add an entry to
``_PROVIDERS``.
"""

from utils.embedder.embedder import EmbeddingLLM
from utils.embedder.openai_embedder import OpenAIEmbeddingLLM

_PROVIDERS: dict[str, type[EmbeddingLLM]] = {
    "openai": OpenAIEmbeddingLLM,
}


def create_embedder(provider: str, model: str) -> EmbeddingLLM:
    """Instantiate the embedder registered under *provider*.

    Args:
        provider: Provider name, e.g. ``"openai"``. Must be a key
            in the internal provider registry.
        model: Model identifier forwarded to the provider's
            constructor (e.g. ``"text-embedding-3-small"`` for
            OpenAI).

    Returns:
        EmbeddingLLM: A freshly constructed embedder instance for
            the requested provider/model.

    Raises:
        ValueError: If *provider* is not a known registry key. The
            error message lists the known providers so callers can
            recover.
    """
    if provider not in _PROVIDERS:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown embedding provider {provider!r}; known providers: {known}")
    return _PROVIDERS[provider](model=model)


def known_providers() -> list[str]:
    """Return the sorted list of registered embedding-provider names.

    Returns:
        list[str]: Provider names suitable for surfacing in a UI
            dropdown without leaking the internal registry dict.
    """
    return sorted(_PROVIDERS)
