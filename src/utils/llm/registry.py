"""Provider registry and factory for chat LLMs.

Maps a string provider name (e.g. ``"openai"``) to its concrete
:class:`ChatLLM` subclass. Adding a new provider is a two-line
change: import the class and add an entry to ``_PROVIDERS``.

Structurally parallel to :mod:`utils.embedder.registry`. No caching
layer here — chat-LLM generation is generally non-deterministic
(temperature > 0), prompts rarely repeat byte-for-byte across
calls, and per-cache-slot memory cost would be high.
"""

from utils.llm.llm import ChatLLM
from utils.llm.openai_llm import OpenAIChatLLM

_PROVIDERS: dict[str, type[ChatLLM]] = {
    "openai": OpenAIChatLLM,
}

_DEFAULT_MAX_TOKENS = 5000


def create_chat_llm(
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> ChatLLM:
    """Instantiate the chat LLM registered under *provider*.

    Args:
        provider: Provider name, e.g. ``"openai"``. Must be a key
            in the internal provider registry.
        model: Model identifier forwarded to the provider's
            constructor (e.g. ``"gpt-4o-mini"`` for OpenAI).
        temperature: Sampling temperature forwarded to the
            provider's constructor.
        max_tokens: Maximum number of tokens to generate. Defaults
            to ``5000`` to match :class:`OpenAIChatLLM`.

    Returns:
        ChatLLM: A freshly constructed chat-LLM instance for the
            requested provider/model.

    Raises:
        ValueError: If *provider* is not a known registry key. The
            error message lists the known providers so callers can
            recover.
    """
    if provider not in _PROVIDERS:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown chat LLM provider {provider!r}; known providers: {known}")
    return _PROVIDERS[provider](model=model, temperature=temperature, max_tokens=max_tokens)


def known_providers() -> list[str]:
    """Return the sorted list of registered chat-LLM-provider names.

    Returns:
        list[str]: Provider names suitable for surfacing in a UI
            dropdown without leaking the internal registry dict.
    """
    return sorted(_PROVIDERS)
