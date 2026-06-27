from abc import ABC, abstractmethod


class EmbeddingLLM(ABC):
    #: Curated model identifiers this provider exposes in the UI. Overridden per provider.
    KNOWN_MODELS: list[str] = []

    @abstractmethod
    def generate_embedding(self, text: str) -> list[float]: ...

    @abstractmethod
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]: ...
