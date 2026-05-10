from abc import ABC, abstractmethod


class EmbeddingLLM(ABC):
    @abstractmethod
    def generate_embedding(self, text: str) -> list[float]: ...

    @abstractmethod
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]: ...
