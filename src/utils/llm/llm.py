from abc import ABC, abstractmethod


class ChatLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str | list[tuple[str, str]]) -> str: ...
