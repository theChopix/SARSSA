from typing import List, Tuple
from abc import ABC, abstractmethod


class ChatLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str | List[Tuple[str, str]]) -> str:
        ...