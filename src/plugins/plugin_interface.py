from abc import ABC, abstractmethod
from typing import Any


class BasePlugin(ABC):
    @abstractmethod
    def run(self, context: dict, **params: Any) -> None:
        pass
