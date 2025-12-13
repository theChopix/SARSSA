
from abc import ABC, abstractmethod


class BasePlugin(ABC):
    @abstractmethod
    def run(self, context: dict, **params) -> dict:
        pass