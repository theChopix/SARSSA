from abc import ABC, abstractmethod
from typing import Any


class BasePlugin(ABC):
    """Base class for all pipeline plugins.

    Attributes:
        name: Optional human-readable display name. When set, the plugin
            registry uses this instead of the auto-derived name.
    """

    name: str | None = None

    @abstractmethod
    def run(self, context: dict, **params: Any) -> None:
        pass
