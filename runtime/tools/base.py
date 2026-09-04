"""Base abstraction for QAIR tools."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract contract for a QAIR agent tool."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique tool name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool description."""
        raise NotImplementedError

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """Return the model-facing input schema."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> Any:
        """Execute the tool with the supplied arguments."""
        raise NotImplementedError
