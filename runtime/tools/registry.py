from __future__ import annotations

from runtime.tools.base import Tool


class ToolRegistry:
    """Registry for named QAIR tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by its name."""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a registered tool by name if present."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Return a registered tool by name, or None if missing."""
        return self._tools.get(name)

    def contains(self, name: str) -> bool:
        """Return whether a tool is registered under the given name."""
        return name in self._tools

    def list(self) -> list[Tool]:
        """Return registered tools in registration order."""
        return list(self._tools.values())
