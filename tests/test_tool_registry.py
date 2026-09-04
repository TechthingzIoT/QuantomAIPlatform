from typing import Any

import pytest

from runtime.tools.base import Tool
from runtime.tools.registry import ToolRegistry


class ExampleTool(Tool):
    @property
    def name(self) -> str:
        return "example"

    @property
    def description(self) -> str:
        return "An example tool."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    def execute(self, arguments: dict[str, Any]) -> Any:
        return arguments


def test_registry_starts_empty():
    registry = ToolRegistry()

    assert registry.list() == []


def test_registry_registers_tool():
    registry = ToolRegistry()
    tool = ExampleTool()

    registry.register(tool)

    assert registry.get("example") is tool


def test_registry_reports_registered_tool():
    registry = ToolRegistry()
    tool = ExampleTool()

    registry.register(tool)

    assert registry.contains("example") is True


def test_registry_reports_missing_tool():
    registry = ToolRegistry()

    assert registry.contains("missing") is False


def test_registry_lists_registered_tools():
    registry = ToolRegistry()
    tool = ExampleTool()

    registry.register(tool)

    assert registry.list() == [tool]


def test_registry_rejects_duplicate_tool_names():
    registry = ToolRegistry()
    first_tool = ExampleTool()
    second_tool = ExampleTool()

    registry.register(first_tool)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(second_tool)


def test_registry_unregisters_tool():
    registry = ToolRegistry()
    tool = ExampleTool()

    registry.register(tool)
    registry.unregister("example")

    assert registry.contains("example") is False
    assert registry.get("example") is None


def test_registry_unregister_missing_tool_is_safe():
    registry = ToolRegistry()

    registry.unregister("missing")

    assert registry.list() == []
