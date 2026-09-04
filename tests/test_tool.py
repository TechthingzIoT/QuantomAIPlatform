from typing import Any

import pytest

from runtime.tools.base import Tool


class ExampleTool(Tool):
    @property
    def name(self) -> str:
        return "example"

    @property
    def description(self) -> str:
        return "An example QAIR tool."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
        }

    def execute(self, arguments: dict[str, Any]) -> Any:
        return arguments["value"]


def test_tool_exposes_name():
    tool = ExampleTool()

    assert tool.name == "example"


def test_tool_exposes_description():
    tool = ExampleTool()

    assert tool.description == "An example QAIR tool."


def test_tool_exposes_input_schema():
    tool = ExampleTool()

    assert tool.input_schema == {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
        },
        "required": ["value"],
    }


def test_tool_executes_arguments():
    tool = ExampleTool()

    result = tool.execute({"value": "hello"})

    assert result == "hello"


def test_tool_is_abstract():
    with pytest.raises(TypeError):
        Tool()


def test_tool_schema_is_mapping():
    tool = ExampleTool()

    assert isinstance(tool.input_schema, dict)
