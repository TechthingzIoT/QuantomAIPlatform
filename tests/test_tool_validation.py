import pytest

from runtime.tools.base import Tool
from runtime.tools.validation import validate_tool_arguments


class ExampleTool(Tool):
    @property
    def name(self) -> str:
        return "example"

    @property
    def description(self) -> str:
        return "An example tool."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }

    def execute(self, arguments: dict) -> str:
        return arguments["query"]


def test_valid_arguments_are_accepted():
    tool = ExampleTool()

    validate_tool_arguments(
        tool,
        {"query": "hello", "limit": 5},
    )


def test_missing_required_argument_is_rejected():
    tool = ExampleTool()

    with pytest.raises(ValueError, match="query"):
        validate_tool_arguments(tool, {})


def test_unexpected_argument_is_rejected():
    tool = ExampleTool()

    with pytest.raises(ValueError, match="unexpected"):
        validate_tool_arguments(
            tool,
            {"query": "hello", "unknown": True},
        )


def test_wrong_argument_type_is_rejected():
    tool = ExampleTool()

    with pytest.raises(TypeError, match="limit"):
        validate_tool_arguments(
            tool,
            {"query": "hello", "limit": "five"},
        )


def test_arguments_must_be_a_dictionary():
    tool = ExampleTool()

    with pytest.raises(TypeError, match="dictionary"):
        validate_tool_arguments(tool, ["hello"])


class PrimitiveTypesTool(Tool):
    @property
    def name(self) -> str:
        return "primitive_types"

    @property
    def description(self) -> str:
        return "Tests primitive argument types."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "enabled": {"type": "boolean"},
                "tags": {"type": "array"},
                "metadata": {"type": "object"},
            },
        }

    def execute(self, arguments: dict) -> dict:
        return arguments


def test_number_argument_is_accepted():
    tool = PrimitiveTypesTool()

    validate_tool_arguments(tool, {"score": 4.5})


def test_boolean_argument_is_accepted():
    tool = PrimitiveTypesTool()

    validate_tool_arguments(tool, {"enabled": True})


def test_array_argument_is_accepted():
    tool = PrimitiveTypesTool()

    validate_tool_arguments(tool, {"tags": ["ai", "robotics"]})


def test_object_argument_is_accepted():
    tool = PrimitiveTypesTool()

    validate_tool_arguments(tool, {"metadata": {"source": "qair"}})


def test_wrong_boolean_type_is_rejected():
    tool = PrimitiveTypesTool()

    with pytest.raises(TypeError, match="enabled"):
        validate_tool_arguments(tool, {"enabled": "true"})


def test_wrong_array_type_is_rejected():
    tool = PrimitiveTypesTool()

    with pytest.raises(TypeError, match="tags"):
        validate_tool_arguments(tool, {"tags": "ai"})


def test_wrong_object_type_is_rejected():
    tool = PrimitiveTypesTool()

    with pytest.raises(TypeError, match="metadata"):
        validate_tool_arguments(tool, {"metadata": ["qair"]})


class InvalidSchemaTool(ExampleTool):
    @property
    def input_schema(self):
        return None


def test_tool_schema_must_be_a_dictionary():
    tool = InvalidSchemaTool()

    with pytest.raises(TypeError, match="schema"):
        validate_tool_arguments(tool, {"query": "hello"})
