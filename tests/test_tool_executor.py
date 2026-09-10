from runtime.tools.executor import ToolExecutor
from runtime.tools.protocol import ToolCall
from runtime.tools.registry import ToolRegistry


class EchoTool:
    name = "echo"
    description = "Echo supplied text."

    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
            },
        },
        "required": ["text"],
    }

    def execute(self, arguments):
        return arguments["text"]


def test_executor_runs_registered_tool():

    registry = ToolRegistry()

    registry.register(EchoTool())

    executor = ToolExecutor(registry)

    result = executor.execute(
        ToolCall(
            name="echo",
            arguments={
                "text": "QAIR",
            },
        )
    )

    assert result.ok is True

    assert result.result == "QAIR"


def test_executor_returns_unknown_tool_failure():

    registry = ToolRegistry()

    executor = ToolExecutor(registry)

    result = executor.execute(
        ToolCall(
            name="missing_tool",
            arguments={},
        )
    )

    assert result.ok is False

    assert result.error_type == "ValueError"

    assert result.error_message == (
        "Unknown tool: missing_tool"
    )


def test_executor_returns_validation_failure():

    registry = ToolRegistry()

    registry.register(EchoTool())

    executor = ToolExecutor(registry)

    result = executor.execute(
        ToolCall(
            name="echo",
            arguments={},
        )
    )

    assert result.ok is False

    assert result.error_type == "ValueError"

    assert result.error_message == (
        "Missing required argument: text"
    )


class BrokenTool:
    name = "broken"
    description = "Always fails."

    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, arguments):
        raise RuntimeError("Tool exploded")


def test_executor_returns_execution_failure():

    registry = ToolRegistry()

    registry.register(BrokenTool())

    executor = ToolExecutor(registry)

    result = executor.execute(
        ToolCall(
            name="broken",
            arguments={},
        )
    )

    assert result.ok is False

    assert result.error_type == "RuntimeError"

    assert result.error_message == "Tool exploded"
