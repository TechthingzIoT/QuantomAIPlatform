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


def test_executor_accepts_optional_execution_context():
    from runtime.tools.context import ToolExecutionContext
    from runtime.tools.protocol import ToolCall

    registry = ToolRegistry()
    registry.register(EchoTool())

    executor = ToolExecutor(registry)

    context = ToolExecutionContext(
        tool_call_id="call_1",
        agent_name="qair-agent",
    )

    result = executor.execute(
        ToolCall(
            name="echo",
            arguments={"text": "hello"},
        ),
        context=context,
    )

    assert result.ok is True
    assert result.result == "hello"


class ContextAwareEchoTool:

    name = "context_echo"

    description = "Echo text with execution context."

    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
            },
        },
        "required": ["text"],
    }

    @property
    def supports_context(self):
        return True

    def execute(self, arguments, *, context=None):
        return {
            "text": arguments["text"],
            "tool_call_id": (
                context.tool_call_id
                if context is not None
                else None
            ),
            "agent_name": (
                context.agent_name
                if context is not None
                else None
            ),
        }


def test_executor_passes_context_to_context_aware_tool():

    from runtime.tools.context import ToolExecutionContext

    registry = ToolRegistry()

    registry.register(ContextAwareEchoTool())

    executor = ToolExecutor(registry)

    context = ToolExecutionContext(
        tool_call_id="call_42",
        agent_name="qair-agent",
    )

    result = executor.execute(
        ToolCall(
            name="context_echo",
            arguments={"text": "hello"},
        ),
        context=context,
    )

    assert result.ok is True

    assert result.result == {
        "text": "hello",
        "tool_call_id": "call_42",
        "agent_name": "qair-agent",
    }


def test_executor_keeps_legacy_tool_api_unchanged():

    registry = ToolRegistry()

    registry.register(EchoTool())

    executor = ToolExecutor(registry)

    result = executor.execute(
        ToolCall(
            name="echo",
            arguments={"text": "legacy"},
        ),
    )

    assert result.ok is True

    assert result.result == "legacy"
