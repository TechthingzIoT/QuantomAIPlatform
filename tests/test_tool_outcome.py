from typing import ClassVar

from runtime.tools.context import ToolExecutionContext
from runtime.tools.executor import ToolExecutor
from runtime.tools.protocol import ToolCall
from runtime.tools.registry import ToolRegistry


class EchoTool:

    name = "echo"

    description = "Echo supplied text."

    input_schema: ClassVar[dict] = {
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


class BrokenTool:

    name = "broken"

    description = "Always fails."

    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {},
    }

    def execute(self, arguments):
        raise RuntimeError("Tool exploded")


def test_executor_returns_execution_outcome():

    registry = ToolRegistry()

    registry.register(EchoTool())

    executor = ToolExecutor(registry)

    outcome = executor.execute_with_outcome(

        ToolCall(
            name="echo",
            arguments={
                "text": "hello",
            },
        )

    )

    assert outcome.result.ok is True

    assert outcome.result.result == "hello"

    assert outcome.event.tool_name == "echo"

    assert outcome.event.ok is True

    assert outcome.event.elapsed_ms >= 0


def test_execution_outcome_contains_context():

    registry = ToolRegistry()

    registry.register(EchoTool())

    executor = ToolExecutor(registry)

    context = ToolExecutionContext(

        tool_call_id="call_99",

        agent_name="qair-agent",

        metadata={
            "iteration": 3,
        },

    )

    outcome = executor.execute_with_outcome(

        ToolCall(
            name="echo",
            arguments={
                "text": "context",
            },
        ),

        context=context,

    )

    assert outcome.result.ok is True

    assert outcome.event.tool_call_id == "call_99"

    assert outcome.event.agent_name == "qair-agent"

    assert outcome.event.iteration == 3


def test_execution_outcome_records_failure():

    registry = ToolRegistry()

    registry.register(BrokenTool())

    executor = ToolExecutor(registry)

    outcome = executor.execute_with_outcome(

        ToolCall(
            name="broken",
            arguments={},
        )

    )

    assert outcome.result.ok is False

    assert outcome.event.ok is False

    assert outcome.event.error_type == "RuntimeError"

    assert outcome.event.error_message == "Tool exploded"


def test_executor_denied_by_policy_returns_failure():
    from runtime.tools.policy import AllowListToolPolicy

    registry = ToolRegistry()
    registry.register(EchoTool())

    executor = ToolExecutor(
        registry,
        policy=AllowListToolPolicy(set()),
    )

    outcome = executor.execute_with_outcome(
        ToolCall(
            name="echo",
            arguments={
                "text": "hello",
            },
        )
    )

    assert outcome.result.ok is False
    assert outcome.result.error_type == "PermissionError"

    assert outcome.event.ok is False
    assert outcome.event.error_type == "PermissionError"


def test_executor_allowed_by_policy_executes_tool():
    from runtime.tools.policy import AllowListToolPolicy

    registry = ToolRegistry()
    registry.register(EchoTool())

    executor = ToolExecutor(
        registry,
        policy=AllowListToolPolicy({"echo"}),
    )

    outcome = executor.execute_with_outcome(
        ToolCall(
            name="echo",
            arguments={
                "text": "hello",
            },
        )
    )

    assert outcome.result.ok is True
    assert outcome.result.result == "hello"

    assert outcome.event.ok is True
