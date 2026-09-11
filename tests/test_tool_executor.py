import time
from typing import ClassVar

from runtime.tools.config import ToolExecutionConfig
from runtime.tools.errors import ToolExecutionTimeoutError
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

    input_schema: ClassVar[dict] = {
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

    input_schema: ClassVar[dict] = {
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


class SlowTool:
    name = "slow"
    description = "Sleeps before returning."

    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, arguments):
        time.sleep(0.2)
        return "finished"


def test_executor_returns_timeout_failure():
    registry = ToolRegistry()
    registry.register(SlowTool())

    executor = ToolExecutor(
        registry,
        config=ToolExecutionConfig(
            timeout_seconds=0.05,
        ),
    )

    result = executor.execute(
        ToolCall(
            name="slow",
            arguments={},
        )
    )

    assert result.ok is False
    assert result.error_type == (
        ToolExecutionTimeoutError.__name__
    )


class RetryableFlakyTool:
    name = "retryable_flaky"

    description = "Fails once before succeeding."

    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self):
        self.calls = 0

    @property
    def supports_retry(self):
        return True

    def execute(self, arguments):
        self.calls += 1

        if self.calls == 1:
            raise RuntimeError("Temporary failure")

        return "recovered"


def test_executor_retries_retryable_tool():
    registry = ToolRegistry()

    tool = RetryableFlakyTool()
    registry.register(tool)

    executor = ToolExecutor(
        registry,
        config=ToolExecutionConfig(
            max_retries=1,
        ),
    )

    result = executor.execute(
        ToolCall(
            name="retryable_flaky",
            arguments={},
        )
    )

    assert result.ok is True
    assert result.result == "recovered"
    assert tool.calls == 2


class NonRetryableFlakyTool:
    name = "non_retryable_flaky"

    description = "Fails and must not be retried."

    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self):
        self.calls = 0

    def execute(self, arguments):
        self.calls += 1
        raise RuntimeError("Failure")


def test_executor_does_not_retry_non_retryable_tool():
    registry = ToolRegistry()

    tool = NonRetryableFlakyTool()
    registry.register(tool)

    executor = ToolExecutor(
        registry,
        config=ToolExecutionConfig(
            max_retries=3,
        ),
    )

    result = executor.execute(
        ToolCall(
            name="non_retryable_flaky",
            arguments={},
        )
    )

    assert result.ok is False
    assert result.error_type == "RuntimeError"
    assert tool.calls == 1


class AlwaysFailingRetryableTool:
    name = "always_failing_retryable"

    description = "Always fails."

    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self):
        self.calls = 0

    @property
    def supports_retry(self):
        return True

    def execute(self, arguments):
        self.calls += 1
        raise RuntimeError("Still failing")


def test_executor_stops_after_max_retries():
    registry = ToolRegistry()

    tool = AlwaysFailingRetryableTool()
    registry.register(tool)

    executor = ToolExecutor(
        registry,
        config=ToolExecutionConfig(
            max_retries=2,
        ),
    )

    result = executor.execute(
        ToolCall(
            name="always_failing_retryable",
            arguments={},
        )
    )

    assert result.ok is False
    assert result.error_type == "RuntimeError"

    # Initial attempt + 2 retries.
    assert tool.calls == 3
