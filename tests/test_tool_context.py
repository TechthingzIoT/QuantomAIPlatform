from dataclasses import FrozenInstanceError

import pytest

from runtime.context.execution import ExecutionContext
from runtime.tools.context import ToolExecutionContext


def test_context_requires_execution_context():
    execution = ExecutionContext(
        run_id="run_1",
        agent_name="qair-agent",
        iteration=1,
    )

    context = ToolExecutionContext(
        execution=execution,
    )

    assert context.execution is execution
    assert context.tool_call_id is None
    assert context.metadata == {}


def test_context_accepts_execution_metadata():
    execution = ExecutionContext(
        run_id="run_1",
        agent_name="qair-agent",
        iteration=1,
    )

    context = ToolExecutionContext(
        execution=execution,
        tool_call_id="call_1",
        metadata={
            "source": "agent",
        },
    )

    assert context.tool_call_id == "call_1"
    assert context.execution.agent_name == "qair-agent"
    assert context.execution.iteration == 1
    assert context.metadata == {
        "source": "agent",
    }


def test_context_is_immutable():
    context = ToolExecutionContext(
        execution=ExecutionContext(
            run_id="run_1",
        ),
        tool_call_id="call_1",
    )

    with pytest.raises(FrozenInstanceError):
        context.tool_call_id = "call_2"


def test_context_rejects_invalid_execution_context():
    with pytest.raises(TypeError):
        ToolExecutionContext(
            execution="not-an-execution-context",
        )
