from runtime.tools.context import ToolExecutionContext


def test_context_defaults():
    context = ToolExecutionContext()

    assert context.tool_call_id is None
    assert context.agent_name is None
    assert context.metadata == {}


def test_context_accepts_execution_metadata():
    context = ToolExecutionContext(
        tool_call_id="call_1",
        agent_name="qair-agent",
        metadata={
            "iteration": 1,
        },
    )

    assert context.tool_call_id == "call_1"
    assert context.agent_name == "qair-agent"
    assert context.metadata == {
        "iteration": 1,
    }


def test_context_is_immutable():
    context = ToolExecutionContext(
        tool_call_id="call_1",
    )

    try:
        context.tool_call_id = "call_2"
    except Exception:
        pass
    else:
        raise AssertionError(
            "ToolExecutionContext should be immutable."
        )
