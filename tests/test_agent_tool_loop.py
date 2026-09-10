from unittest.mock import MagicMock

from runtime.agents.agent import Agent
from runtime.chat.history import ConversationHistory
from runtime.inference.response import (
    InferenceResponse,
    ToolCallRequest,
)
from runtime.tools.registry import ToolRegistry


class LabStatusTool:
    name = "get_lab_status"
    description = "Get the current status of the QAIR lab."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, arguments):
        return {
            "status": "operational",
            "temperature": 24.5,
        }


def test_agent_run_executes_tool_and_returns_final_answer():
    runtime = MagicMock()

    runtime.generate.side_effect = [
        InferenceResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="get_lab_status",
                    arguments={},
                )
            ],
        ),
        InferenceResponse(
            content=(
                "The QAIR lab is operational and "
                "the temperature is 24.5°C."
            ),
            tool_calls=[],
        ),
    ]

    registry = ToolRegistry()
    registry.register(LabStatusTool())

    agent = Agent(
        runtime=runtime,
        history=ConversationHistory(),
        tool_registry=registry,
    )

    agent.running = True

    result = agent.run(
        "What is the current status of the QAIR lab?"
    )

    assert result == (
        "The QAIR lab is operational and "
        "the temperature is 24.5°C."
    )

    assert runtime.generate.call_count == 2

    first_call = runtime.generate.call_args_list[0]
    second_call = runtime.generate.call_args_list[1]

    expected_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_lab_status",
                "description": (
                    "Get the current status of the QAIR lab."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    assert first_call.kwargs["tools"] == expected_tools
    assert second_call.kwargs["tools"] == expected_tools

    assert first_call.kwargs["use_knowledge"] is True
    assert second_call.kwargs["use_knowledge"] is True

    first_messages = first_call.args[0]
    second_messages = second_call.args[0]

    assert first_messages == [
        {
            "role": "user",
            "content": (
                "What is the current status of the QAIR lab?"
            ),
        }
    ]

    assert second_messages[0] == {
        "role": "user",
        "content": (
            "What is the current status of the QAIR lab?"
        ),
    }

    assert second_messages[1]["role"] == "assistant"
    assert second_messages[1]["content"] is None

    tool_calls = second_messages[1]["tool_calls"]

    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["name"] == "get_lab_status"
    assert tool_calls[0]["arguments"] == {}

    assert second_messages[2] == {
        "role": "tool",
        "content": (
            '{"ok": true, "result": {"status": "operational", '
            '"temperature": 24.5}}'
        ),
        "tool_call_id": "call_1",
    }

    final_message = agent.history.last()

    assert final_message is not None
    assert final_message.content == result
