from unittest.mock import MagicMock

from runtime.agents.agent import Agent
from runtime.inference.response import (
    InferenceResponse,
    ToolCallRequest,
)
from runtime.tools.registry import ToolRegistry


class EchoTool:
    name = "echo"
    description = "Echo a message."
    input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
            }
        },
        "required": ["message"],
    }

    def to_definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def execute(self, arguments):
        return arguments["message"]


def test_agent_run_with_outcome_records_tool_events_in_run_and_telemetry():
    runtime = MagicMock()

    tool_call = ToolCallRequest(
        id="call-1",
        name="echo",
        arguments={
            "message": "hello",
        },
    )

    runtime.generate.side_effect = [
        InferenceResponse(
            content=None,
            tool_calls=[tool_call],
        ),
        InferenceResponse(
            content="Tool execution complete.",
        ),
    ]

    registry = ToolRegistry()
    registry.register(EchoTool())

    agent = Agent(
        runtime=runtime,
        tool_registry=registry,
        name="test-agent",
    )

    outcome = agent.run_with_outcome(
        "Use the echo tool."
    )

    assert outcome.content == "Tool execution complete."

    assert len(outcome.tool_events) == 1
    assert len(agent.telemetry) == 1

    run_event = outcome.tool_events[0]
    telemetry_event = agent.telemetry.events()[0]

    assert telemetry_event == run_event
    assert run_event.tool_name == "echo"
    assert run_event.ok is True
    assert run_event.tool_call_id == "call-1"
    assert run_event.agent_name == "test-agent"
    assert run_event.iteration == 0


def test_agent_run_returns_content_from_run_outcome():
    runtime = MagicMock()

    runtime.generate.return_value = InferenceResponse(
        content="Hello from QAIR."
    )

    agent = Agent(runtime=runtime)

    result = agent.run("Hello")

    assert result == "Hello from QAIR."
