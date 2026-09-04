from unittest.mock import MagicMock

import pytest

from runtime.agents.agent import Agent
from runtime.tools.base import Tool
from runtime.tools.registry import ToolRegistry


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo a message."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        }

    def execute(self, arguments: dict):
        return {"echo": arguments["message"]}


@pytest.fixture
def registry():
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


@pytest.fixture
def agent(registry):
    return Agent(
        runtime=MagicMock(),
    )


def test_execute_tool_returns_tool_result(registry):
    agent = Agent(
        runtime=MagicMock(),
        tool_registry=registry,
    )

    result = agent.execute_tool(
        "echo",
        {"message": "hello"},
    )

    assert result == {"echo": "hello"}


def test_execute_tool_rejects_unknown_tool():
    registry = ToolRegistry()
    agent = Agent(
        runtime=MagicMock(),
        tool_registry=registry,
    )

    with pytest.raises(ValueError, match="Unknown tool"):
        agent.execute_tool("missing", {})


def test_execute_tool_validates_arguments(registry):
    agent = Agent(
        runtime=MagicMock(),
        tool_registry=registry,
    )

    with pytest.raises(ValueError, match="Missing required argument"):
        agent.execute_tool("echo", {})


def test_execute_tool_does_not_execute_when_validation_fails():
    tool = MagicMock(spec=Tool)
    type(tool).name = property(lambda self: "echo")
    type(tool).description = property(lambda self: "Echo.")
    type(tool).input_schema = property(
        lambda self: {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        }
    )

    registry = ToolRegistry()
    registry.register(tool)

    agent = Agent(
        runtime=MagicMock(),
        tool_registry=registry,
    )

    with pytest.raises(ValueError, match="Missing required argument"):
        agent.execute_tool("echo", {})

    tool.execute.assert_not_called()


def test_execute_tool_passes_arguments_unchanged():
    tool = MagicMock(spec=Tool)
    type(tool).name = property(lambda self: "echo")
    type(tool).description = property(lambda self: "Echo.")
    type(tool).input_schema = property(
        lambda self: {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        }
    )
    tool.execute.return_value = "ok"

    registry = ToolRegistry()
    registry.register(tool)

    agent = Agent(
        runtime=MagicMock(),
        tool_registry=registry,
    )

    arguments = {"message": "hello"}

    result = agent.execute_tool("echo", arguments)

    assert result == "ok"
    tool.execute.assert_called_once_with(arguments)


def test_agent_uses_injected_tool_registry(registry):
    agent = Agent(
        runtime=MagicMock(),
        tool_registry=registry,
    )

    assert agent.tool_registry is registry


def test_agent_creates_default_tool_registry():
    agent = Agent(runtime=MagicMock())

    assert isinstance(agent.tool_registry, ToolRegistry)
    assert agent.tool_registry.list() == []
