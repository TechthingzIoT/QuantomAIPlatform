from unittest.mock import MagicMock

import pytest

from runtime.agents.agent import Agent
from runtime.chat.message import ChatMessage, MessageRole
from runtime.inference.response import (
    InferenceResponse,
    ToolCallRequest,
)
from runtime.tools.registry import ToolRegistry


@pytest.fixture
def runtime():
    runtime = MagicMock()
    runtime.generate.return_value = InferenceResponse(content="Hello from QAIR.")
    return runtime


@pytest.fixture
def agent(runtime):
    return Agent(runtime=runtime)


def test_agent_initializes(runtime):
    agent = Agent(runtime=runtime, name="test-agent")

    assert agent.name == "test-agent"
    assert agent.runtime is runtime
    assert agent.history.empty()
    assert agent.running is False


def test_agent_has_default_name(runtime):
    agent = Agent(runtime=runtime)

    assert agent.name


def test_agent_start(agent):
    agent.start()

    assert agent.running is True


def test_agent_start_is_idempotent(agent):
    agent.start()
    agent.start()

    assert agent.running is True


def test_agent_start_delegates_to_runtime(agent, runtime):
    agent.start()

    runtime.start.assert_called_once_with()
    assert agent.running is True


def test_agent_start_is_idempotent_for_runtime(agent, runtime):
    agent.start()
    agent.start()

    runtime.start.assert_called_once_with()
    assert agent.running is True


def test_agent_stop(agent):
    agent.start()

    agent.stop()

    assert agent.running is False


def test_agent_stop_is_idempotent(agent):
    agent.stop()

    assert agent.running is False


def test_agent_stop_delegates_to_runtime(agent, runtime):
    agent.start()
    agent.stop()

    runtime.stop.assert_called_once_with()
    assert agent.running is False


def test_agent_stop_is_idempotent_for_runtime(agent, runtime):
    agent.start()
    agent.stop()
    agent.stop()

    runtime.stop.assert_called_once_with()
    assert agent.running is False


def test_agent_run_starts_agent(runtime):
    agent = Agent(runtime=runtime)

    response = agent.run("Hello")

    assert response == "Hello from QAIR."
    assert agent.running is True


def test_agent_run_records_conversation(agent, runtime):
    response = agent.run("Hello")

    assert response == "Hello from QAIR."
    assert len(agent.history) == 2

    messages = agent.history.to_messages()

    assert messages[0] == {
        "role": "user",
        "content": "Hello",
    }
    assert messages[1] == {
        "role": "assistant",
        "content": "Hello from QAIR.",
    }


def test_agent_run_delegates_to_runtime(agent, runtime):
    agent.run("Hello")

    runtime.generate.assert_called_once_with(
        [
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        tools=None,
        use_knowledge=True,
    )


def test_agent_run_propagates_registered_tools_to_runtime(runtime):
    class LabStatusTool:
        name = "get_lab_status"
        description = "Return the current lab status."
        input_schema = {
            "type": "object",
            "properties": {},
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
            return "Lab operational."

    registry = ToolRegistry()
    registry.register(LabStatusTool())

    agent = Agent(
        runtime=runtime,
        tool_registry=registry,
    )

    agent.run("What is the lab status?")

    runtime.generate.assert_called_once_with(
        [
            {
                "role": "user",
                "content": "What is the lab status?",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_lab_status",
                    "description": "Return the current lab status.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        ],
        use_knowledge=True,
    )


def test_agent_preserves_conversation(agent, runtime):
    agent.run("Hello")
    agent.run("What did I say?")

    assert runtime.generate.call_count == 2

    second_messages = runtime.generate.call_args_list[1].args[0]

    assert second_messages == [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hello from QAIR.",
        },
        {
            "role": "user",
            "content": "What did I say?",
        },
    ]


def test_agent_reset_clears_history(agent):
    agent.run("Hello")

    agent.reset()

    assert agent.history.empty()


def test_agent_rejects_non_string_input(agent):
    with pytest.raises(TypeError, match="prompt must be a string"):
        agent.run(123)


def test_agent_rejects_empty_input(agent):
    with pytest.raises(ValueError, match="prompt cannot be empty"):
        agent.run("")


def test_agent_runtime_failure_does_not_record_assistant_message(
    agent,
    runtime,
):
    runtime.generate.side_effect = RuntimeError("Inference failure")

    with pytest.raises(RuntimeError, match="Inference failure"):
        agent.run("Hello")

    assert len(agent.history) == 1

    assert agent.history.last() is not None
    assert agent.history.last().role == MessageRole.USER


def test_agent_can_use_existing_history(runtime):
    agent = Agent(runtime=runtime)

    agent.history.add(
        ChatMessage(
            role=MessageRole.USER,
            content="Previous message",
        )
    )

    agent.run("Current message")

    messages = runtime.generate.call_args.args[0]

    assert messages[0]["content"] == "Previous message"
    assert messages[-1]["content"] == "Current message"


def test_agent_accepts_custom_history(runtime):
    from runtime.chat.history import ConversationHistory

    history = ConversationHistory()

    agent = Agent(
        runtime=runtime,
        history=history,
    )

    assert agent.history is history


def test_agent_integrates_with_real_runtime():
    from runtime.core.runtime import QAIRRuntime

    engine = MagicMock()
    engine.loaded = True
    engine.settings.max_tokens = 256
    engine.settings.context_size = 2048
    engine.count_tokens.return_value = 1
    engine.generate.return_value = InferenceResponse(content="Integrated response.")

    retriever = MagicMock()
    retriever.search.return_value = []

    runtime = QAIRRuntime(
        engine=engine,
        knowledge_retriever=retriever,
    )
    agent = Agent(runtime=runtime)

    response = agent.run("Test integration")

    assert response == "Integrated response."
    assert len(agent.history) == 2

    engine.generate.assert_called_once_with(
        [
            {
                "role": "user",
                "content": "Test integration",
            }
        ],
        tools=None,
        max_tokens=None,
        temperature=None,
        top_p=None,
    )


# ============================================================
# Tool Orchestration
# ============================================================


class EchoTool:
    @property
    def name(self):
        return "echo"

    @property
    def description(self):
        return "Echo the supplied text."

    @property
    def input_schema(self):
        return {
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


class AddTool:
    @property
    def name(self):
        return "add"

    @property
    def description(self):
        return "Add two integers."

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                },
                "b": {
                    "type": "integer",
                },
            },
            "required": ["a", "b"],
        }

    def execute(self, arguments):
        return arguments["a"] + arguments["b"]


def test_run_executes_tool_and_returns_final_response():
    runtime = MagicMock()

    runtime.generate.side_effect = [
        InferenceResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="echo",
                    arguments={"text": "Hello tool"},
                ),
            ],
        ),
        InferenceResponse(
            content="The tool returned Hello tool.",
        ),
    ]

    registry = ToolRegistry()
    registry.register(EchoTool())

    agent = Agent(
        runtime=runtime,
        tool_registry=registry,
    )

    result = agent.run("Use the echo tool.")

    assert result == "The tool returned Hello tool."
    assert runtime.generate.call_count == 2

    messages = agent.history.to_messages()

    assert messages[0] == {
        "role": "user",
        "content": "Use the echo tool.",
    }

    assert messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "name": "echo",
                "arguments": {
                    "text": "Hello tool",
                },
            },
        ],
    }

    assert messages[2] == {
        "role": "tool",
        "content": "Hello tool",
        "tool_call_id": "call_1",
    }

    assert messages[3] == {
        "role": "assistant",
        "content": "The tool returned Hello tool.",
    }


def test_run_passes_tool_history_to_next_inference():
    runtime = MagicMock()

    runtime.generate.side_effect = [
        InferenceResponse(
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="echo",
                    arguments={"text": "QAIR"},
                ),
            ],
        ),
        InferenceResponse(
            content="Done.",
        ),
    ]

    registry = ToolRegistry()
    registry.register(EchoTool())

    agent = Agent(
        runtime=runtime,
        tool_registry=registry,
    )

    agent.run("Run the tool.")

    second_call_messages = runtime.generate.call_args_list[1].args[0]

    assert second_call_messages == [
        {
            "role": "user",
            "content": "Run the tool.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "echo",
                    "arguments": {
                        "text": "QAIR",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": "QAIR",
            "tool_call_id": "call_1",
        },
    ]


def test_run_executes_multiple_tool_calls():
    runtime = MagicMock()

    runtime.generate.side_effect = [
        InferenceResponse(
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="add",
                    arguments={"a": 2, "b": 3},
                ),
                ToolCallRequest(
                    id="call_2",
                    name="add",
                    arguments={"a": 10, "b": 5},
                ),
            ],
        ),
        InferenceResponse(
            content="The calculations are complete.",
        ),
    ]

    registry = ToolRegistry()
    registry.register(AddTool())

    agent = Agent(
        runtime=runtime,
        tool_registry=registry,
    )

    result = agent.run("Calculate the values.")

    assert result == "The calculations are complete."

    messages = agent.history.to_messages()

    assert messages[2] == {
        "role": "tool",
        "content": "5",
        "tool_call_id": "call_1",
    }

    assert messages[3] == {
        "role": "tool",
        "content": "15",
        "tool_call_id": "call_2",
    }


def test_run_rejects_unknown_tool():
    runtime = MagicMock()

    runtime.generate.return_value = InferenceResponse(
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="unknown",
                arguments={},
            ),
        ],
    )

    agent = Agent(runtime=runtime)

    with pytest.raises(
        ValueError,
        match="Unknown tool: unknown",
    ):
        agent.run("Use an unknown tool.")


def test_run_rejects_invalid_tool_arguments():
    runtime = MagicMock()

    runtime.generate.return_value = InferenceResponse(
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="echo",
                arguments={},
            ),
        ],
    )

    registry = ToolRegistry()
    registry.register(EchoTool())

    agent = Agent(
        runtime=runtime,
        tool_registry=registry,
    )

    with pytest.raises(
        ValueError,
        match="Missing required argument: text",
    ):
        agent.run("Use the echo tool.")


def test_run_rejects_response_without_content_or_tool_calls():
    runtime = MagicMock()

    runtime.generate.return_value = InferenceResponse()

    agent = Agent(runtime=runtime)

    with pytest.raises(
        RuntimeError,
        match="Inference response did not contain assistant "
        "content or tool calls",
    ):
        agent.run("Do something.")


def test_run_stops_infinite_tool_loop():
    runtime = MagicMock()

    tool_response = InferenceResponse(
        tool_calls=[
            ToolCallRequest(
                id="call_loop",
                name="echo",
                arguments={"text": "loop"},
            ),
        ],
    )

    runtime.generate.return_value = tool_response

    registry = ToolRegistry()
    registry.register(EchoTool())

    agent = Agent(
        runtime=runtime,
        tool_registry=registry,
    )

    with pytest.raises(
        RuntimeError,
        match="Maximum tool execution iterations exceeded",
    ):
        agent.run("Loop forever.")

    assert (
        runtime.generate.call_count
        == Agent.MAX_TOOL_ITERATIONS
    )
