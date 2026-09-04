from unittest.mock import MagicMock

import pytest

from runtime.agents.agent import Agent
from runtime.chat.message import ChatMessage, MessageRole


@pytest.fixture
def runtime():
    runtime = MagicMock()
    runtime.generate.return_value = "Hello from QAIR."
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
    engine.generate.return_value = "Integrated response."

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
        max_tokens=None,
        temperature=None,
        top_p=None,
    )
