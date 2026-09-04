from unittest.mock import MagicMock, patch

import pytest

from runtime.chat.message import MessageRole
from runtime.chat.session import ChatSession
from runtime.prompts.selection import PromptSelector

# ============================================================
# Test Helper
# ============================================================

def make_session(prompt_selector=None):
    """
    Create a ChatSession with a mocked inference engine.

    This keeps session tests isolated from the real GGUF model.
    """

    with patch(
        "runtime.chat.session.InferenceEngine"
    ) as engine_class:

        engine = engine_class.return_value

        engine.generate.return_value = "Hello from QAIR."

        engine.summary.return_value = {
            "loaded": True,
            "model": "test-model.gguf",
            "context": 2048,
            "gpu_layers": 18,
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 256,
        }

        engine.model = MagicMock()
        engine.model.name = "test-model.gguf"

        session = ChatSession(
            prompt_selector=prompt_selector
        )

        # Replace the real engine with our mock.
        session.engine = engine

        return session


# ============================================================
# Initialization
# ============================================================

def test_session_initialization():
    session = make_session()

    assert session.running is False
    assert len(session.history) == 1
    assert session.engine is not None

    messages = session.history.to_messages()

    assert messages[0]["role"] == "system"
    assert messages[0]["content"].strip() != ""


# ============================================================
# Agent Integration
# ============================================================

def test_session_creates_agent():
    session = make_session()

    from runtime.agents.agent import Agent

    assert isinstance(session.agent, Agent)


def test_session_agent_shares_runtime():
    session = make_session()

    assert session.agent.runtime is session.runtime


def test_session_agent_shares_history():
    session = make_session()

    assert session.agent.history is session.history


def test_ask_delegates_to_agent():
    session = make_session()
    session.agent = MagicMock()
    session.agent.run.return_value = "Agent response."

    response = session.ask("Hello")

    assert response == "Agent response."
    session.agent.run.assert_called_once_with("Hello")


# ============================================================
# Conversation
# ============================================================

def test_ask_records_conversation():
    session = make_session()

    response = session.ask("Hello")

    assert response == "Hello from QAIR."
    assert len(session.history) == 3

    messages = session.history.to_messages()

    assert messages[0]["role"] == "system"

    assert messages[1] == {
        "role": "user",
        "content": "Hello",
    }

    assert messages[2] == {
        "role": "assistant",
        "content": "Hello from QAIR.",
    }


def test_ask_sends_full_conversation_to_engine():
    session = make_session()

    session.ask("Hello")
    session.ask("What did I say?")

    assert session.engine.generate.call_count == 2

    second_call = session.engine.generate.call_args_list[1]

    messages = second_call.args[0]

    assert messages[0]["role"] == "system"

    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"

    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hello from QAIR."

    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "What did I say?"


# ============================================================
# Conversation Management
# ============================================================

def test_clear():
    session = make_session()

    session.ask("Hello")

    assert len(session.history) == 3

    session.clear()

    # System prompt must remain.
    assert len(session.history) == 1

    messages = session.history.to_messages()

    assert messages[0]["role"] == "system"


# ============================================================
# Commands
# ============================================================

def test_exit_command():
    session = make_session()

    session.running = True

    handled = session.command("exit")

    assert handled is True
    assert session.running is False


def test_quit_command():
    session = make_session()

    session.running = True

    handled = session.command("quit")

    assert handled is True
    assert session.running is False


def test_clear_command():
    session = make_session()

    session.ask("Hello")

    assert len(session.history) == 3

    handled = session.command("/clear")

    assert handled is True
    assert len(session.history) == 1


def test_history_command():
    session = make_session()

    session.ask("Hello")

    handled = session.command("/history")

    assert handled is True


def test_stats_command():
    session = make_session()

    handled = session.command("/stats")

    assert handled is True


def test_help_command():
    session = make_session()

    handled = session.command("/help")

    assert handled is True


def test_unknown_command():
    session = make_session()

    handled = session.command("/unknown")

    assert handled is False


# ============================================================
# Startup / Shutdown
# ============================================================

def test_startup():
    session = make_session()

    session.startup()

    assert session.running is True
    session.engine.load.assert_called_once()


def test_shutdown():
    session = make_session()

    session.running = True

    session.shutdown()

    assert session.running is False
    session.engine.unload.assert_called_once()


# ============================================================
# System Message
# ============================================================

def test_system_message_is_first_message():
    session = make_session()

    messages = session.history.to_messages()

    assert messages[0]["role"] == "system"


def test_system_message_role():
    session = make_session()

    assert session.system_message.role == MessageRole.SYSTEM


def test_system_message_has_content():
    session = make_session()

    assert isinstance(
        session.system_message.content,
        str,
    )

    assert session.system_message.content.strip() != ""


# ============================================================
# Multiple Turns
# ============================================================

def test_multiple_questions_preserve_history():
    session = make_session()

    session.ask("Hello")
    session.ask("How are you?")
    session.ask("What is QAIR?")

    assert len(session.history) == 7

    messages = session.history.to_messages()

    assert messages[0]["role"] == "system"

    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"

    assert messages[2]["role"] == "assistant"

    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "How are you?"

    assert messages[4]["role"] == "assistant"

    assert messages[5]["role"] == "user"
    assert messages[5]["content"] == "What is QAIR?"

    assert messages[6]["role"] == "assistant"


# ============================================================
# Empty Input
# ============================================================

def test_ask_empty_string():
    session = make_session()

    response = session.ask("")

    assert response == "Hello from QAIR."
    assert len(session.history) == 3


# ============================================================
# Engine Interaction
# ============================================================

def test_ask_calls_engine_generate_once():
    session = make_session()

    session.ask("Test message")

    session.engine.generate.assert_called_once()


def test_startup_loads_engine_before_running():
    session = make_session()

    assert session.running is False

    session.startup()

    session.engine.load.assert_called_once()
    assert session.running is True


def test_shutdown_unloads_engine():
    session = make_session()

    session.running = True

    session.shutdown()

    session.engine.unload.assert_called_once()
    assert session.running is False


# ============================================================
# Prompt Selection
# ============================================================

def test_session_default_prompt():
    session = make_session()

    assert session.active_prompt == "assistant"


def test_session_uses_selected_prompt():
    prompt_selector = PromptSelector()

    session = make_session(
        prompt_selector=prompt_selector
    )

    session.set_prompt("embedded")

    assert session.active_prompt == "embedded"

    assert session.history[0].role == MessageRole.SYSTEM

    assert session.history[0].content == (
        session.prompt_selector.select("embedded")
    )


def test_switching_prompt_resets_history():
    session = make_session()

    session.ask("Hello")

    assert len(session.history) == 3

    session.set_prompt("robotics")

    assert len(session.history) == 1
    assert session.active_prompt == "robotics"

    assert (
        session.history[0].role
        == MessageRole.SYSTEM
    )


def test_clear_preserves_active_prompt():
    session = make_session()

    session.set_prompt("embedded")

    session.ask("Write ESP32 code.")

    session.clear()

    assert session.active_prompt == "embedded"
    assert len(session.history) == 1

    assert (
        session.history[0].role
        == MessageRole.SYSTEM
    )

    assert (
        session.history[0].content
        == session.prompt_selector.select("embedded")
    )


def test_unknown_prompt_does_not_change_active_prompt():
    session = make_session()

    original = session.active_prompt

    with pytest.raises(ValueError):
        session.set_prompt(
            "does_not_exist"
        )

    assert session.active_prompt == original


# ============================================================
# Prompt Commands
# ============================================================

def test_prompt_command_lists_prompts():
    session = make_session()

    handled = session.command("/prompt")

    assert handled is True


def test_prompt_command_switches_prompt():
    session = make_session()

    handled = session.command(
        "/prompt embedded"
    )

    assert handled is True
    assert session.active_prompt == "embedded"


def test_prompt_command_invalid_prompt():
    session = make_session()

    handled = session.command(
        "/prompt does_not_exist"
    )

    assert handled is True
    assert session.active_prompt == "assistant"

   