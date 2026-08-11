from unittest.mock import MagicMock, patch

from runtime.chat.session import ChatSession
from runtime.chat.message import MessageRole


def make_session():
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

        session = ChatSession()
        session.engine = engine

        return session


def test_session_initialization():
    session = make_session()

    assert session.running is False
    assert len(session.history) == 1
    assert session.history.to_messages()[0]["role"] == "system"


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


def test_clear():
    session = make_session()

    session.ask("Hello")

    assert len(session.history) == 3

    session.clear()

    assert len(session.history) == 1

    messages = session.history.to_messages()

    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(
        "You are QAIR, the Quantom AI Runtime assistant"
    )

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

    handled = session.command("/clear")

    assert handled is True
    assert len(session.history) == 1

    messages = session.history.to_messages()

    assert messages[0]["role"] == "system"


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
