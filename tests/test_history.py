from runtime.chat.history import ConversationHistory
from runtime.chat.message import ChatMessage, MessageRole


def test_history_starts_empty():
    history = ConversationHistory()

    assert history.empty()
    assert history.count() == 0
    assert len(history) == 0
    assert history.last() is None


def test_add_message():
    history = ConversationHistory()

    message = ChatMessage(
        role=MessageRole.USER,
        content="Hello QAIR",
    )

    history.add(message)

    assert history.count() == 1
    assert history.last() == message
    assert not history.empty()


def test_multiple_messages():
    history = ConversationHistory()

    history.add(
        ChatMessage(
            role=MessageRole.USER,
            content="Hello",
        )
    )

    history.add(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Hello there!",
        )
    )

    assert history.count() == 2
    assert len(history) == 2


def test_to_messages():
    history = ConversationHistory()

    history.add(
        ChatMessage(
            role=MessageRole.USER,
            content="Hello",
        )
    )

    history.add(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Hi!",
        )
    )

    assert history.to_messages() == [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hi!",
        },
    ]


def test_prompt():
    history = ConversationHistory()

    history.add(
        ChatMessage(
            role=MessageRole.USER,
            content="What is ESP32?",
        )
    )

    history.add(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="ESP32 is a microcontroller.",
        )
    )

    prompt = history.prompt()

    assert "user: What is ESP32?" in prompt
    assert "assistant: ESP32 is a microcontroller." in prompt


def test_clear():
    history = ConversationHistory()

    history.add(
        ChatMessage(
            role=MessageRole.USER,
            content="Hello",
        )
    )

    history.clear()

    assert history.empty()
    assert history.count() == 0
    assert history.last() is None


def test_iteration():
    history = ConversationHistory()

    first = ChatMessage(
        role=MessageRole.USER,
        content="First",
    )

    second = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Second",
    )

    history.add(first)
    history.add(second)

    messages = list(history)

    assert messages == [first, second]


def test_indexing():
    history = ConversationHistory()

    first = ChatMessage(
        role=MessageRole.USER,
        content="First",
    )

    history.add(first)

    assert history[0] == first


def test_history_round_trip():
    history = ConversationHistory()

    history.add(
        ChatMessage(
            role=MessageRole.USER,
            content="Hello",
        )
    )

    history.add(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Hi there!",
        )
    )

    restored = ConversationHistory.from_dict(
        history.to_dict()
    )

    assert restored.count() == history.count()
    assert restored.to_dict() == history.to_dict()