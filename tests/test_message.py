"""
=========================================================
QAIR Message Tests
=========================================================

Tests for:
    runtime.chat.message

Coverage:
    • MessageRole values
    • ChatMessage construction
    • Serialization
    • Deserialization
    • String representation
    • Round-trip consistency

Author:
    TIOTAIROBOTIX
=========================================================
"""

from runtime.chat.message import ChatMessage, MessageRole


def test_message_roles():
    """Verify supported message roles."""

    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"


def test_chat_message_creation():
    """Verify ChatMessage construction."""

    message = ChatMessage(
        role=MessageRole.USER,
        content="Hello QAIR",
    )

    assert message.role == MessageRole.USER
    assert message.content == "Hello QAIR"


def test_message_to_dict():
    """Verify message serialization."""

    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Hello from QAIR.",
    )

    data = message.to_dict()

    assert data == {
        "role": "assistant",
        "content": "Hello from QAIR.",
    }


def test_message_from_dict():
    """Verify message deserialization."""

    data = {
        "role": "user",
        "content": "What is QAIR?",
    }

    message = ChatMessage.from_dict(data)

    assert message.role == MessageRole.USER
    assert message.content == "What is QAIR?"


def test_message_string_representation():
    """Verify human-readable representation."""

    message = ChatMessage(
        role=MessageRole.SYSTEM,
        content="You are QAIR.",
    )

    assert str(message) == "system: You are QAIR."


def test_message_round_trip():
    """Verify serialization/deserialization consistency."""

    original = ChatMessage(
        role=MessageRole.USER,
        content="Test message",
    )

    restored = ChatMessage.from_dict(
        original.to_dict()
    )

    assert restored.role == original.role
    assert restored.content == original.content
    assert restored.to_dict() == original.to_dict()