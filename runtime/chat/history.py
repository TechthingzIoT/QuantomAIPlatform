"""
QAIR Conversation History

Maintains the complete conversation history for a chat session.

Responsibilities
----------------
- Store messages
- Append new messages
- Export history
- Import history
- Clear history
- Build messages for inference
"""

from __future__ import annotations

import json
from pathlib import Path

from runtime.chat.message import ChatMessage


class ConversationHistory:
    """Container for an ordered conversation history."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    # ==================================================
    # Basic Operations
    # ==================================================

    def add(self, message: ChatMessage) -> None:
        """Append a message to the conversation."""

        if not isinstance(message, ChatMessage):
            raise TypeError(
                "Conversation history only accepts ChatMessage objects."
            )

        self.messages.append(message)

    def clear(self) -> None:
        """Remove all messages from the conversation."""

        self.messages.clear()

    def last(self) -> ChatMessage | None:
        """Return the most recent message, or None if empty."""

        if not self.messages:
            return None

        return self.messages[-1]

    # ==================================================
    # Statistics
    # ==================================================

    def count(self) -> int:
        """Return the number of messages."""

        return len(self.messages)

    def empty(self) -> bool:
        """Return True when the conversation contains no messages."""

        return not self.messages

    # ==================================================
    # Serialization
    # ==================================================

    def to_dict(self) -> list[dict[str, str]]:
        """Serialize the conversation to dictionaries."""

        return [message.to_dict() for message in self.messages]

    def to_messages(self) -> list[dict[str, str]]:
        """
        Return messages in the format expected by
        llama.cpp chat completion APIs.
        """

        return self.to_dict()

    @classmethod
    def from_dict(
        cls,
        data: list[dict[str, str]],
    ) -> "ConversationHistory":
        """Create conversation history from serialized messages."""

        if not isinstance(data, list):
            raise TypeError("Conversation history must be a list.")

        history = cls()

        for item in data:
            if not isinstance(item, dict):
                raise TypeError(
                    "Each conversation message must be a dictionary."
                )

            history.add(ChatMessage.from_dict(item))

        return history

    # ==================================================
    # Persistence
    # ==================================================

    def save(self, path: str | Path) -> None:
        """Save conversation history as JSON."""

        path = Path(path)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self.to_dict(),
                file,
                indent=4,
                ensure_ascii=False,
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ConversationHistory":
        """Load conversation history from JSON."""

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Conversation history not found: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return cls.from_dict(data)

    # ==================================================
    # Legacy Prompt Builder
    # ==================================================

    def prompt(self) -> str:
        """
        Build the legacy text representation.

        Retained for backward compatibility.

        New inference code should use ``to_messages()``.
        """

        return "\n".join(str(message) for message in self.messages)

    # ==================================================
    # Convenience
    # ==================================================

    def __len__(self) -> int:
        return self.count()

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, index: int) -> ChatMessage:
        return self.messages[index]