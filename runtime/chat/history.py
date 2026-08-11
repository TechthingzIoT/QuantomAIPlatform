"""
=========================================================
QAIR Conversation History
=========================================================

Maintains the complete conversation history for a chat
session.

Responsibilities
----------------
• Store messages
• Append new messages
• Export history
• Import history
• Clear history
• Build prompts for inference

Author:
    TIOTAIROBOTIX
=========================================================
"""

from __future__ import annotations

import json
from pathlib import Path

from runtime.chat.message import ChatMessage


class ConversationHistory:
    """
    Conversation history container.
    """

    def __init__(self):
        self.messages: list[ChatMessage] = []

    # ==================================================
    # Basic Operations
    # ==================================================

    def add(self, message: ChatMessage) -> None:
        """
        Append a message.
        """
        self.messages.append(message)

    def clear(self) -> None:
        """
        Remove every message.
        """
        self.messages.clear()

    def last(self) -> ChatMessage | None:
        """
        Return the latest message.
        """
        if not self.messages:
            return None

        return self.messages[-1]

    # ==================================================
    # Statistics
    # ==================================================

    def count(self) -> int:
        return len(self.messages)

    def empty(self) -> bool:
        return len(self.messages) == 0

    # ==================================================
    # Serialization
    # ==================================================

    def to_dict(self) -> list[dict]:
        """
        Export history.
        """
        return [m.to_dict() for m in self.messages]

    def to_messages(self) -> list[dict]:
        """
        Return the conversation in the format expected by
        llama.cpp's chat completion API.
        """
        return [
            message.to_dict()
            for message in self.messages
        ]

    @classmethod
    def from_dict(
        cls,
        data: list[dict],
    ) -> "ConversationHistory":
        """
        Build history from dictionaries.
        """
        history = cls()

        for item in data:
            history.add(ChatMessage.from_dict(item))

        return history

    # ==================================================
    # Persistence
    # ==================================================

    def save(self, path: str | Path) -> None:
        """
        Save history as JSON.
        """
        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=4,
                ensure_ascii=False,
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ConversationHistory":
        """
        Load history from JSON.
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    # ==================================================
    # Legacy Prompt Builder
    # ==================================================

    def prompt(self) -> str:
        """
        Legacy prompt builder.

        Retained for backward compatibility.
        New inference should use to_messages().
        """
        lines = []

        for message in self.messages:
            lines.append(str(message))

        return "\n".join(lines)

    # ==================================================
    # Convenience
    # ==================================================

    def __len__(self) -> int:
        return self.count()

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, index):
        return self.messages[index]