"""
QAIR Chat Message

Conversation message models used by the QAIR runtime.

Responsibilities
----------------
- Represent chat messages
- Standardize conversation roles
- Provide serialization helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MessageRole(str, Enum):
    """Supported conversation roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(slots=True)
class ChatMessage:
    """Represents a single conversation message."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert the message to a serializable dictionary."""

        return {
            "role": self.role.value,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "ChatMessage":
        """Create a ChatMessage from a dictionary."""

        if "role" not in data:
            raise ValueError(
                "Message data missing required field: 'role'."
            )

        if "content" not in data:
            raise ValueError(
                "Message data missing required field: 'content'."
            )

        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
        )

    def __str__(self) -> str:
        """Return a human-readable representation."""

        return f"{self.role.value}: {self.content}"