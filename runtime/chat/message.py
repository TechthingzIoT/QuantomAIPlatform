"""
=========================================================
QAIR Chat Message
=========================================================

Conversation message models used by the QAIR runtime.

Responsibilities
----------------
• Represent chat messages
• Standardize conversation roles
• Provide serialization helpers

Author:
    TIOTAIROBOTIX
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MessageRole(str, Enum):
    """
    Supported conversation roles.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(slots=True)
class ChatMessage:
    """
    Represents a single conversation message.
    """

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """
        Convert the message to a dictionary.

        Returns
        -------
        dict[str, str]
        """

        return {
            "role": self.role.value,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "ChatMessage":
        """
        Create a ChatMessage from a dictionary.
        """

        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
        )

    def __str__(self) -> str:
        """
        Human-readable representation.
        """

        return f"{self.role.value}: {self.content}"