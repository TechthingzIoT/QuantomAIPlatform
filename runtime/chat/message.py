"""
QAIR Chat Message

Conversation message models used by the QAIR runtime.

Responsibilities

----------------

- Represent chat messages
- Standardize conversation roles
- Represent assistant tool calls
- Represent tool execution results
- Provide serialization helpers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from runtime.inference.response import ToolCallRequest


class MessageRole(str, Enum):
    """Supported conversation roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ChatMessage:
    """Represents a single conversation message."""

    role: MessageRole
    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        """Validate message structure."""

        if not isinstance(self.role, MessageRole):
            raise TypeError("role must be a MessageRole.")

        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("content must be a string or None.")

        if not isinstance(self.tool_calls, list):
            raise TypeError("tool_calls must be a list.")

        for tool_call in self.tool_calls:
            if not isinstance(tool_call, ToolCallRequest):
                raise TypeError(
                    "tool_calls must contain ToolCallRequest objects."
                )

        if self.tool_call_id is not None:
            if not isinstance(self.tool_call_id, str):
                raise TypeError("tool_call_id must be a string or None.")

            normalized_id = self.tool_call_id.strip()

            if not normalized_id:
                raise ValueError("tool_call_id cannot be empty.")

            self.tool_call_id = normalized_id

        if self.role is MessageRole.TOOL:
            if self.content is None:
                raise ValueError("Tool messages must contain content.")

            if self.tool_call_id is None:
                raise ValueError(
                    "Tool messages must contain a tool_call_id."
                )

        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError(
                "Only assistant messages may contain tool calls."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to a serializable dictionary."""

        data: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }

        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": dict(tool_call.arguments),
                }
                for tool_call in self.tool_calls
            ]

        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        """Create a ChatMessage from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("Message data must be a dictionary.")

        if "role" not in data:
            raise ValueError("Message data missing required field: 'role'.")

        role = MessageRole(data["role"])

        content = data.get("content")

        raw_tool_calls = data.get("tool_calls", [])

        if not isinstance(raw_tool_calls, list):
            raise TypeError("tool_calls must be a list.")

        tool_calls = [
            ToolCallRequest(
                id=item["id"],
                name=item["name"],
                arguments=item["arguments"],
            )
            for item in raw_tool_calls
        ]

        return cls(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
        )

    def __str__(self) -> str:
        """Return a human-readable representation."""

        if self.tool_calls:
            names = ", ".join(
                tool_call.name for tool_call in self.tool_calls
            )
            return f"{self.role.value}: tool_calls=[{names}]"

        return f"{self.role.value}: {self.content}"
