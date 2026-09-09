"""
Provider-neutral inference response models.

Inference backends return these models instead of exposing
provider-specific response payloads to the QAIR runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """A tool call requested by an inference model."""

    id: str

    name: str

    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("id must be a string.")

        normalized_id = self.id.strip()

        if not normalized_id:
            raise ValueError("id cannot be empty.")

        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")

        normalized_name = self.name.strip()

        if not normalized_name:
            raise ValueError("name cannot be empty.")

        if not isinstance(self.arguments, dict):
            raise TypeError("arguments must be a dictionary.")

        object.__setattr__(self, "id", normalized_id)
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(slots=True, frozen=True)
class InferenceResponse:
    """Provider-neutral response returned by an inference backend."""

    content: str | None = None

    tool_calls: list[ToolCallRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("content must be a string or None.")

        if not isinstance(self.tool_calls, list):
            raise TypeError("tool_calls must be a list.")

        for tool_call in self.tool_calls:
            if not isinstance(tool_call, ToolCallRequest):
                raise TypeError(
                    "tool_calls must contain ToolCallRequest objects."
                )

        object.__setattr__(self, "tool_calls", list(self.tool_calls))
