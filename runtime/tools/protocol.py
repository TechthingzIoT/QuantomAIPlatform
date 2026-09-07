from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Provider-neutral request for executing a QAIR tool."""

    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")

        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("name cannot be empty.")

        if not isinstance(self.arguments, dict):
            raise TypeError("arguments must be a dictionary.")

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "arguments", dict(self.arguments))


def parse_tool_call(payload: object) -> ToolCall:
    """Parse a dictionary or JSON payload into a ToolCall."""
    if isinstance(payload, str):
        stripped_payload = payload.strip()
        if stripped_payload.startswith(("{", "[")):
            try:
                payload = json.loads(stripped_payload)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid tool call JSON.") from exc
        else:
            raise TypeError("Tool call payload must be a dictionary.")

    if not isinstance(payload, dict):
        raise TypeError("Tool call payload must be a dictionary.")

    required_fields = {"name", "arguments"}
    payload_fields = set(payload)

    missing = required_fields - payload_fields
    if missing:
        field = min(missing)
        raise ValueError(f"Tool call payload missing required field: '{field}'.")

    unexpected = payload_fields - required_fields
    if unexpected:
        field = min(unexpected)
        raise ValueError(f"Tool call payload has unexpected field: '{field}'.")

    return ToolCall(
        name=payload["name"],
        arguments=payload["arguments"],
    )
