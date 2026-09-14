from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime.context.execution import ExecutionContext


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Immutable context associated with a tool execution."""

    execution: ExecutionContext
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.execution, ExecutionContext):
            raise TypeError(
                "execution must be an ExecutionContext."
            )

        if self.tool_call_id is not None:
            if not isinstance(self.tool_call_id, str):
                raise TypeError(
                    "tool_call_id must be a string or None."
                )

            tool_call_id = self.tool_call_id.strip()

            if not tool_call_id:
                raise ValueError(
                    "tool_call_id cannot be empty."
                )

            object.__setattr__(
                self,
                "tool_call_id",
                tool_call_id,
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )
