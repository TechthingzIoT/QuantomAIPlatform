from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Immutable context associated with a tool execution."""

    tool_call_id: str | None = None

    run_id: str | None = None
    agent_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
