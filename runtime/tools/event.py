from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolExecutionEvent:

    """Operational information about a tool execution."""

    tool_name: str

    elapsed_ms: float

    ok: bool

    attempt_count: int = 1

    retry_count: int = 0

    tool_call_id: str | None = None

    run_id: str | None = None

    agent_name: str | None = None

    iteration: int | None = None

    error_type: str | None = None

    error_message: str | None = None
