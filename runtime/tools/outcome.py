from __future__ import annotations

from dataclasses import dataclass

from runtime.tools.event import ToolExecutionEvent
from runtime.tools.result import ToolExecutionResult


@dataclass(frozen=True, slots=True)
class ToolExecutionOutcome:
    """Combined result and operational event for a tool execution."""

    result: ToolExecutionResult

    event: ToolExecutionEvent
