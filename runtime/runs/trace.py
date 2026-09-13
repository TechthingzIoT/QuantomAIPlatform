from __future__ import annotations

from dataclasses import dataclass

from runtime.runs.run import Run
from runtime.tools.event import ToolExecutionEvent


@dataclass(frozen=True, slots=True)
class RunTrace:
    """Chronological execution evidence associated with a run."""

    run: Run
    tool_events: list[ToolExecutionEvent]
