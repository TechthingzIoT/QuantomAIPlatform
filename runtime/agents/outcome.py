from __future__ import annotations

from dataclasses import dataclass, field

from runtime.tools.event import ToolExecutionEvent


@dataclass(frozen=True, slots=True)
class AgentRunOutcome:
    """Result and operational trace from an agent run."""

    content: str

    tool_events: list[ToolExecutionEvent] = field(
        default_factory=list
    )
