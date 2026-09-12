from __future__ import annotations

from dataclasses import dataclass, field

from runtime.tools.event import ToolExecutionEvent
from runtime.tools.report import ToolTelemetryReport


@dataclass(frozen=True, slots=True)
class AgentRunOutcome:
    """Result and operational trace from an agent run."""

    content: str

    run_id: str

    tool_events: list[ToolExecutionEvent] = field(
        default_factory=list
    )

    telemetry_report: ToolTelemetryReport | None = None
