from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolTelemetryQuery:
    """Filter criteria for querying tool execution telemetry."""

    tool_name: str | None = None

    run_id: str | None = None

    agent_name: str | None = None

    iteration: int | None = None

    ok: bool | None = None
