from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolTelemetryMetrics:

    """Aggregated metrics for tool execution telemetry."""

    total_executions: int

    successful_executions: int

    failed_executions: int

    success_rate: float

    failure_rate: float

    total_execution_time_ms: float

    average_execution_time_ms: float
