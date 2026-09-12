from __future__ import annotations

from dataclasses import dataclass

from runtime.tools.metrics import ToolTelemetryMetrics


@dataclass(frozen=True, slots=True)
class ToolTelemetryReport:
    """Aggregated analytics report for tool execution telemetry."""

    overall: ToolTelemetryMetrics

    by_tool: dict[str, ToolTelemetryMetrics]

    by_agent: dict[str, ToolTelemetryMetrics]

    by_iteration: dict[int, ToolTelemetryMetrics]
