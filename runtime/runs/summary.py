from __future__ import annotations

from dataclasses import dataclass

from runtime.runs.run import Run
from runtime.tools.metrics import ToolTelemetryMetrics


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Combined lifecycle and telemetry summary for a QAIR run."""

    run: Run
    tool_metrics: ToolTelemetryMetrics
