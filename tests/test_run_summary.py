import pytest

from runtime.runs.run import Run
from runtime.runs.summary import RunSummary
from runtime.tools.metrics import ToolTelemetryMetrics


def make_metrics() -> ToolTelemetryMetrics:
    return ToolTelemetryMetrics(
        total_executions=3,
        successful_executions=2,
        failed_executions=1,
        success_rate=66.66666666666667,
        failure_rate=33.333333333333336,
        total_execution_time_ms=30.0,
        average_execution_time_ms=10.0,
    )


def test_run_summary_stores_run():
    run = Run(id="run-1")

    summary = RunSummary(
        run=run,
        tool_metrics=make_metrics(),
    )

    assert summary.run is run


def test_run_summary_stores_tool_metrics():
    metrics = make_metrics()

    summary = RunSummary(
        run=Run(id="run-1"),
        tool_metrics=metrics,
    )

    assert summary.tool_metrics is metrics


def test_run_summary_is_immutable():
    summary = RunSummary(
        run=Run(id="run-1"),
        tool_metrics=make_metrics(),
    )

    with pytest.raises(
        AttributeError
    ):
        summary.tool_metrics = make_metrics()
