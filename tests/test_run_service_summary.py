import pytest

from runtime.runs.registry import RunRegistry
from runtime.runs.service import RunService
from runtime.tools.event import ToolExecutionEvent
from runtime.tools.telemetry import ToolExecutionTelemetry


def test_service_summary_returns_registered_run():
    service = RunService()

    run = service.create("run-1")

    summary = service.summary("run-1")

    assert summary.run is run


def test_service_summary_returns_run_metrics():
    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
            run_id="run-1",
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="browser",
            elapsed_ms=20.0,
            ok=False,
            run_id="run-1",
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="calculator",
            elapsed_ms=5.0,
            ok=True,
            run_id="run-2",
        )
    )

    service = RunService(
        telemetry=telemetry,
    )

    service.create("run-1")

    summary = service.summary("run-1")

    metrics = summary.tool_metrics

    assert metrics.total_executions == 2
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 1
    assert metrics.total_execution_time_ms == 30.0
    assert metrics.average_execution_time_ms == 15.0


def test_service_summary_returns_zero_metrics_for_run_without_tools():
    service = RunService()

    service.create("run-1")

    summary = service.summary("run-1")

    metrics = summary.tool_metrics

    assert metrics.total_executions == 0
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.total_execution_time_ms == 0.0


def test_service_summary_unknown_run_raises_key_error():
    service = RunService()

    with pytest.raises(KeyError):
        service.summary("unknown")


def test_service_summary_uses_shared_registry_and_telemetry():
    registry = RunRegistry()
    telemetry = ToolExecutionTelemetry()

    service = RunService(
        registry=registry,
        telemetry=telemetry,
    )

    run = service.create("run-1")

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
            run_id="run-1",
        )
    )

    summary = service.summary("run-1")

    assert registry.get("run-1") is run
    assert summary.run is run
    assert summary.tool_metrics.total_executions == 1
