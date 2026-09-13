import pytest

from runtime.runs.service import RunService
from runtime.tools.event import ToolExecutionEvent
from runtime.tools.telemetry import ToolExecutionTelemetry


def test_service_trace_returns_registered_run():

    service = RunService()

    run = service.create("run-1")

    trace = service.trace("run-1")

    assert trace.run is run


def test_service_trace_returns_events_for_run_only():

    telemetry = ToolExecutionTelemetry()

    first_event = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=10.0,
        ok=True,
        run_id="run-1",
    )

    second_event = ToolExecutionEvent(
        tool_name="calculator",
        elapsed_ms=20.0,
        ok=True,
        run_id="run-2",
    )

    third_event = ToolExecutionEvent(
        tool_name="browser",
        elapsed_ms=30.0,
        ok=False,
        run_id="run-1",
    )

    telemetry.record(first_event)
    telemetry.record(second_event)
    telemetry.record(third_event)

    service = RunService(
        telemetry=telemetry,
    )

    service.create("run-1")

    trace = service.trace("run-1")

    assert trace.tool_events == [
        first_event,
        third_event,
    ]


def test_service_trace_returns_empty_events_for_run_without_tools():

    service = RunService()

    service.create("run-1")

    trace = service.trace("run-1")

    assert trace.tool_events == []


def test_service_trace_unknown_run_raises_key_error():

    service = RunService()

    with pytest.raises(KeyError):

        service.trace("unknown")
