import pytest

from runtime.tools.event import ToolExecutionEvent
from runtime.tools.telemetry import ToolExecutionTelemetry


def make_event(
    *,
    tool_name: str = "test_tool",
    elapsed_ms: float = 10.0,
    ok: bool = True,
) -> ToolExecutionEvent:

    return ToolExecutionEvent(
        tool_name=tool_name,
        elapsed_ms=elapsed_ms,
        ok=ok,
    )


def test_telemetry_initializes_empty():

    telemetry = ToolExecutionTelemetry()

    assert len(telemetry) == 0
    assert telemetry.events() == []


def test_telemetry_records_event():

    telemetry = ToolExecutionTelemetry()

    event = make_event()

    telemetry.record(event)

    assert len(telemetry) == 1
    assert telemetry.events() == [event]


def test_telemetry_rejects_invalid_event():

    telemetry = ToolExecutionTelemetry()

    with pytest.raises(TypeError):
        telemetry.record("invalid")  # type: ignore[arg-type]


def test_telemetry_returns_successful_events():

    telemetry = ToolExecutionTelemetry()

    success = make_event(ok=True)

    failure = make_event(
        tool_name="failed_tool",
        ok=False,
    )

    telemetry.record(success)
    telemetry.record(failure)

    assert telemetry.successful() == [success]


def test_telemetry_returns_failed_events():

    telemetry = ToolExecutionTelemetry()

    success = make_event(ok=True)

    failure = make_event(
        tool_name="failed_tool",
        ok=False,
    )

    telemetry.record(success)
    telemetry.record(failure)

    assert telemetry.failures() == [failure]


def test_telemetry_calculates_total_execution_time():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        make_event(elapsed_ms=10.5)
    )

    telemetry.record(
        make_event(elapsed_ms=25.5)
    )

    assert telemetry.total_execution_time_ms() == 36.0


def test_telemetry_clear_removes_events():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(make_event())

    assert len(telemetry) == 1

    telemetry.clear()

    assert len(telemetry) == 0
    assert telemetry.events() == []
