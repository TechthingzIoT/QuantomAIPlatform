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


def test_telemetry_returns_events_for_specific_tool():
    telemetry = ToolExecutionTelemetry()

    first = make_event(tool_name="search")
    second = make_event(tool_name="calculator")
    third = make_event(tool_name="search")

    telemetry.record(first)
    telemetry.record(second)
    telemetry.record(third)

    assert telemetry.events_for_tool(
        "search"
    ) == [first, third]


def test_telemetry_returns_unique_tool_names():
    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        make_event(tool_name="search")
    )
    telemetry.record(
        make_event(tool_name="calculator")
    )
    telemetry.record(
        make_event(tool_name="search")
    )

    assert telemetry.tool_names() == [
        "search",
        "calculator",
    ]


def test_telemetry_calculates_execution_counts():
    telemetry = ToolExecutionTelemetry()

    telemetry.record(make_event(ok=True))
    telemetry.record(make_event(ok=True))
    telemetry.record(make_event(ok=False))

    assert telemetry.total_executions() == 3
    assert telemetry.successful_execution_count() == 2
    assert telemetry.failed_execution_count() == 1


def test_telemetry_calculates_success_and_failure_rates():
    telemetry = ToolExecutionTelemetry()

    telemetry.record(make_event(ok=True))
    telemetry.record(make_event(ok=True))
    telemetry.record(make_event(ok=False))
    telemetry.record(make_event(ok=False))

    assert telemetry.success_rate() == 50.0
    assert telemetry.failure_rate() == 50.0


def test_telemetry_returns_zero_rates_when_empty():
    telemetry = ToolExecutionTelemetry()

    assert telemetry.success_rate() == 0.0
    assert telemetry.failure_rate() == 0.0


def test_telemetry_calculates_total_execution_time():
    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        make_event(elapsed_ms=10.5)
    )
    telemetry.record(
        make_event(elapsed_ms=25.5)
    )

    assert telemetry.total_execution_time_ms() == 36.0


def test_telemetry_calculates_average_execution_time():
    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        make_event(elapsed_ms=10.0)
    )
    telemetry.record(
        make_event(elapsed_ms=20.0)
    )
    telemetry.record(
        make_event(elapsed_ms=30.0)
    )

    assert telemetry.average_execution_time_ms() == 20.0


def test_telemetry_returns_zero_average_when_empty():
    telemetry = ToolExecutionTelemetry()

    assert telemetry.average_execution_time_ms() == 0.0


def test_telemetry_clear_removes_events():
    telemetry = ToolExecutionTelemetry()

    telemetry.record(make_event())

    assert len(telemetry) == 1

    telemetry.clear()

    assert len(telemetry) == 0
    assert telemetry.events() == []
