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


def test_telemetry_returns_aggregated_metrics():
    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        make_event(
            tool_name="tool_a",
            elapsed_ms=10.0,
            ok=True,
        )
    )

    telemetry.record(
        make_event(
            tool_name="tool_b",
            elapsed_ms=20.0,
            ok=False,
        )
    )

    telemetry.record(
        make_event(
            tool_name="tool_c",
            elapsed_ms=30.0,
            ok=True,
        )
    )

    metrics = telemetry.metrics()

    assert metrics.total_executions == 3
    assert metrics.successful_executions == 2
    assert metrics.failed_executions == 1

    assert metrics.success_rate == (
        2 / 3
    ) * 100

    assert metrics.failure_rate == (
        1 / 3
    ) * 100

    assert metrics.total_execution_time_ms == 60.0
    assert metrics.average_execution_time_ms == 20.0


def test_telemetry_returns_zero_metrics_when_empty():
    telemetry = ToolExecutionTelemetry()

    metrics = telemetry.metrics()

    assert metrics.total_executions == 0
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.success_rate == 0.0
    assert metrics.failure_rate == 0.0
    assert metrics.total_execution_time_ms == 0
    assert metrics.average_execution_time_ms == 0.0


def test_telemetry_returns_events_for_agent():
    telemetry = ToolExecutionTelemetry()

    first_event = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=10.0,
        ok=True,
        agent_name="research_agent",
    )

    second_event = ToolExecutionEvent(
        tool_name="calculator",
        elapsed_ms=5.0,
        ok=True,
        agent_name="math_agent",
    )

    third_event = ToolExecutionEvent(
        tool_name="browser",
        elapsed_ms=20.0,
        ok=False,
        agent_name="research_agent",
    )

    telemetry.record(first_event)
    telemetry.record(second_event)
    telemetry.record(third_event)

    assert telemetry.events_for_agent(
        "research_agent"
    ) == [
        first_event,
        third_event,
    ]


def test_telemetry_returns_empty_events_for_unknown_agent():
    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
            agent_name="research_agent",
        )
    )

    assert telemetry.events_for_agent(
        "unknown_agent"
    ) == []


def test_telemetry_returns_metrics_for_specific_tool():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=20.0,
            ok=False,
            error_type="RuntimeError",
            error_message="Search failed",
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="calculator",
            elapsed_ms=5.0,
            ok=True,
        )
    )

    metrics = telemetry.metrics_for_tool("search")

    assert metrics.total_executions == 2
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 1
    assert metrics.success_rate == 50.0
    assert metrics.failure_rate == 50.0
    assert metrics.total_execution_time_ms == 30.0
    assert metrics.average_execution_time_ms == 15.0


def test_telemetry_returns_zero_metrics_for_unknown_tool():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
        )
    )

    metrics = telemetry.metrics_for_tool("unknown")

    assert metrics.total_executions == 0
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.success_rate == 0.0
    assert metrics.failure_rate == 0.0
    assert metrics.total_execution_time_ms == 0
    assert metrics.average_execution_time_ms == 0.0


def test_telemetry_returns_metrics_for_specific_agent():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
            agent_name="research_agent",
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="browser",
            elapsed_ms=20.0,
            ok=False,
            agent_name="research_agent",
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="calculator",
            elapsed_ms=5.0,
            ok=True,
            agent_name="math_agent",
        )
    )

    metrics = telemetry.metrics_for_agent(
        "research_agent"
    )

    assert metrics.total_executions == 2
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 1
    assert metrics.success_rate == 50.0
    assert metrics.failure_rate == 50.0
    assert metrics.total_execution_time_ms == 30.0
    assert metrics.average_execution_time_ms == 15.0


def test_telemetry_returns_zero_metrics_for_unknown_agent():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
            agent_name="research_agent",
        )
    )

    metrics = telemetry.metrics_for_agent(
        "unknown_agent"
    )

    assert metrics.total_executions == 0
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.success_rate == 0.0
    assert metrics.failure_rate == 0.0
    assert metrics.total_execution_time_ms == 0.0
    assert metrics.average_execution_time_ms == 0.0


def test_telemetry_returns_events_for_iteration():

    telemetry = ToolExecutionTelemetry()

    first_event = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=10.0,
        ok=True,
        iteration=1,
    )

    second_event = ToolExecutionEvent(
        tool_name="calculator",
        elapsed_ms=5.0,
        ok=True,
        iteration=2,
    )

    third_event = ToolExecutionEvent(
        tool_name="browser",
        elapsed_ms=20.0,
        ok=False,
        iteration=1,
    )

    telemetry.record(first_event)
    telemetry.record(second_event)
    telemetry.record(third_event)

    assert telemetry.events_for_iteration(1) == [
        first_event,
        third_event,
    ]


def test_telemetry_returns_empty_events_for_unknown_iteration():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
            iteration=1,
        )
    )

    assert telemetry.events_for_iteration(99) == []


def test_telemetry_returns_metrics_for_specific_iteration():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
            iteration=1,
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="browser",
            elapsed_ms=20.0,
            ok=False,
            iteration=1,
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="calculator",
            elapsed_ms=5.0,
            ok=True,
            iteration=2,
        )
    )

    metrics = telemetry.metrics_for_iteration(1)

    assert metrics.total_executions == 2
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 1
    assert metrics.success_rate == 50.0
    assert metrics.failure_rate == 50.0
    assert metrics.total_execution_time_ms == 30.0
    assert metrics.average_execution_time_ms == 15.0


def test_telemetry_returns_zero_metrics_for_unknown_iteration():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
            iteration=1,
        )
    )

    metrics = telemetry.metrics_for_iteration(99)

    assert metrics.total_executions == 0
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.success_rate == 0.0
    assert metrics.failure_rate == 0.0
    assert metrics.total_execution_time_ms == 0.0
    assert metrics.average_execution_time_ms == 0.0
