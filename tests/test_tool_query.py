from runtime.tools.query import ToolTelemetryQuery


def test_query_defaults_to_no_filters():
    query = ToolTelemetryQuery()

    assert query.tool_name is None
    assert query.agent_name is None
    assert query.iteration is None
    assert query.ok is None


def test_query_accepts_tool_name():
    query = ToolTelemetryQuery(
        tool_name="search",
    )

    assert query.tool_name == "search"


def test_query_accepts_agent_name():
    query = ToolTelemetryQuery(
        agent_name="research_agent",
    )

    assert query.agent_name == "research_agent"


def test_query_accepts_iteration():
    query = ToolTelemetryQuery(
        iteration=2,
    )

    assert query.iteration == 2


def test_query_accepts_execution_status():
    query = ToolTelemetryQuery(
        ok=False,
    )

    assert query.ok is False


def test_query_supports_multiple_filters():
    query = ToolTelemetryQuery(
        tool_name="search",
        agent_name="research_agent",
        iteration=2,
        ok=True,
    )

    assert query.tool_name == "search"
    assert query.agent_name == "research_agent"
    assert query.iteration == 2
    assert query.ok is True


from runtime.tools.event import ToolExecutionEvent
from runtime.tools.telemetry import ToolExecutionTelemetry


def test_telemetry_query_returns_all_events_without_filters():
    telemetry = ToolExecutionTelemetry()

    first_event = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=10.0,
        ok=True,
        agent_name="research_agent",
        iteration=1,
    )

    second_event = ToolExecutionEvent(
        tool_name="browser",
        elapsed_ms=20.0,
        ok=False,
        agent_name="research_agent",
        iteration=2,
    )

    telemetry.record(first_event)
    telemetry.record(second_event)

    events = telemetry.query(
        ToolTelemetryQuery()
    )

    assert events == [
        first_event,
        second_event,
    ]


def test_telemetry_query_filters_by_tool_name():
    telemetry = ToolExecutionTelemetry()

    search_event = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=10.0,
        ok=True,
    )

    calculator_event = ToolExecutionEvent(
        tool_name="calculator",
        elapsed_ms=5.0,
        ok=True,
    )

    telemetry.record(search_event)
    telemetry.record(calculator_event)

    events = telemetry.query(
        ToolTelemetryQuery(
            tool_name="search",
        )
    )

    assert events == [search_event]


def test_telemetry_query_filters_by_agent_name():
    telemetry = ToolExecutionTelemetry()

    research_event = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=10.0,
        ok=True,
        agent_name="research_agent",
    )

    math_event = ToolExecutionEvent(
        tool_name="calculator",
        elapsed_ms=5.0,
        ok=True,
        agent_name="math_agent",
    )

    telemetry.record(research_event)
    telemetry.record(math_event)

    events = telemetry.query(
        ToolTelemetryQuery(
            agent_name="research_agent",
        )
    )

    assert events == [research_event]


def test_telemetry_query_filters_by_iteration():
    telemetry = ToolExecutionTelemetry()

    first_event = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=10.0,
        ok=True,
        iteration=1,
    )

    second_event = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=20.0,
        ok=True,
        iteration=2,
    )

    telemetry.record(first_event)
    telemetry.record(second_event)

    events = telemetry.query(
        ToolTelemetryQuery(
            iteration=2,
        )
    )

    assert events == [second_event]


def test_telemetry_query_filters_by_execution_status():
    telemetry = ToolExecutionTelemetry()

    success_event = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=10.0,
        ok=True,
    )

    failure_event = ToolExecutionEvent(
        tool_name="browser",
        elapsed_ms=20.0,
        ok=False,
    )

    telemetry.record(success_event)
    telemetry.record(failure_event)

    events = telemetry.query(
        ToolTelemetryQuery(
            ok=False,
        )
    )

    assert events == [failure_event]


def test_telemetry_query_combines_multiple_filters():
    telemetry = ToolExecutionTelemetry()

    matching_event = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=10.0,
        ok=True,
        agent_name="research_agent",
        iteration=2,
    )

    wrong_iteration = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=20.0,
        ok=True,
        agent_name="research_agent",
        iteration=1,
    )

    wrong_status = ToolExecutionEvent(
        tool_name="search",
        elapsed_ms=30.0,
        ok=False,
        agent_name="research_agent",
        iteration=2,
    )

    telemetry.record(matching_event)
    telemetry.record(wrong_iteration)
    telemetry.record(wrong_status)

    events = telemetry.query(
        ToolTelemetryQuery(
            tool_name="search",
            agent_name="research_agent",
            iteration=2,
            ok=True,
        )
    )

    assert events == [matching_event]


def test_telemetry_query_returns_empty_list_when_nothing_matches():
    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
        )
    )

    events = telemetry.query(
        ToolTelemetryQuery(
            tool_name="unknown",
        )
    )

    assert events == []


def test_telemetry_returns_metrics_for_query():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
            agent_name="research_agent",
            iteration=1,
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=20.0,
            ok=False,
            agent_name="research_agent",
            iteration=1,
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="calculator",
            elapsed_ms=5.0,
            ok=True,
            agent_name="math_agent",
            iteration=1,
        )
    )

    metrics = telemetry.metrics_for_query(
        ToolTelemetryQuery(
            tool_name="search",
            agent_name="research_agent",
        )
    )

    assert metrics.total_executions == 2
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 1
    assert metrics.success_rate == 50.0
    assert metrics.failure_rate == 50.0
    assert metrics.total_execution_time_ms == 30.0
    assert metrics.average_execution_time_ms == 15.0


def test_telemetry_returns_zero_metrics_for_query_with_no_matches():

    telemetry = ToolExecutionTelemetry()

    telemetry.record(
        ToolExecutionEvent(
            tool_name="search",
            elapsed_ms=10.0,
            ok=True,
        )
    )

    metrics = telemetry.metrics_for_query(
        ToolTelemetryQuery(
            tool_name="unknown",
        )
    )

    assert metrics.total_executions == 0
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.success_rate == 0.0
    assert metrics.failure_rate == 0.0
    assert metrics.total_execution_time_ms == 0.0
    assert metrics.average_execution_time_ms == 0.0
