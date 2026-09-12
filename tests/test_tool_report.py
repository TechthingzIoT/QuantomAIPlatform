from runtime.tools.event import ToolExecutionEvent
from runtime.tools.telemetry import ToolExecutionTelemetry


def test_telemetry_report_contains_overall_metrics():
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
            tool_name="browser",
            elapsed_ms=20.0,
            ok=False,
        )
    )

    report = telemetry.report()

    assert report.overall.total_executions == 2
    assert report.overall.successful_executions == 1
    assert report.overall.failed_executions == 1
    assert report.overall.total_execution_time_ms == 30.0


def test_telemetry_report_contains_metrics_by_tool():
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
        )
    )

    telemetry.record(
        ToolExecutionEvent(
            tool_name="calculator",
            elapsed_ms=5.0,
            ok=True,
        )
    )

    report = telemetry.report()

    assert report.by_tool["search"].total_executions == 2
    assert report.by_tool["search"].successful_executions == 1
    assert report.by_tool["search"].failed_executions == 1

    assert report.by_tool["calculator"].total_executions == 1
    assert report.by_tool["calculator"].successful_executions == 1


def test_telemetry_report_contains_metrics_by_agent():
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

    report = telemetry.report()

    assert report.by_agent["research_agent"].total_executions == 2
    assert report.by_agent["research_agent"].failed_executions == 1

    assert report.by_agent["math_agent"].total_executions == 1
    assert report.by_agent["math_agent"].successful_executions == 1


def test_telemetry_report_contains_metrics_by_iteration():
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
            iteration=2,
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

    report = telemetry.report()

    assert report.by_iteration[1].total_executions == 1
    assert report.by_iteration[1].successful_executions == 1

    assert report.by_iteration[2].total_executions == 2
    assert report.by_iteration[2].successful_executions == 1
    assert report.by_iteration[2].failed_executions == 1


def test_empty_telemetry_report_contains_empty_groupings():
    telemetry = ToolExecutionTelemetry()

    report = telemetry.report()

    assert report.overall.total_executions == 0
    assert report.by_tool == {}
    assert report.by_agent == {}
    assert report.by_iteration == {}
