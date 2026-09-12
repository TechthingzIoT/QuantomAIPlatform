from __future__ import annotations

from collections.abc import Iterable

from runtime.tools.event import ToolExecutionEvent
from runtime.tools.metrics import ToolTelemetryMetrics
from runtime.tools.query import ToolTelemetryQuery
from runtime.tools.report import ToolTelemetryReport


class ToolExecutionTelemetry:

    """In-memory telemetry store for tool execution events."""

    def __init__(self) -> None:

        self._events: list[ToolExecutionEvent] = []

    def record(
        self,
        event: ToolExecutionEvent,
    ) -> None:

        """Record a tool execution event."""

        if not isinstance(event, ToolExecutionEvent):

            raise TypeError(
                "event must be a ToolExecutionEvent."
            )

        self._events.append(event)

    def events(self) -> list[ToolExecutionEvent]:

        """Return recorded execution events."""

        return list(self._events)

    def events_for_tool(
        self,
        tool_name: str,
    ) -> list[ToolExecutionEvent]:

        """Return execution events for a specific tool."""

        return [
            event
            for event in self._events
            if event.tool_name == tool_name
        ]

    def events_for_agent(
        self,
        agent_name: str,
    ) -> list[ToolExecutionEvent]:

        """Return execution events for a specific agent."""

        return [
            event
            for event in self._events
            if event.agent_name == agent_name
        ]

    def events_for_run(
        self,
        run_id: str,
    ) -> list[ToolExecutionEvent]:
        """Return events associated with a specific run."""

        return [
            event
            for event in self._events
            if event.run_id == run_id
        ]

    def events_for_iteration(
        self,
        iteration: int,
    ) -> list[ToolExecutionEvent]:
        """Return execution events for a specific iteration."""

        return [
            event
            for event in self._events
            if event.iteration == iteration
        ]

    def query(
        self,
        query: ToolTelemetryQuery,
    ) -> list[ToolExecutionEvent]:
        """Return execution events matching the query filters."""

        if not isinstance(query, ToolTelemetryQuery):
            raise TypeError(
                "query must be a ToolTelemetryQuery."
            )

        return [
            event
            for event in self._events
            if (
                query.tool_name is None
                or event.tool_name == query.tool_name
            )
            and (
                query.run_id is None
                or event.run_id == query.run_id
            )
            and (
                query.agent_name is None
                or event.agent_name == query.agent_name
            )
            and (
                query.iteration is None
                or event.iteration == query.iteration
            )
            and (
                query.ok is None
                or event.ok == query.ok
            )
        ]

    def successful(self) -> list[ToolExecutionEvent]:

        """Return successful execution events."""

        return [
            event
            for event in self._events
            if event.ok
        ]

    def failures(self) -> list[ToolExecutionEvent]:

        """Return failed execution events."""

        return [
            event
            for event in self._events
            if not event.ok
        ]

    def tool_names(self) -> list[str]:

        """Return unique tool names in execution order."""

        return list(
            dict.fromkeys(
                event.tool_name
                for event in self._events
            )
        )

    def agent_names(self) -> list[str]:
        """Return unique agent names in execution order."""

        return list(
            dict.fromkeys(
                event.agent_name
                for event in self._events
                if event.agent_name is not None
            )
        )

    def run_ids(self) -> list[str]:
        """Return unique run IDs in execution order."""

        return list(
            dict.fromkeys(
                event.run_id
                for event in self._events
                if event.run_id is not None
            )
        )

    def iterations(self) -> list[int]:
        """Return unique iteration numbers in execution order."""

        return list(
            dict.fromkeys(
                event.iteration
                for event in self._events
                if event.iteration is not None
            )
        )

    def total_executions(self) -> int:

        """Return the total number of executions."""

        return len(self._events)

    def successful_execution_count(self) -> int:

        """Return the number of successful executions."""

        return len(self.successful())

    def failed_execution_count(self) -> int:

        """Return the number of failed executions."""

        return len(self.failures())

    def success_rate(self) -> float:

        """Return the execution success rate as a percentage."""

        return self.metrics().success_rate

    def failure_rate(self) -> float:

        """Return the execution failure rate as a percentage."""

        return self.metrics().failure_rate

    def total_execution_time_ms(self) -> float:

        """Return total execution time in milliseconds."""

        return self.metrics().total_execution_time_ms

    def average_execution_time_ms(self) -> float:

        """Return average execution time in milliseconds."""

        return self.metrics().average_execution_time_ms

    def metrics(self) -> ToolTelemetryMetrics:

        """Return an aggregated snapshot of telemetry metrics."""

        return self._calculate_metrics(
            self._events
        )

    def metrics_for_query(
        self,
        query: ToolTelemetryQuery,
    ) -> ToolTelemetryMetrics:
        """Return aggregated telemetry metrics matching a query."""

        return self._calculate_metrics(
            self.query(query)
        )

    def metrics_for_tool(
        self,
        tool_name: str,
    ) -> ToolTelemetryMetrics:

        """Return aggregated telemetry metrics for a specific tool."""

        return self._calculate_metrics(
            self.events_for_tool(tool_name)
        )

    def metrics_for_agent(
        self,
        agent_name: str,
    ) -> ToolTelemetryMetrics:

        """Return aggregated telemetry metrics for a specific agent."""

        return self._calculate_metrics(
            self.events_for_agent(agent_name)
        )

    def metrics_for_run(
        self,
        run_id: str,
    ) -> ToolTelemetryMetrics:
        """Return aggregated telemetry metrics for a specific run."""

        return self._calculate_metrics(
            self.events_for_run(run_id)
        )

    def metrics_for_iteration(
        self,
        iteration: int,
    ) -> ToolTelemetryMetrics:
        """Return aggregated telemetry metrics for a specific iteration."""

        return self._calculate_metrics(
            self.events_for_iteration(iteration)
        )

    def _calculate_metrics(
        self,
        events: Iterable[ToolExecutionEvent],
    ) -> ToolTelemetryMetrics:

        """Calculate aggregated metrics for execution events."""

        events = list(events)

        total_executions = len(events)

        successful_executions = sum(
            event.ok
            for event in events
        )

        failed_executions = (
            total_executions
            - successful_executions
        )

        total_execution_time_ms = sum(
            event.elapsed_ms
            for event in events
        )

        if total_executions == 0:

            return ToolTelemetryMetrics(
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                success_rate=0.0,
                failure_rate=0.0,
                total_execution_time_ms=0.0,
                average_execution_time_ms=0.0,
            )

        success_rate = (
            successful_executions
            / total_executions
        ) * 100

        failure_rate = (
            failed_executions
            / total_executions
        ) * 100

        average_execution_time_ms = (
            total_execution_time_ms
            / total_executions
        )

        return ToolTelemetryMetrics(
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            success_rate=success_rate,
            failure_rate=failure_rate,
            total_execution_time_ms=total_execution_time_ms,
            average_execution_time_ms=(
                average_execution_time_ms
            ),
        )

    def report(self) -> ToolTelemetryReport:
        """Return a complete analytics report for recorded telemetry."""

        return ToolTelemetryReport(
            overall=self.metrics(),
            by_tool={
                tool_name: self.metrics_for_tool(tool_name)
                for tool_name in self.tool_names()
            },
            by_run={
                run_id: self.metrics_for_run(run_id)
                for run_id in self.run_ids()
            },
            by_agent={
                agent_name: self.metrics_for_agent(agent_name)
                for agent_name in self.agent_names()
            },
            by_iteration={
                iteration: self.metrics_for_iteration(iteration)
                for iteration in self.iterations()
            },
        )

    def clear(self) -> None:

        """Clear all recorded telemetry events."""

        self._events.clear()

    def __len__(self) -> int:

        """Return the number of recorded events."""

        return len(self._events)
