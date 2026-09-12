from __future__ import annotations

from collections.abc import Iterable

from runtime.tools.event import ToolExecutionEvent
from runtime.tools.metrics import ToolTelemetryMetrics


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

    def clear(self) -> None:

        """Clear all recorded telemetry events."""

        self._events.clear()

    def __len__(self) -> int:

        """Return the number of recorded events."""

        return len(self._events)
