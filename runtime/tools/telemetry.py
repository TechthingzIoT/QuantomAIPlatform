from __future__ import annotations

from runtime.tools.event import ToolExecutionEvent


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
        total = self.total_executions()

        if total == 0:
            return 0.0

        return (
            self.successful_execution_count()
            / total
        ) * 100

    def failure_rate(self) -> float:
        """Return the execution failure rate as a percentage."""
        total = self.total_executions()

        if total == 0:
            return 0.0

        return (
            self.failed_execution_count()
            / total
        ) * 100

    def total_execution_time_ms(self) -> float:
        """Return total execution time in milliseconds."""
        return sum(
            event.elapsed_ms
            for event in self._events
        )

    def average_execution_time_ms(self) -> float:
        """Return average execution time in milliseconds."""
        total = self.total_executions()

        if total == 0:
            return 0.0

        return (
            self.total_execution_time_ms()
            / total
        )

    def clear(self) -> None:
        """Clear all recorded telemetry events."""
        self._events.clear()

    def __len__(self) -> int:
        """Return the number of recorded events."""
        return len(self._events)
