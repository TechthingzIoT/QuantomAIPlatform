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

    def failures(self) -> list[ToolExecutionEvent]:
        """Return failed execution events."""

        return [
            event
            for event in self._events
            if not event.ok
        ]

    def successful(self) -> list[ToolExecutionEvent]:
        """Return successful execution events."""

        return [
            event
            for event in self._events
            if event.ok
        ]

    def total_execution_time_ms(self) -> float:
        """Return total execution time in milliseconds."""

        return sum(
            event.elapsed_ms
            for event in self._events
        )

    def clear(self) -> None:
        """Clear all recorded telemetry events."""

        self._events.clear()

    def __len__(self) -> int:
        """Return the number of recorded events."""

        return len(self._events)
