from __future__ import annotations

from runtime.events.event import RuntimeEvent
from runtime.events.query import RuntimeEventQuery


class RuntimeEventStore:
    """In-memory chronological store for QAIR runtime events."""

    def __init__(self) -> None:
        self._events: list[RuntimeEvent] = []

    def record(
        self,
        event: RuntimeEvent,
    ) -> None:
        """Record a runtime event."""

        if not isinstance(event, RuntimeEvent):
            raise TypeError(
                "event must be a RuntimeEvent."
            )

        self._events.append(event)

    def events(self) -> list[RuntimeEvent]:
        """Return all recorded events in chronological order."""

        return list(self._events)

    def events_for_run(
        self,
        run_id: str,
    ) -> list[RuntimeEvent]:
        """Return events associated with a run."""

        return [
            event
            for event in self._events
            if event.run_id == run_id
        ]

    def events_for_agent(
        self,
        agent_name: str,
    ) -> list[RuntimeEvent]:
        """Return events associated with an agent."""

        return [
            event
            for event in self._events
            if event.agent_name == agent_name
        ]

    def events_for_type(
        self,
        event_type,
    ) -> list[RuntimeEvent]:
        """Return events of a specific type."""

        return [
            event
            for event in self._events
            if event.type is event_type
        ]

    def query(
        self,
        query: RuntimeEventQuery,
    ) -> list[RuntimeEvent]:
        """Return events matching the supplied filters."""

        if not isinstance(query, RuntimeEventQuery):
            raise TypeError(
                "query must be a RuntimeEventQuery."
            )

        return [
            event
            for event in self._events
            if (
                query.type is None
                or event.type is query.type
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
        ]

    def __len__(self) -> int:
        return len(self._events)
