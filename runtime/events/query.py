from __future__ import annotations

from dataclasses import dataclass

from runtime.events.event_type import RuntimeEventType


@dataclass(frozen=True, slots=True)
class RuntimeEventQuery:
    """Filter criteria for querying runtime events."""

    type: RuntimeEventType | None = None
    run_id: str | None = None
    agent_name: str | None = None
    iteration: int | None = None
