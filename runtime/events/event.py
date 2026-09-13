from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from runtime.events.event_type import RuntimeEventType


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Immutable event representing an occurrence during QAIR execution."""

    type: RuntimeEventType
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    run_id: str | None = None
    agent_name: str | None = None
    iteration: int | None = None

    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, RuntimeEventType):
            raise TypeError(
                "type must be a RuntimeEventType."
            )

        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                "timestamp must be a datetime."
            )

        if self.run_id is not None and not isinstance(
            self.run_id,
            str,
        ):
            raise TypeError(
                "run_id must be a string or None."
            )

        if self.agent_name is not None and not isinstance(
            self.agent_name,
            str,
        ):
            raise TypeError(
                "agent_name must be a string or None."
            )

        if self.iteration is not None and not isinstance(
            self.iteration,
            int,
        ):
            raise TypeError(
                "iteration must be an integer or None."
            )

        if not isinstance(self.data, dict):
            raise TypeError(
                "data must be a dictionary."
            )

        object.__setattr__(
            self,
            "data",
            dict(self.data),
        )
