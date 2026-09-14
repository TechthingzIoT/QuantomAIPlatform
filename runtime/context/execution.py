from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Immutable context associated with a QAIR execution."""

    run_id: str
    agent_name: str | None = None
    iteration: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str):
            raise TypeError(
                "run_id must be a string."
            )

        run_id = self.run_id.strip()

        if not run_id:
            raise ValueError(
                "run_id cannot be empty."
            )

        object.__setattr__(
            self,
            "run_id",
            run_id,
        )

        if self.agent_name is not None:
            if not isinstance(self.agent_name, str):
                raise TypeError(
                    "agent_name must be a string or None."
                )

            agent_name = self.agent_name.strip()

            if not agent_name:
                raise ValueError(
                    "agent_name cannot be empty."
                )

            object.__setattr__(
                self,
                "agent_name",
                agent_name,
            )

        if self.iteration is not None:
            if not isinstance(self.iteration, int):
                raise TypeError(
                    "iteration must be an integer or None."
                )

            if self.iteration < 0:
                raise ValueError(
                    "iteration cannot be negative."
                )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )
