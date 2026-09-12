from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from runtime.runs.status import RunStatus


@dataclass(slots=True)
class Run:
    """Represents the lifecycle of a QAIR execution run."""

    id: str
    agent_name: str | None = None
    status: RunStatus = RunStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    def start(self) -> None:
        """Mark the run as running."""

        if self.status is not RunStatus.PENDING:
            raise RuntimeError(
                "Only pending runs can be started."
            )

        self.status = RunStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def complete(self) -> None:
        """Mark the run as completed."""

        if self.status is not RunStatus.RUNNING:
            raise RuntimeError(
                "Only running runs can be completed."
            )

        self.status = RunStatus.COMPLETED
        self.completed_at = datetime.now(UTC)

    def fail(self, error: str) -> None:
        """Mark the run as failed."""

        if self.status is not RunStatus.RUNNING:
            raise RuntimeError(
                "Only running runs can fail."
            )

        if not isinstance(error, str):
            raise TypeError(
                "error must be a string."
            )

        self.status = RunStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(UTC)
