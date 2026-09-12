from __future__ import annotations

from runtime.runs.run import Run
from runtime.runs.status import RunStatus


class RunRegistry:
    """In-memory registry for QAIR execution runs."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    def register(self, run: Run) -> None:
        """Register a run."""

        if not isinstance(run, Run):
            raise TypeError("run must be a Run instance.")

        if run.id in self._runs:
            raise ValueError(
                f"Run '{run.id}' is already registered."
            )

        self._runs[run.id] = run

    def get(self, run_id: str) -> Run | None:
        """Return a run by ID, if it exists."""

        return self._runs.get(run_id)

    def require(self, run_id: str) -> Run:
        """Return a run by ID or raise KeyError."""

        run = self.get(run_id)

        if run is None:
            raise KeyError(
                f"Run '{run_id}' is not registered."
            )

        return run

    def contains(self, run_id: str) -> bool:
        """Return whether a run exists."""

        return run_id in self._runs

    def runs(self) -> list[Run]:
        """Return all registered runs in registration order."""

        return list(self._runs.values())

    def runs_with_status(
        self,
        status: RunStatus,
    ) -> list[Run]:
        """Return all runs with a specific status."""

        if not isinstance(status, RunStatus):
            raise TypeError(
                "status must be a RunStatus instance."
            )

        return [
            run
            for run in self._runs.values()
            if run.status is status
        ]

    def remove(self, run_id: str) -> Run | None:
        """Remove and return a run by ID."""

        return self._runs.pop(run_id, None)

    def clear(self) -> None:
        """Remove all registered runs."""

        self._runs.clear()

    def __len__(self) -> int:
        """Return the number of registered runs."""

        return len(self._runs)
