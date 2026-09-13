from __future__ import annotations

from runtime.runs.query import RunQuery
from runtime.runs.registry import RunRegistry
from runtime.runs.run import Run
from runtime.runs.summary import RunSummary
from runtime.tools.telemetry import ToolExecutionTelemetry


class RunService:
    """Service for managing QAIR execution run lifecycles."""

    def __init__(
        self,
        registry: RunRegistry | None = None,
        telemetry: ToolExecutionTelemetry | None = None,
    ) -> None:
        self.registry = (
            registry
            if registry is not None
            else RunRegistry()
        )
        self.telemetry = (
            telemetry
            if telemetry is not None
            else ToolExecutionTelemetry()
        )

    def create(
        self,
        run_id: str,
    ) -> Run:
        """Create and register a new run."""

        run = Run(id=run_id)

        self.registry.register(run)

        return run

    def get(
        self,
        run_id: str,
    ) -> Run | None:
        """Return a run by ID, if registered."""

        return self.registry.get(run_id)

    def require(
        self,
        run_id: str,
    ) -> Run:
        """Return a registered run or raise KeyError."""

        return self.registry.require(run_id)

    def start(
        self,
        run_id: str,
    ) -> Run:
        """Start a registered run."""

        run = self.require(run_id)

        run.start()

        return run

    def complete(
        self,
        run_id: str,
    ) -> Run:
        """Complete a registered run."""

        run = self.require(run_id)

        run.complete()

        return run

    def runs(self) -> list[Run]:
        """Return all registered runs."""

        return self.registry.runs()

    def query(
        self,
        query: RunQuery,
    ) -> list[Run]:
        """Return runs matching the supplied query."""

        return self.registry.query(query)

    def remove(
        self,
        run_id: str,
    ) -> Run | None:
        """Remove and return a run by ID."""

        return self.registry.remove(run_id)

    def summary(
        self,
        run_id: str,
    ) -> RunSummary:
        """Return lifecycle and telemetry summary for a run."""

        run = self.require(run_id)

        return RunSummary(
            run=run,
            tool_metrics=self.telemetry.metrics_for_run(
                run_id
            ),
        )

    def fail(
        self,
        run_id: str,
        error: str,
    ) -> Run:
        """Fail a registered run."""

        run = self.require(run_id)

        run.fail(error)

        return run
