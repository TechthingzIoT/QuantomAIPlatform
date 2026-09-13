from __future__ import annotations

from runtime.events.event import RuntimeEvent
from runtime.events.event_type import RuntimeEventType
from runtime.events.store import RuntimeEventStore
from runtime.runs.query import RunQuery
from runtime.runs.registry import RunRegistry
from runtime.runs.run import Run
from runtime.runs.summary import RunSummary
from runtime.runs.trace import RunTrace
from runtime.tools.telemetry import ToolExecutionTelemetry


class RunService:
    """Service for managing QAIR execution run lifecycles."""

    def __init__(
        self,
        registry: RunRegistry | None = None,
        telemetry: ToolExecutionTelemetry | None = None,
        event_store: RuntimeEventStore | None = None,
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

        self.event_store = (
            event_store
            if event_store is not None
            else RuntimeEventStore()
        )

    def create(
        self,
        run_id: str,
        agent_name: str | None = None,
    ) -> Run:
        """Create and register a new run."""

        run = Run(
            id=run_id,
            agent_name=agent_name,
        )
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

        self.event_store.record(
            RuntimeEvent(
                type=RuntimeEventType.RUN_STARTED,
                run_id=run.id,
                agent_name=run.agent_name,
            )
        )

        return run

    def complete(
        self,
        run_id: str,
    ) -> Run:
        """Complete a registered run."""

        run = self.require(run_id)

        run.complete()

        self.event_store.record(
            RuntimeEvent(
                type=RuntimeEventType.RUN_COMPLETED,
                run_id=run.id,
                agent_name=run.agent_name,
            )
        )

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

    def trace(
        self,
        run_id: str,
    ) -> RunTrace:
        """Return execution trace for a registered run."""

        run = self.require(run_id)

        return RunTrace(
            run=run,
            tool_events=self.telemetry.events_for_run(
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

        self.event_store.record(
            RuntimeEvent(
                type=RuntimeEventType.RUN_FAILED,
                run_id=run.id,
                agent_name=run.agent_name,
                data={
                    "error": error,
                },
            )
        )

        return run
