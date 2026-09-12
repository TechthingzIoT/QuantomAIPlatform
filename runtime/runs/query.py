from __future__ import annotations

from dataclasses import dataclass

from runtime.runs.status import RunStatus


@dataclass(frozen=True, slots=True)
class RunQuery:
    """Filter criteria for querying runtime runs."""

    status: RunStatus | None = None
