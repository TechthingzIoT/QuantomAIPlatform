from __future__ import annotations

from enum import Enum


class RunStatus(str, Enum):
    """Lifecycle states for a QAIR execution run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
