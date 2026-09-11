from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolExecutionConfig:
    """Configuration for QAIR tool execution."""

    timeout_seconds: float | None = None
    max_retries: int = 0

    def __post_init__(self) -> None:
        if (
            self.timeout_seconds is not None
            and self.timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries must be greater than or equal to zero."
            )
