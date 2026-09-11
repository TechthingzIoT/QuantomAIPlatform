from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)

class ToolExecutionConfig:

    """Configuration for QAIR tool execution."""

    timeout_seconds: float | None = None

    max_retries: int = 0

    retry_delay_seconds: float = 0.0

    retry_strategy: str = "fixed"

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

        if self.retry_delay_seconds < 0:
            raise ValueError(
                "retry_delay_seconds must be greater than or equal to zero."
            )

        if self.retry_strategy not in {
            "fixed",
            "exponential",
        }:
            raise ValueError(
                "retry_strategy must be either "
                "'fixed' or 'exponential'."
            )
