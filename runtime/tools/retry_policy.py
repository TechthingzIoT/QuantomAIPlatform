from __future__ import annotations

from typing import Protocol


class RetryPolicy(Protocol):
    """Determine whether a tool execution error may be retried."""

    def should_retry(
        self,
        error: Exception,
        *,
        attempt: int,
    ) -> bool:
        """Return whether the execution error should be retried."""


class DefaultRetryPolicy:
    """Default retry policy preserving standard tool retry behavior."""

    def should_retry(
        self,
        error: Exception,
        *,
        attempt: int,
    ) -> bool:
        """Return whether the execution error should be retried."""

        return True
