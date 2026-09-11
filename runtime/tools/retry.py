from __future__ import annotations

from abc import ABC, abstractmethod


class RetryStrategy(ABC):
    """Base abstraction for tool retry delay strategies."""

    @abstractmethod
    def get_delay_seconds(
        self,
        attempt: int,
        *,
        base_delay_seconds: float,
    ) -> float:
        """Return the delay before the next retry."""


class FixedRetryStrategy(RetryStrategy):
    """Use the same delay for every retry."""

    def get_delay_seconds(
        self,
        attempt: int,
        *,
        base_delay_seconds: float,
    ) -> float:
        return base_delay_seconds


class ExponentialRetryStrategy(RetryStrategy):
    """Increase retry delay exponentially."""

    def get_delay_seconds(
        self,
        attempt: int,
        *,
        base_delay_seconds: float,
    ) -> float:
        return base_delay_seconds * (2**attempt)
