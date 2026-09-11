import pytest

from runtime.tools.retry import (
    ExponentialRetryStrategy,
    FixedRetryStrategy,
)


@pytest.mark.parametrize(
    ("attempt", "base_delay_seconds"),
    [
        (0, 0.5),
        (1, 0.5),
        (2, 0.5),
    ],
)
def test_fixed_retry_strategy_returns_same_delay(
    attempt,
    base_delay_seconds,
):

    strategy = FixedRetryStrategy()

    delay = strategy.get_delay_seconds(
        attempt,
        base_delay_seconds=base_delay_seconds,
    )

    assert delay == base_delay_seconds


@pytest.mark.parametrize(
    ("attempt", "expected_delay"),
    [
        (0, 0.5),
        (1, 1.0),
        (2, 2.0),
        (3, 4.0),
    ],
)
def test_exponential_retry_strategy_increases_delay(
    attempt,
    expected_delay,
):

    strategy = ExponentialRetryStrategy()

    delay = strategy.get_delay_seconds(
        attempt,
        base_delay_seconds=0.5,
    )

    assert delay == expected_delay
