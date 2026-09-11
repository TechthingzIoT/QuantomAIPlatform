import pytest

from runtime.tools.config import ToolExecutionConfig


def test_config_defaults_to_no_timeout_and_no_retries():
    config = ToolExecutionConfig()

    assert config.timeout_seconds is None
    assert config.max_retries == 0


def test_config_accepts_positive_timeout():
    config = ToolExecutionConfig(
        timeout_seconds=30.0,
    )

    assert config.timeout_seconds == 30.0


def test_config_accepts_positive_retry_count():
    config = ToolExecutionConfig(
        max_retries=2,
    )

    assert config.max_retries == 2


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0,
        -1,
        -0.5,
    ],
)
def test_config_rejects_non_positive_timeout(
    timeout_seconds,
):
    with pytest.raises(
        ValueError,
        match="timeout_seconds must be greater than zero.",
    ):
        ToolExecutionConfig(
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.parametrize(
    "max_retries",
    [
        -1,
        -2,
    ],
)
def test_config_rejects_negative_retry_count(
    max_retries,
):
    with pytest.raises(
        ValueError,
        match=(
            "max_retries must be greater than or equal to zero."
        ),
    ):
        ToolExecutionConfig(
            max_retries=max_retries,
        )


def test_config_defaults_to_no_retry_delay():
    config = ToolExecutionConfig()

    assert config.retry_delay_seconds == 0.0


def test_config_accepts_positive_retry_delay():
    config = ToolExecutionConfig(
        retry_delay_seconds=0.5,
    )

    assert config.retry_delay_seconds == 0.5


@pytest.mark.parametrize(
    "retry_delay_seconds",
    [
        -1,
        -0.1,
    ],
)
def test_config_rejects_negative_retry_delay(
    retry_delay_seconds,
):
    with pytest.raises(
        ValueError,
        match=(
            "retry_delay_seconds must be greater than or equal to zero."
        ),
    ):
        ToolExecutionConfig(
            retry_delay_seconds=retry_delay_seconds,
        )
