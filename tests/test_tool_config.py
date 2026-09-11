import pytest

from runtime.tools.config import ToolExecutionConfig


def test_config_defaults_to_no_timeout():

    config = ToolExecutionConfig()

    assert config.timeout_seconds is None


def test_config_accepts_positive_timeout():

    config = ToolExecutionConfig(
        timeout_seconds=30.0,
    )

    assert config.timeout_seconds == 30.0


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
