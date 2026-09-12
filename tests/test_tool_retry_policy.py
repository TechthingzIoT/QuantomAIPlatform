from runtime.tools.retry_policy import (
    DefaultRetryPolicy,
)


def test_default_retry_policy_retries_errors():
    policy = DefaultRetryPolicy()

    assert policy.should_retry(
        RuntimeError("temporary failure"),
        attempt=0,
    ) is True


def test_default_retry_policy_retries_timeout_errors():
    policy = DefaultRetryPolicy()

    assert policy.should_retry(
        TimeoutError("temporary timeout"),
        attempt=0,
    ) is True


def test_default_retry_policy_retries_connection_errors():
    policy = DefaultRetryPolicy()

    assert policy.should_retry(
        ConnectionError("connection lost"),
        attempt=0,
    ) is True
