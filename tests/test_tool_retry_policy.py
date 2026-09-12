from runtime.tools.retry_policy import (
    DefaultRetryPolicy,
)


def test_default_retry_policy_retries_errors():

    policy = DefaultRetryPolicy()

    error = RuntimeError("temporary failure")

    assert policy.should_retry(
        error,
        attempt=0,
    ) is True
