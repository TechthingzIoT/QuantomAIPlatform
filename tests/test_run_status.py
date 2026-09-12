from runtime.runs.status import RunStatus


def test_run_status_values():

    assert RunStatus.PENDING.value == "pending"

    assert RunStatus.RUNNING.value == "running"

    assert RunStatus.COMPLETED.value == "completed"

    assert RunStatus.FAILED.value == "failed"


def test_run_status_is_string_enum():

    assert isinstance(RunStatus.RUNNING, str)
