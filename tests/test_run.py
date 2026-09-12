import pytest

from runtime.runs.run import Run
from runtime.runs.status import RunStatus


def test_run_initializes_pending():

    run = Run(id="run-1")

    assert run.id == "run-1"
    assert run.agent_name is None
    assert run.status is RunStatus.PENDING
    assert run.started_at is None
    assert run.completed_at is None
    assert run.error is None


def test_run_can_have_agent_name():

    run = Run(
        id="run-1",
        agent_name="research_agent",
    )

    assert run.agent_name == "research_agent"


def test_pending_run_can_start():

    run = Run(id="run-1")

    run.start()

    assert run.status is RunStatus.RUNNING
    assert run.started_at is not None
    assert run.completed_at is None


def test_running_run_can_complete():

    run = Run(id="run-1")

    run.start()
    run.complete()

    assert run.status is RunStatus.COMPLETED
    assert run.started_at is not None
    assert run.completed_at is not None


def test_running_run_can_fail():

    run = Run(id="run-1")

    run.start()
    run.fail("Inference failed.")

    assert run.status is RunStatus.FAILED
    assert run.error == "Inference failed."
    assert run.completed_at is not None


def test_completed_run_cannot_start_again():

    run = Run(id="run-1")

    run.start()
    run.complete()

    with pytest.raises(RuntimeError):
        run.start()


def test_pending_run_cannot_complete():

    run = Run(id="run-1")

    with pytest.raises(RuntimeError):
        run.complete()


def test_pending_run_cannot_fail():

    run = Run(id="run-1")

    with pytest.raises(RuntimeError):
        run.fail("Something failed.")


def test_fail_requires_string_error():

    run = Run(id="run-1")

    run.start()

    with pytest.raises(TypeError):
        run.fail(123)  # type: ignore[arg-type]
