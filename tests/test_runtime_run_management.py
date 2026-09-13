from runtime.core.runtime import QAIRRuntime
from runtime.runs.query import RunQuery
from runtime.runs.status import RunStatus


def test_runtime_can_create_and_get_run():
    runtime = QAIRRuntime()

    run = runtime.create_run("run-1")

    assert run.id == "run-1"
    assert runtime.get_run("run-1") is run


def test_runtime_can_require_run():
    runtime = QAIRRuntime()

    run = runtime.create_run("run-1")

    assert runtime.require_run("run-1") is run


def test_runtime_can_list_runs():
    runtime = QAIRRuntime()

    first = runtime.create_run("run-1")
    second = runtime.create_run("run-2")

    assert runtime.list_runs() == [
        first,
        second,
    ]


def test_runtime_can_query_runs():
    runtime = QAIRRuntime()

    pending = runtime.create_run("pending")
    running = runtime.create_run("running")

    runtime.run_service.start(running.id)

    result = runtime.query_runs(
        RunQuery(status=RunStatus.RUNNING)
    )

    assert result == [running]
    assert pending not in result


def test_runtime_can_get_run_summary():
    runtime = QAIRRuntime()

    run = runtime.create_run("run-1")

    summary = runtime.get_run_summary(run.id)

    assert summary.run is run


def test_runtime_can_remove_run():
    runtime = QAIRRuntime()

    run = runtime.create_run("run-1")

    removed = runtime.remove_run(run.id)

    assert removed is run
    assert runtime.get_run(run.id) is None
