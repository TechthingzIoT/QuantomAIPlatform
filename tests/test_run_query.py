import pytest

from runtime.runs.query import RunQuery
from runtime.runs.registry import RunRegistry
from runtime.runs.run import Run
from runtime.runs.status import RunStatus


def make_run(
    run_id: str,
    status: RunStatus = RunStatus.PENDING,
) -> Run:
    run = Run(id=run_id)

    if status == RunStatus.RUNNING:
        run.start()
    elif status == RunStatus.COMPLETED:
        run.start()
        run.complete()
    elif status == RunStatus.FAILED:
        run.start()
        run.fail("Test failure")

    return run


def test_run_query_defaults_to_no_filters():
    query = RunQuery()

    assert query.status is None


def test_run_query_accepts_status():
    query = RunQuery(
        status=RunStatus.RUNNING,
    )

    assert query.status == RunStatus.RUNNING


def test_registry_query_returns_all_runs_without_filters():
    registry = RunRegistry()

    first = make_run("run-1")
    second = make_run(
        "run-2",
        RunStatus.RUNNING,
    )

    registry.register(first)
    registry.register(second)

    assert registry.query(RunQuery()) == [
        first,
        second,
    ]


def test_registry_query_filters_runs_by_status():
    registry = RunRegistry()

    pending = make_run("run-1")
    running = make_run(
        "run-2",
        RunStatus.RUNNING,
    )
    completed = make_run(
        "run-3",
        RunStatus.COMPLETED,
    )

    registry.register(pending)
    registry.register(running)
    registry.register(completed)

    results = registry.query(
        RunQuery(
            status=RunStatus.RUNNING,
        )
    )

    assert results == [running]


def test_registry_query_returns_multiple_matching_runs():
    registry = RunRegistry()

    first = make_run(
        "run-1",
        RunStatus.FAILED,
    )
    second = make_run(
        "run-2",
        RunStatus.RUNNING,
    )
    third = make_run(
        "run-3",
        RunStatus.FAILED,
    )

    registry.register(first)
    registry.register(second)
    registry.register(third)

    results = registry.query(
        RunQuery(
            status=RunStatus.FAILED,
        )
    )

    assert results == [
        first,
        third,
    ]


def test_registry_query_returns_empty_for_no_matches():
    registry = RunRegistry()

    registry.register(
        make_run("run-1")
    )

    results = registry.query(
        RunQuery(
            status=RunStatus.COMPLETED,
        )
    )

    assert results == []


def test_registry_query_rejects_invalid_query():
    registry = RunRegistry()

    with pytest.raises(TypeError):
        registry.query(
            "invalid"  # type: ignore[arg-type]
        )
