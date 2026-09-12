import pytest

from runtime.runs.registry import RunRegistry
from runtime.runs.run import Run
from runtime.runs.status import RunStatus


def make_run(
    run_id: str = "run-1",
) -> Run:
    return Run(id=run_id)


def test_registry_initializes_empty():
    registry = RunRegistry()

    assert len(registry) == 0
    assert registry.runs() == []


def test_registry_registers_run():
    registry = RunRegistry()
    run = make_run()

    registry.register(run)

    assert len(registry) == 1
    assert registry.get(run.id) is run


def test_registry_rejects_invalid_run():
    registry = RunRegistry()

    with pytest.raises(TypeError):
        registry.register("invalid")  # type: ignore[arg-type]


def test_registry_rejects_duplicate_run_id():
    registry = RunRegistry()

    registry.register(make_run("run-1"))

    with pytest.raises(ValueError):
        registry.register(make_run("run-1"))


def test_registry_returns_none_for_unknown_run():
    registry = RunRegistry()

    assert registry.get("unknown") is None


def test_registry_requires_registered_run():
    registry = RunRegistry()
    run = make_run()

    registry.register(run)

    assert registry.require(run.id) is run


def test_registry_require_raises_for_unknown_run():
    registry = RunRegistry()

    with pytest.raises(KeyError):
        registry.require("unknown")


def test_registry_checks_run_existence():
    registry = RunRegistry()
    run = make_run()

    registry.register(run)

    assert registry.contains(run.id) is True
    assert registry.contains("unknown") is False


def test_registry_returns_runs_in_registration_order():
    registry = RunRegistry()

    first = make_run("run-1")
    second = make_run("run-2")
    third = make_run("run-3")

    registry.register(first)
    registry.register(second)
    registry.register(third)

    assert registry.runs() == [
        first,
        second,
        third,
    ]


def test_registry_returns_runs_with_status():
    registry = RunRegistry()

    pending = make_run("pending")
    running = make_run("running")
    completed = make_run("completed")

    running.start()
    completed.start()
    completed.complete()

    registry.register(pending)
    registry.register(running)
    registry.register(completed)

    assert registry.runs_with_status(
        RunStatus.PENDING
    ) == [pending]

    assert registry.runs_with_status(
        RunStatus.RUNNING
    ) == [running]

    assert registry.runs_with_status(
        RunStatus.COMPLETED
    ) == [completed]


def test_registry_rejects_invalid_status():
    registry = RunRegistry()

    with pytest.raises(TypeError):
        registry.runs_with_status(
            "completed"  # type: ignore[arg-type]
        )


def test_registry_removes_run():
    registry = RunRegistry()
    run = make_run()

    registry.register(run)

    removed = registry.remove(run.id)

    assert removed is run
    assert len(registry) == 0
    assert registry.get(run.id) is None


def test_registry_remove_returns_none_for_unknown_run():
    registry = RunRegistry()

    assert registry.remove("unknown") is None


def test_registry_clear_removes_all_runs():
    registry = RunRegistry()

    registry.register(make_run("run-1"))
    registry.register(make_run("run-2"))

    registry.clear()

    assert len(registry) == 0
    assert registry.runs() == []
