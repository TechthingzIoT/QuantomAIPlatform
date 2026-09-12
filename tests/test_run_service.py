import pytest

from runtime.runs.registry import RunRegistry
from runtime.runs.service import RunService
from runtime.runs.status import RunStatus


def test_service_creates_its_own_registry():
    service = RunService()

    assert isinstance(service.registry, RunRegistry)


def test_service_accepts_custom_registry():
    registry = RunRegistry()

    service = RunService(registry=registry)

    assert service.registry is registry


def test_service_creates_and_registers_run():
    service = RunService()

    run = service.create("run-1")

    assert run.id == "run-1"
    assert service.get("run-1") is run


def test_service_returns_none_for_unknown_run():
    service = RunService()

    assert service.get("unknown") is None


def test_service_requires_registered_run():
    service = RunService()

    run = service.create("run-1")

    assert service.require("run-1") is run


def test_service_require_raises_for_unknown_run():
    service = RunService()

    with pytest.raises(KeyError):
        service.require("unknown")


def test_service_starts_run():
    service = RunService()

    service.create("run-1")

    run = service.start("run-1")

    assert run.status is RunStatus.RUNNING
    assert run.started_at is not None


def test_service_completes_run():
    service = RunService()

    service.create("run-1")
    service.start("run-1")

    run = service.complete("run-1")

    assert run.status is RunStatus.COMPLETED
    assert run.completed_at is not None


def test_service_fails_run():
    service = RunService()

    service.create("run-1")
    service.start("run-1")

    run = service.fail(
        "run-1",
        "Inference failed",
    )

    assert run.status is RunStatus.FAILED
    assert run.error == "Inference failed"
    assert run.completed_at is not None


def test_service_start_unknown_run_raises_key_error():
    service = RunService()

    with pytest.raises(KeyError):
        service.start("unknown")


def test_service_complete_unknown_run_raises_key_error():
    service = RunService()

    with pytest.raises(KeyError):
        service.complete("unknown")


def test_service_fail_unknown_run_raises_key_error():
    service = RunService()

    with pytest.raises(KeyError):
        service.fail(
            "unknown",
            "error",
        )


def test_service_uses_shared_registry():
    registry = RunRegistry()
    service = RunService(registry=registry)

    run = service.create("run-1")

    assert registry.get("run-1") is run
