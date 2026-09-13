from runtime.events.event_type import RuntimeEventType
from runtime.events.store import RuntimeEventStore
from runtime.runs.service import RunService


def test_service_records_run_started_event():

    event_store = RuntimeEventStore()

    service = RunService(
        event_store=event_store,
    )

    service.create(
        "run-1",
        agent_name="test-agent",
    )

    service.start("run-1")

    events = event_store.events()

    assert len(events) == 1

    event = events[0]

    assert event.type is RuntimeEventType.RUN_STARTED

    assert event.run_id == "run-1"

    assert event.agent_name == "test-agent"


def test_service_records_run_completed_event():

    event_store = RuntimeEventStore()

    service = RunService(
        event_store=event_store,
    )

    service.create("run-1")

    service.start("run-1")

    service.complete("run-1")

    events = event_store.events()

    assert len(events) == 2

    event = events[1]

    assert event.type is RuntimeEventType.RUN_COMPLETED

    assert event.run_id == "run-1"


def test_service_records_run_failed_event():

    event_store = RuntimeEventStore()

    service = RunService(
        event_store=event_store,
    )

    service.create("run-1")

    service.start("run-1")

    service.fail(
        "run-1",
        "Inference failed.",
    )

    events = event_store.events()

    assert len(events) == 2

    event = events[1]

    assert event.type is RuntimeEventType.RUN_FAILED

    assert event.run_id == "run-1"

    assert event.data == {
        "error": "Inference failed.",
    }
