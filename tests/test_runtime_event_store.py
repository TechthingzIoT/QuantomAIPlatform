import pytest

from runtime.events.event import RuntimeEvent
from runtime.events.event_type import RuntimeEventType
from runtime.events.query import RuntimeEventQuery
from runtime.events.store import RuntimeEventStore


def make_event(
    event_type=RuntimeEventType.RUN_STARTED,
    *,
    run_id=None,
    agent_name=None,
    iteration=None,
):

    return RuntimeEvent(
        type=event_type,
        run_id=run_id,
        agent_name=agent_name,
        iteration=iteration,
    )


def test_store_records_events():

    store = RuntimeEventStore()

    event = make_event()

    store.record(event)

    assert len(store) == 1
    assert store.events() == [event]


def test_store_preserves_chronological_order():

    store = RuntimeEventStore()

    first = make_event(
        RuntimeEventType.RUN_STARTED,
    )

    second = make_event(
        RuntimeEventType.INFERENCE_STARTED,
    )

    store.record(first)
    store.record(second)

    assert store.events() == [
        first,
        second,
    ]


def test_store_rejects_invalid_event():

    store = RuntimeEventStore()

    with pytest.raises(
        TypeError,
        match="RuntimeEvent",
    ):
        store.record("invalid")


def test_store_filters_events_by_run():

    store = RuntimeEventStore()

    first = make_event(run_id="run-1")
    second = make_event(run_id="run-2")

    store.record(first)
    store.record(second)

    assert store.events_for_run("run-1") == [
        first,
    ]


def test_store_filters_events_by_agent():

    store = RuntimeEventStore()

    first = make_event(agent_name="agent-a")
    second = make_event(agent_name="agent-b")

    store.record(first)
    store.record(second)

    assert store.events_for_agent("agent-a") == [
        first,
    ]


def test_store_filters_events_by_type():

    store = RuntimeEventStore()

    first = make_event(
        RuntimeEventType.RUN_STARTED,
    )

    second = make_event(
        RuntimeEventType.RUN_COMPLETED,
    )

    store.record(first)
    store.record(second)

    assert store.events_for_type(
        RuntimeEventType.RUN_STARTED,
    ) == [first]


def test_store_queries_events():

    store = RuntimeEventStore()

    first = make_event(
        RuntimeEventType.TOOL_COMPLETED,
        run_id="run-1",
        agent_name="agent-a",
        iteration=0,
    )

    second = make_event(
        RuntimeEventType.TOOL_COMPLETED,
        run_id="run-2",
        agent_name="agent-a",
        iteration=1,
    )

    third = make_event(
        RuntimeEventType.RUN_COMPLETED,
        run_id="run-1",
        agent_name="agent-b",
        iteration=0,
    )

    store.record(first)
    store.record(second)
    store.record(third)

    result = store.query(
        RuntimeEventQuery(
            type=RuntimeEventType.TOOL_COMPLETED,
            agent_name="agent-a",
        )
    )

    assert result == [
        first,
        second,
    ]


def test_store_query_rejects_invalid_query():

    store = RuntimeEventStore()

    with pytest.raises(
        TypeError,
        match="RuntimeEventQuery",
    ):
        store.query("invalid")
