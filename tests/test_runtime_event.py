from datetime import UTC, datetime

import pytest

from runtime.events.event import RuntimeEvent
from runtime.events.event_type import RuntimeEventType


def test_runtime_event_creates_with_defaults():

    event = RuntimeEvent(
        type=RuntimeEventType.RUN_STARTED,
    )

    assert event.type is RuntimeEventType.RUN_STARTED
    assert isinstance(event.timestamp, datetime)
    assert event.timestamp.tzinfo is UTC
    assert event.run_id is None
    assert event.agent_name is None
    assert event.iteration is None
    assert event.data == {}


def test_runtime_event_copies_data():

    data = {
        "model": "qwen",
    }

    event = RuntimeEvent(
        type=RuntimeEventType.INFERENCE_STARTED,
        data=data,
    )

    data["model"] = "changed"

    assert event.data == {
        "model": "qwen",
    }


def test_runtime_event_rejects_invalid_type():

    with pytest.raises(
        TypeError,
        match="RuntimeEventType",
    ):
        RuntimeEvent(type="run.started")


def test_runtime_event_rejects_invalid_data():

    with pytest.raises(
        TypeError,
        match="data must be a dictionary",
    ):
        RuntimeEvent(
            type=RuntimeEventType.RUN_STARTED,
            data=[],
        )
