from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from runtime.agents.agent import Agent
from runtime.inference.response import (
    InferenceResponse,
    ToolCallRequest,
)
from runtime.runs.service import RunService
from runtime.runs.status import RunStatus
from runtime.tools.registry import ToolRegistry
from runtime.tools.telemetry import ToolExecutionTelemetry


def test_agent_creates_and_completes_runtime_run():

    runtime = MagicMock()

    runtime.generate.return_value = InferenceResponse(
        content="Run completed."
    )

    telemetry = ToolExecutionTelemetry()

    run_service = RunService(
        telemetry=telemetry
    )

    agent = Agent(
        runtime=runtime,
        telemetry=telemetry,
        run_service=run_service,
    )

    outcome = agent.run_with_outcome(
        "Hello QAIR"
    )

    run = run_service.require(
        outcome.run_id
    )

    assert run.status is RunStatus.COMPLETED
    assert run.started_at is not None
    assert run.completed_at is not None


def test_agent_fails_runtime_run_when_inference_raises():

    runtime = MagicMock()

    runtime.generate.side_effect = RuntimeError(
        "Inference failed."
    )

    run_service = RunService()

    agent = Agent(
        runtime=runtime,
        run_service=run_service,
    )

    with pytest.raises(
        RuntimeError,
        match="Inference failed.",
    ):
        agent.run_with_outcome(
            "Hello QAIR"
        )

    runs = run_service.registry.runs()

    assert len(runs) == 1

    run = runs[0]

    assert run.status is RunStatus.FAILED
    assert run.error == "Inference failed."
    assert run.started_at is not None
    assert run.completed_at is not None


def test_agent_and_run_service_share_telemetry_when_created_by_agent():

    runtime = MagicMock()

    telemetry = ToolExecutionTelemetry()

    agent = Agent(
        runtime=runtime,
        telemetry=telemetry,
    )

    assert (
        agent.run_service.telemetry
        is telemetry
    )


def test_agent_run_records_agent_name():

    runtime = MagicMock()

    runtime.generate.return_value = InferenceResponse(
        content="Task complete."
    )

    agent = Agent(
        runtime=runtime,
        name="ownership-agent",
    )

    outcome = agent.run_with_outcome("Complete the task.")

    run = agent.run_service.require(outcome.run_id)

    assert run.agent_name == "ownership-agent"


def test_agent_run_emits_started_and_completed_events():
    from runtime.events.event_type import RuntimeEventType

    runtime = MagicMock()
    runtime.generate.return_value = InferenceResponse(
        content="Task complete."
    )

    run_service = RunService()

    agent = Agent(
        runtime=runtime,
        run_service=run_service,
        name="event-agent",
    )

    outcome = agent.run_with_outcome("Hello QAIR")

    events = run_service.event_store.events_for_run(
        outcome.run_id
    )

    event_types = [event.type for event in events]

    assert RuntimeEventType.RUN_STARTED in event_types
    assert RuntimeEventType.AGENT_STARTED in event_types
    assert RuntimeEventType.RUN_COMPLETED in event_types
    assert RuntimeEventType.AGENT_COMPLETED in event_types


def test_agent_run_emits_failed_event():
    from runtime.events.event_type import RuntimeEventType

    runtime = MagicMock()
    runtime.generate.side_effect = RuntimeError(
        "Inference exploded."
    )

    run_service = RunService()

    agent = Agent(
        runtime=runtime,
        run_service=run_service,
        name="failure-agent",
    )

    with pytest.raises(
        RuntimeError,
        match="Inference exploded.",
    ):
        agent.run_with_outcome("Hello QAIR")

    events = run_service.event_store.events()

    event_types = [event.type for event in events]

    assert RuntimeEventType.RUN_STARTED in event_types
    assert RuntimeEventType.AGENT_STARTED in event_types
    assert RuntimeEventType.AGENT_FAILED in event_types
    assert RuntimeEventType.RUN_FAILED in event_types

    failed_event = next(
        event
        for event in events
        if event.type is RuntimeEventType.AGENT_FAILED
    )

    assert failed_event.agent_name == "failure-agent"
    assert failed_event.data["error"] == "Inference exploded."
    assert failed_event.data["error_type"] == "RuntimeError"

class RuntimeLifecycleTool:

    name = "runtime_lifecycle_tool"

    description = (
        "Tool used to verify the QAIR runtime event lifecycle."
    )

    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, arguments):

        return {
            "status": "ok",
        }


def test_agent_tool_run_emits_complete_runtime_lifecycle():

    from runtime.events.event_type import RuntimeEventType

    runtime = MagicMock()

    runtime.generate.side_effect = [

        InferenceResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="lifecycle_call_1",
                    name="runtime_lifecycle_tool",
                    arguments={},
                )
            ],
        ),

        InferenceResponse(
            content="Tool lifecycle completed successfully.",
            tool_calls=[],
        ),

    ]

    registry = ToolRegistry()

    registry.register(RuntimeLifecycleTool())

    run_service = RunService()

    agent = Agent(
        runtime=runtime,
        name="lifecycle-agent",
        tool_registry=registry,
        run_service=run_service,
    )

    outcome = agent.run_with_outcome(
        "Run the lifecycle verification."
    )

    events = run_service.event_store.events_for_run(
        outcome.run_id
    )

    event_types = [
        event.type
        for event in events
    ]

    assert RuntimeEventType.RUN_STARTED in event_types

    assert RuntimeEventType.AGENT_STARTED in event_types

    assert RuntimeEventType.TOOL_STARTED in event_types

    assert RuntimeEventType.TOOL_COMPLETED in event_types

    assert RuntimeEventType.AGENT_COMPLETED in event_types

    assert RuntimeEventType.RUN_COMPLETED in event_types

    tool_started = next(
        event
        for event in events
        if event.type is RuntimeEventType.TOOL_STARTED
    )

    tool_completed = next(
        event
        for event in events
        if event.type is RuntimeEventType.TOOL_COMPLETED
    )

    assert tool_started.run_id == outcome.run_id

    assert tool_completed.run_id == outcome.run_id

    assert tool_started.agent_name == "lifecycle-agent"

    assert tool_completed.agent_name == "lifecycle-agent"

    assert tool_started.iteration == 0

    assert tool_completed.iteration == 0

    assert (
        tool_started.data["tool_name"]
        == "runtime_lifecycle_tool"
    )

    assert (
        tool_completed.data["tool_name"]
        == "runtime_lifecycle_tool"
    )


class RuntimeFailingTool:
    name = "runtime_failing_tool"

    description = (
        "A tool that intentionally fails for lifecycle testing."
    )

    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, arguments):
        raise RuntimeError(
            "Intentional lifecycle tool failure."
        )


def test_agent_run_emits_tool_failed_event_and_recovers():

    from runtime.events.event_type import RuntimeEventType

    runtime = MagicMock()

    runtime.generate.side_effect = [

        InferenceResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_failure_1",
                    name="runtime_failing_tool",
                    arguments={},
                )
            ],
        ),

        InferenceResponse(
            content=(
                "The tool failed, but I handled the failure "
                "and completed the task."
            ),
            tool_calls=[],
        ),

    ]

    registry = ToolRegistry()

    registry.register(RuntimeFailingTool())

    run_service = RunService()

    agent = Agent(
        runtime=runtime,
        run_service=run_service,
        tool_registry=registry,
        name="failure-recovery-agent",
    )

    outcome = agent.run_with_outcome(
        "Run the lifecycle failure recovery test."
    )

    assert outcome.content == (
        "The tool failed, but I handled the failure "
        "and completed the task."
    )

    events = run_service.event_store.events_for_run(
        outcome.run_id
    )

    event_types = [event.type for event in events]

    assert RuntimeEventType.TOOL_STARTED in event_types

    assert RuntimeEventType.TOOL_FAILED in event_types

    assert RuntimeEventType.AGENT_COMPLETED in event_types

    tool_started = next(
        event
        for event in events
        if event.type is RuntimeEventType.TOOL_STARTED
    )

    tool_failed = next(
        event
        for event in events
        if event.type is RuntimeEventType.TOOL_FAILED
    )

    assert tool_started.run_id == outcome.run_id

    assert tool_failed.run_id == outcome.run_id

    assert (
        tool_started.agent_name
        == "failure-recovery-agent"
    )

    assert (
        tool_failed.agent_name
        == "failure-recovery-agent"
    )

    assert tool_started.iteration == 0

    assert tool_failed.iteration == 0

    assert (
        tool_failed.data["tool_name"]
        == "runtime_failing_tool"
    )

    assert (
        tool_failed.data["error_type"]
        == "RuntimeError"
    )

    assert (
        tool_failed.data["error_message"]
        == "Intentional lifecycle tool failure."
    )

    assert runtime.generate.call_count == 2


def test_agent_run_emits_inference_lifecycle_events():

    from runtime.events.event_type import RuntimeEventType

    runtime = MagicMock()

    runtime.generate.return_value = InferenceResponse(
        content="Inference completed successfully."
    )

    run_service = RunService()

    agent = Agent(
        runtime=runtime,
        run_service=run_service,
        name="inference-agent",
    )

    outcome = agent.run_with_outcome(
        "Run inference lifecycle test."
    )

    events = run_service.event_store.events_for_run(
        outcome.run_id
    )

    event_types = [
        event.type
        for event in events
    ]

    assert RuntimeEventType.INFERENCE_STARTED in event_types
    assert RuntimeEventType.INFERENCE_COMPLETED in event_types

    started = next(
        event
        for event in events
        if event.type is RuntimeEventType.INFERENCE_STARTED
    )

    completed = next(
        event
        for event in events
        if event.type is RuntimeEventType.INFERENCE_COMPLETED
    )

    assert started.run_id == outcome.run_id
    assert completed.run_id == outcome.run_id

    assert started.agent_name == "inference-agent"
    assert completed.agent_name == "inference-agent"

    assert started.iteration == 0
    assert completed.iteration == 0


def test_agent_run_emits_inference_failed_event():

    from runtime.events.event_type import RuntimeEventType

    runtime = MagicMock()

    runtime.generate.side_effect = RuntimeError(
        "Inference exploded."
    )

    run_service = RunService()

    agent = Agent(
        runtime=runtime,
        run_service=run_service,
        name="inference-failure-agent",
    )

    with pytest.raises(
        RuntimeError,
        match="Inference exploded.",
    ):
        agent.run_with_outcome(
            "Trigger inference failure."
        )

    events = run_service.event_store.events()

    event_types = [
        event.type
        for event in events
    ]

    assert RuntimeEventType.INFERENCE_STARTED in event_types
    assert RuntimeEventType.INFERENCE_FAILED in event_types

    failed = next(
        event
        for event in events
        if event.type is RuntimeEventType.INFERENCE_FAILED
    )

    assert failed.agent_name == "inference-failure-agent"

    assert failed.data["error"] == "Inference exploded."

    assert failed.data["error_type"] == "RuntimeError"
