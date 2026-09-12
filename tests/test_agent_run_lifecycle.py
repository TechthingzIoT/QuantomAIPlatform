from unittest.mock import MagicMock

import pytest

from runtime.agents.agent import Agent
from runtime.inference.response import InferenceResponse
from runtime.runs.service import RunService
from runtime.runs.status import RunStatus
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
