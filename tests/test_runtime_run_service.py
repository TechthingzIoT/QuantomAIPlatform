from runtime.agents.agent import Agent
from runtime.core.runtime import QAIRRuntime
from runtime.runs.service import RunService
from runtime.tools.telemetry import ToolExecutionTelemetry


def test_runtime_creates_run_service_by_default():
    runtime = QAIRRuntime()

    assert isinstance(runtime.run_service, RunService)


def test_runtime_run_service_has_telemetry():
    runtime = QAIRRuntime()

    assert isinstance(
        runtime.run_service.telemetry,
        ToolExecutionTelemetry,
    )


def test_runtime_accepts_custom_run_service():
    run_service = RunService()

    runtime = QAIRRuntime(
        run_service=run_service,
    )

    assert runtime.run_service is run_service


def test_agent_uses_runtime_run_service_by_default():
    runtime = QAIRRuntime()

    agent = Agent(
        runtime=runtime,
    )

    assert agent.run_service is runtime.run_service


def test_agent_uses_runtime_telemetry_by_default():
    runtime = QAIRRuntime()

    agent = Agent(
        runtime=runtime,
    )

    assert (
        agent.telemetry
        is runtime.run_service.telemetry
    )


def test_agent_can_override_runtime_run_service():
    runtime = QAIRRuntime()

    custom_service = RunService()

    agent = Agent(
        runtime=runtime,
        run_service=custom_service,
    )

    assert agent.run_service is custom_service


def test_agent_can_use_custom_telemetry_with_custom_service():
    runtime = QAIRRuntime()

    telemetry = ToolExecutionTelemetry()

    service = RunService(
        telemetry=telemetry,
    )

    agent = Agent(
        runtime=runtime,
        run_service=service,
        telemetry=telemetry,
    )

    assert agent.run_service is service
    assert agent.telemetry is telemetry
    assert agent.run_service.telemetry is telemetry
