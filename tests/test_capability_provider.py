from unittest.mock import MagicMock

import pytest

from runtime.capabilities import CapabilityProvider
from runtime.core.runtime import QAIRRuntime


def make_runtime() -> QAIRRuntime:
    runtime = MagicMock(spec=QAIRRuntime)
    runtime.running = False
    runtime.loaded = False
    runtime.knowledge_retriever = None
    return runtime


def test_provider_rejects_invalid_runtime():
    with pytest.raises(
        TypeError,
        match="runtime must be a QAIRRuntime",
    ):
        CapabilityProvider(object())


def test_provider_lists_core_capabilities():
    runtime = make_runtime()
    runtime.running = True
    runtime.loaded = True

    provider = CapabilityProvider(runtime)

    capabilities = provider.list()

    assert [capability.name for capability in capabilities] == [
        "inference.generate",
        "knowledge.retrieve",
    ]


def test_inference_capability_available_when_runtime_is_ready():
    runtime = make_runtime()
    runtime.running = True
    runtime.loaded = True

    provider = CapabilityProvider(runtime)

    capability = provider.get("inference.generate")

    assert capability is not None
    assert capability.available is True
    assert provider.can("inference.generate") is True


def test_inference_capability_unavailable_when_runtime_not_running():
    runtime = make_runtime()
    runtime.running = False
    runtime.loaded = False

    provider = CapabilityProvider(runtime)

    assert provider.can("inference.generate") is False


def test_inference_capability_unavailable_when_engine_not_loaded():
    runtime = make_runtime()
    runtime.running = True
    runtime.loaded = False

    provider = CapabilityProvider(runtime)

    assert provider.can("inference.generate") is False


def test_knowledge_capability_available_when_runtime_is_running():
    runtime = make_runtime()
    runtime.running = True
    runtime.knowledge_retriever = MagicMock()

    provider = CapabilityProvider(runtime)

    capability = provider.get("knowledge.retrieve")

    assert capability is not None
    assert capability.available is True
    assert provider.can("knowledge.retrieve") is True


def test_knowledge_capability_unavailable_when_runtime_is_stopped():
    runtime = make_runtime()
    runtime.running = False
    runtime.knowledge_retriever = MagicMock()

    provider = CapabilityProvider(runtime)

    assert provider.can("knowledge.retrieve") is False


def test_knowledge_capability_unavailable_without_retriever():
    runtime = make_runtime()
    runtime.running = True
    runtime.knowledge_retriever = None

    provider = CapabilityProvider(runtime)

    assert provider.can("knowledge.retrieve") is False


def test_unknown_capability_returns_none():
    runtime = make_runtime()
    provider = CapabilityProvider(runtime)

    assert provider.get("unknown.capability") is None
    assert provider.can("unknown.capability") is False


def test_get_normalizes_capability_name():
    runtime = make_runtime()
    runtime.running = True
    runtime.loaded = True

    provider = CapabilityProvider(runtime)

    capability = provider.get("  inference.generate  ")

    assert capability is not None
    assert capability.name == "inference.generate"


def test_get_rejects_non_string_name():
    runtime = make_runtime()
    provider = CapabilityProvider(runtime)

    assert provider.get(None) is None
