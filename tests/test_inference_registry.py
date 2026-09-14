import pytest

from runtime.config.settings import settings
from runtime.inference.backend import InferenceBackend
from runtime.inference.llama_cpp import LlamaCppBackend
from runtime.inference.registry import (
    BackendRegistry,
    backend_registry,
    create_default_backend_registry,
)


class FakeBackend(InferenceBackend):
    def __init__(self, runtime_settings):
        self.settings = runtime_settings

    def load(self, model):
        pass

    def unload(self):
        pass

    @property
    def loaded(self):
        return False

    @property
    def model(self):
        return None

    def count_tokens(self, text):
        return 0

    def generate(
        self,
        messages,
        *,
        tools=None,
        max_tokens,
        temperature,
        top_p,
    ):
        raise NotImplementedError


def test_registry_registers_and_resolves_backend():
    registry = BackendRegistry()
    registry.register("fake", FakeBackend)

    backend = registry.resolve("fake", settings)

    assert isinstance(backend, FakeBackend)
    assert backend.settings is settings


def test_registry_normalizes_names():
    registry = BackendRegistry()
    registry.register("  FAKE  ", FakeBackend)

    assert registry.contains("fake")
    assert registry.contains(" FAKE ")
    assert registry.names() == ("fake",)


def test_registry_rejects_empty_name():
    registry = BackendRegistry()

    with pytest.raises(ValueError, match="name cannot be empty"):
        registry.register("   ", FakeBackend)


def test_registry_rejects_non_callable_factory():
    registry = BackendRegistry()

    with pytest.raises(TypeError, match="factory must be callable"):
        registry.register("fake", object())


def test_registry_rejects_unknown_backend():
    registry = BackendRegistry()

    with pytest.raises(ValueError, match="Unknown inference backend"):
        registry.resolve("missing", settings)


def test_registry_rejects_invalid_factory_result():
    registry = BackendRegistry()
    registry.register("fake", lambda runtime_settings: object())

    with pytest.raises(TypeError, match="does not implement InferenceBackend"):
        registry.resolve("fake", settings)


def test_default_registry_contains_llama_cpp():
    assert backend_registry.contains("llama_cpp")


def test_default_registry_resolves_llama_cpp():
    registry = create_default_backend_registry()

    backend = registry.resolve("llama_cpp", settings)

    assert isinstance(backend, LlamaCppBackend)
    assert backend.settings is settings


def test_registry_names_are_sorted():
    registry = BackendRegistry()

    registry.register("zeta", FakeBackend)
    registry.register("alpha", FakeBackend)

    assert registry.names() == ("alpha", "zeta")
