from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime.inference.engine import InferenceEngine
from runtime.models.model import Model


def make_model() -> Model:
    return Model(
        name="test-model.gguf",
        path=Path("/tmp/test-model.gguf"),
        size=1024 * 1024,
        extension=".gguf",
    )


@pytest.fixture
def model():
    return make_model()


@pytest.fixture
def model_manager(model):
    manager = MagicMock()
    manager.active_model.return_value = model
    return manager


@pytest.fixture
def backend():
    backend = MagicMock()
    backend.loaded = False
    backend.model = None
    return backend


@pytest.fixture
def engine(model_manager, backend):
    return InferenceEngine(
        model_manager=model_manager,
        backend=backend,
    )


def test_engine_accepts_custom_model_manager():
    custom_manager = MagicMock()

    engine = InferenceEngine(model_manager=custom_manager)

    assert engine.manager is custom_manager


def test_engine_accepts_custom_backend(model_manager, backend):
    engine = InferenceEngine(
        model_manager=model_manager,
        backend=backend,
    )

    assert engine.backend is backend


def test_engine_uses_default_llama_cpp_backend(model_manager):
    with patch("runtime.inference.engine.LlamaCppBackend") as backend_class:
        engine = InferenceEngine(model_manager=model_manager)

    backend_class.assert_called_once_with(engine.settings)
    assert engine.backend is backend_class.return_value


def test_engine_loads_active_model(engine, backend, model):
    engine.load()

    backend.load.assert_called_once_with(model)
    assert engine.model is model


def test_engine_load_fails_without_active_model(backend):
    manager = MagicMock()
    manager.active_model.return_value = None

    engine = InferenceEngine(
        model_manager=manager,
        backend=backend,
    )

    with pytest.raises(RuntimeError, match="No active model selected."):
        engine.load()

    backend.load.assert_not_called()


def test_engine_loaded_delegates_to_backend(engine, backend):
    backend.loaded = True

    assert engine.loaded is True

    backend.loaded = False

    assert engine.loaded is False


def test_engine_unload_delegates_to_backend(engine, backend, model):
    engine.load()

    engine.unload()

    backend.unload.assert_called_once_with()
    assert engine.loaded is False
    assert engine.model is None


def test_engine_reload_unloads_and_loads_again(
    engine,
    backend,
    model,
):
    engine.load()
    engine.reload()

    assert backend.unload.call_count == 1
    assert backend.load.call_count == 2
    backend.load.assert_called_with(model)
    assert engine.model is model


def test_engine_model_returns_loaded_model(engine, model):
    assert engine.model is None

    engine.load()

    assert engine.model is model


def test_engine_count_tokens_delegates_to_backend(
    engine,
    backend,
):
    backend.loaded = True
    backend.count_tokens.return_value = 4

    result = engine.count_tokens("hello world")

    assert result == 4
    backend.count_tokens.assert_called_once_with("hello world")


def test_engine_count_tokens_loads_model_lazily(
    engine,
    backend,
    model,
):
    backend.loaded = False
    backend.count_tokens.return_value = 2

    result = engine.count_tokens("hello")

    assert result == 2
    backend.load.assert_called_once_with(model)
    backend.count_tokens.assert_called_once_with("hello")


def test_engine_count_tokens_rejects_non_string(engine):
    with pytest.raises(TypeError, match="text must be a string"):
        engine.count_tokens(123)


def test_engine_generate_uses_settings_defaults(
    engine,
    backend,
):
    backend.loaded = True
    backend.generate.return_value = "Backend response"

    messages = [{"role": "user", "content": "Hello"}]

    result = engine.generate(messages)

    assert result == "Backend response"

    backend.generate.assert_called_once_with(
        messages,
        max_tokens=engine.settings.max_tokens,
        temperature=engine.settings.temperature,
        top_p=engine.settings.top_p,
    )


def test_engine_generate_accepts_overrides(
    engine,
    backend,
):
    backend.loaded = True
    backend.generate.return_value = "Custom response"

    messages = [{"role": "user", "content": "Hello"}]

    result = engine.generate(
        messages,
        max_tokens=128,
        temperature=0.2,
        top_p=0.8,
    )

    assert result == "Custom response"

    backend.generate.assert_called_once_with(
        messages,
        max_tokens=128,
        temperature=0.2,
        top_p=0.8,
    )


def test_engine_generate_loads_model_lazily(
    engine,
    backend,
    model,
):
    backend.loaded = False
    backend.generate.return_value = "Lazy response"

    messages = [{"role": "user", "content": "Hello"}]

    result = engine.generate(messages)

    assert result == "Lazy response"
    backend.load.assert_called_once_with(model)
    backend.generate.assert_called_once_with(
        messages,
        max_tokens=engine.settings.max_tokens,
        temperature=engine.settings.temperature,
        top_p=engine.settings.top_p,
    )


def test_engine_summary(engine, backend, model):
    backend.loaded = True

    engine.load()

    summary = engine.summary()

    assert summary == {
        "loaded": True,
        "model": model.name,
        "context": engine.settings.context_size,
        "gpu_layers": engine.settings.gpu_layers,
        "temperature": engine.settings.temperature,
        "top_p": engine.settings.top_p,
        "max_tokens": engine.settings.max_tokens,
    }
