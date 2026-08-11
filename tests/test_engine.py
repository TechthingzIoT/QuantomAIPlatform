from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from runtime.inference.engine import InferenceEngine


def make_active_model():
    return SimpleNamespace(
        name="test-model.gguf",
        path="/tmp/test-model.gguf",
    )


@pytest.fixture
def engine():
    with patch(
        "runtime.inference.engine.ModelManager"
    ) as manager_class:

        manager = manager_class.return_value
        manager.active_model.return_value = make_active_model()

        instance = InferenceEngine()
        instance.manager = manager

        yield instance


def test_engine_starts_unloaded(engine):
    assert engine.loaded is False
    assert engine.model is None


def test_load_model(engine):
    fake_llama = MagicMock()

    with patch(
        "runtime.inference.engine.Llama",
        return_value=fake_llama,
    ) as llama_class:

        engine.load()

        assert engine.loaded is True
        assert engine.model.name == "test-model.gguf"

        llama_class.assert_called_once_with(
            model_path="/tmp/test-model.gguf",
            n_ctx=engine.settings.context_size,
            n_gpu_layers=engine.settings.gpu_layers,
            verbose=engine.settings.verbose,
        )


def test_load_without_active_model():
    with patch(
        "runtime.inference.engine.ModelManager"
    ) as manager_class:

        manager_class.return_value.active_model.return_value = None

        engine = InferenceEngine()

        with pytest.raises(
            RuntimeError,
            match="No active model selected",
        ):
            engine.load()


def test_generate(engine):
    fake_llama = MagicMock()

    fake_llama.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": " Hello from QAIR! ",
                }
            }
        ]
    }

    engine._model = fake_llama
    engine._model_info = make_active_model()

    messages = [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    response = engine.generate(messages)

    assert response == "Hello from QAIR!"

    fake_llama.create_chat_completion.assert_called_once_with(
        messages=messages,
        max_tokens=engine.settings.max_tokens,
        temperature=engine.settings.temperature,
        top_p=engine.settings.top_p,
    )


def test_generate_auto_loads(engine):
    fake_llama = MagicMock()

    fake_llama.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Loaded automatically.",
                }
            }
        ]
    }

    with patch(
        "runtime.inference.engine.Llama",
        return_value=fake_llama,
    ):
        response = engine.generate(
            [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        )

    assert engine.loaded is True
    assert response == "Loaded automatically."


def test_unload(engine):
    engine._model = MagicMock()
    engine._model_info = make_active_model()

    engine.unload()

    assert engine.loaded is False
    assert engine.model is None


def test_reload(engine):
    engine._model = MagicMock()
    engine._model_info = make_active_model()

    with patch.object(
        engine,
        "unload",
        wraps=engine.unload,
    ) as unload_mock, patch.object(
        engine,
        "load",
        wraps=engine.load,
    ) as load_mock, patch(
        "runtime.inference.engine.Llama",
        return_value=MagicMock(),
    ):

        engine.reload()

        unload_mock.assert_called_once()
        load_mock.assert_called_once()

    assert engine.loaded is True


def test_summary_when_unloaded(engine):
    summary = engine.summary()

    assert summary["loaded"] is False
    assert summary["model"] is None
    assert summary["context"] == engine.settings.context_size
    assert summary["gpu_layers"] == engine.settings.gpu_layers
    assert summary["temperature"] == engine.settings.temperature
    assert summary["top_p"] == engine.settings.top_p
    assert summary["max_tokens"] == engine.settings.max_tokens


def test_summary_when_loaded(engine):
    engine._model = MagicMock()
    engine._model_info = make_active_model()

    summary = engine.summary()

    assert summary["loaded"] is True
    assert summary["model"] == "test-model.gguf"