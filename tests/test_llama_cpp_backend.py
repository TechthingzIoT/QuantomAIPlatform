from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime.inference.llama_cpp import LlamaCppBackend
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
def backend():
    return LlamaCppBackend()


def test_backend_loads_model(backend, model):
    with patch("runtime.inference.llama_cpp.Llama") as llama_class:
        backend.load(model)

    llama_class.assert_called_once_with(
        model_path=str(model.path),
        n_ctx=backend.settings.context_size,
        n_gpu_layers=backend.settings.gpu_layers,
        verbose=backend.settings.verbose,
    )

    assert backend.loaded is True
    assert backend.model is model


def test_backend_unload_clears_model(backend, model):
    with patch("runtime.inference.llama_cpp.Llama"):
        backend.load(model)

    assert backend.loaded is True

    backend.unload()

    assert backend.loaded is False
    assert backend.model is None


def test_backend_count_tokens(backend, model):
    llama = MagicMock()
    llama.tokenize.return_value = [1, 2, 3, 4]

    with patch("runtime.inference.llama_cpp.Llama", return_value=llama):
        backend.load(model)
        result = backend.count_tokens("hello world")

    assert result == 4

    llama.tokenize.assert_called_once_with(
        b"hello world",
        add_bos=False,
    )


def test_backend_count_tokens_rejects_non_string(backend, model):
    with patch("runtime.inference.llama_cpp.Llama"):
        backend.load(model)

    with pytest.raises(TypeError, match="text must be a string"):
        backend.count_tokens(123)


def test_backend_count_tokens_requires_loaded_model(backend):
    with pytest.raises(RuntimeError, match="No model is loaded"):
        backend.count_tokens("hello")


def test_backend_generate(backend, model):
    llama = MagicMock()
    llama.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": "  Hello from QAIR!  ",
                }
            }
        ]
    }

    messages = [{"role": "user", "content": "Hello"}]

    with patch("runtime.inference.llama_cpp.Llama", return_value=llama):
        backend.load(model)

        result = backend.generate(
            messages,
            max_tokens=128,
            temperature=0.2,
            top_p=0.8,
        )

    assert result == "Hello from QAIR!"

    llama.create_chat_completion.assert_called_once_with(
        messages=messages,
        max_tokens=128,
        temperature=0.2,
        top_p=0.8,
    )


def test_backend_generate_requires_loaded_model(backend):
    with pytest.raises(RuntimeError, match="No model is loaded"):
        backend.generate(
            [{"role": "user", "content": "Hello"}],
            max_tokens=128,
            temperature=0.2,
            top_p=0.8,
        )


def test_backend_implements_inference_backend_contract(backend):
    from runtime.inference.backend import InferenceBackend

    assert isinstance(backend, InferenceBackend)
