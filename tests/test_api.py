from unittest.mock import MagicMock

import pytest

from runtime.api import routes


@pytest.fixture
def mock_runtime(monkeypatch):
    runtime = MagicMock()
    runtime.running = True
    runtime.loaded = True

    monkeypatch.setattr(
        routes,
        "runtime",
        runtime,
    )

    return runtime


def test_health(mock_runtime):
    response = routes.health()

    assert response.status == "ok"
    assert response.running is True
    assert response.loaded is True


def test_list_models(mock_runtime):
    model = MagicMock()
    model.name = "test-model.gguf"

    mock_runtime.list_models.return_value = [model]

    response = routes.list_models()

    assert len(response.data) == 1
    assert response.data[0]["id"] == "test-model.gguf"
    assert response.data[0]["object"] == "model"
    assert response.data[0]["owned_by"] == "qair"


def test_chat_completions_delegates_to_runtime(mock_runtime):
    mock_runtime.generate.return_value = "Hello from QAIR."

    from runtime.api.schemas import ChatCompletionRequest

    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        temperature=0.3,
        max_tokens=64,
    )

    response = routes.chat_completions(request)

    mock_runtime.generate.assert_called_once_with(
        [
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        max_tokens=64,
        temperature=0.3,
        top_p=None,
    )

    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Hello from QAIR."


def test_chat_completions_returns_503_for_runtime_error(
    mock_runtime,
):
    mock_runtime.generate.side_effect = RuntimeError(
        "No active model selected."
    )

    from runtime.api.schemas import ChatCompletionRequest

    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        routes.chat_completions(request)

    assert exc_info.value.status_code == 503
    assert "No active model selected." in exc_info.value.detail


def test_chat_completions_returns_500_for_inference_error(
    mock_runtime,
):
    mock_runtime.generate.side_effect = ValueError(
        "Inference failure"
    )

    from runtime.api.schemas import ChatCompletionRequest

    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        routes.chat_completions(request)

    assert exc_info.value.status_code == 500
    assert "Inference failed" in exc_info.value.detail