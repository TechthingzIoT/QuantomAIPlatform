"""
QAIR API Tests
"""

from unittest.mock import MagicMock

import pytest

from fastapi import HTTPException

from runtime.api import routes
from runtime.api.schemas import ChatCompletionRequest

from fastapi.testclient import TestClient

from runtime.api.app import app

@pytest.fixture
def mock_runtime(monkeypatch):
    """Provide an isolated mocked QAIR runtime."""

    runtime = MagicMock()

    runtime.running = True
    runtime.loaded = True

    monkeypatch.setattr(
        routes,
        "runtime",
        runtime,
    )

    return runtime


# ============================================================
# Health
# ============================================================


def test_health(mock_runtime):
    response = routes.health()

    assert response.status == "ok"
    assert response.running is True
    assert response.loaded is True


# ============================================================
# Models
# ============================================================


def test_list_models(mock_runtime):
    model = MagicMock()
    model.name = "test-model.gguf"

    mock_runtime.list_models.return_value = [
        model
    ]

    response = routes.list_models()

    assert len(response.data) == 1
    assert response.data[0]["id"] == "test-model.gguf"
    assert response.data[0]["object"] == "model"
    assert response.data[0]["owned_by"] == "qair"


# ============================================================
# Chat Completions
# ============================================================


def test_chat_completions_delegates_to_runtime(
    mock_runtime,
):
    mock_runtime.generate.return_value = (
        "Hello from QAIR."
    )

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

    assert (
        response.choices[0].message.role
        == "assistant"
    )

    assert (
        response.choices[0].message.content
        == "Hello from QAIR."
    )


def test_chat_completions_defaults_knowledge_to_disabled(
    mock_runtime,
):
    mock_runtime.generate.return_value = (
        "Normal response."
    )

    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
    )

    assert request.use_knowledge is False
    assert request.knowledge_limit == 5

    routes.chat_completions(request)

    # Legacy requests must remain compatible.
    mock_runtime.generate.assert_called_once_with(
        [
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        max_tokens=None,
        temperature=None,
        top_p=None,
    )


def test_chat_completions_enables_knowledge(
    mock_runtime,
):
    mock_runtime.generate.return_value = (
        "RAG response."
    )

    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": "What is QAIR?",
            }
        ],
        use_knowledge=True,
        knowledge_limit=3,
    )

    response = routes.chat_completions(request)

    mock_runtime.generate.assert_called_once_with(
        [
            {
                "role": "user",
                "content": "What is QAIR?",
            }
        ],
        max_tokens=None,
        temperature=None,
        top_p=None,
        use_knowledge=True,
        knowledge_limit=3,
    )

    assert (
        response.choices[0].message.content
        == "RAG response."
    )


def test_chat_completions_forwards_rag_options_with_generation_options(
    mock_runtime,
):
    mock_runtime.generate.return_value = (
        "RAG response."
    )

    request = ChatCompletionRequest(
        model="test-model.gguf",
        messages=[
            {
                "role": "user",
                "content": "Explain edge AI.",
            }
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=128,
        use_knowledge=True,
        knowledge_limit=7,
    )

    routes.chat_completions(request)

    mock_runtime.generate.assert_called_once_with(
        [
            {
                "role": "user",
                "content": "Explain edge AI.",
            }
        ],
        max_tokens=128,
        temperature=0.2,
        top_p=0.9,
        use_knowledge=True,
        knowledge_limit=7,
    )


# ============================================================
# API Validation
# ============================================================


def test_chat_completion_request_rejects_zero_knowledge_limit():
    with pytest.raises(ValueError):
        ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            knowledge_limit=0,
        )


def test_chat_completion_request_rejects_negative_knowledge_limit():
    with pytest.raises(ValueError):
        ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            knowledge_limit=-1,
        )


def test_chat_completion_request_accepts_positive_knowledge_limit():
    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        knowledge_limit=10,
    )

    assert request.knowledge_limit == 10


# ============================================================
# Error Handling
# ============================================================


def test_chat_completions_returns_503_for_runtime_error(
    mock_runtime,
):
    mock_runtime.generate.side_effect = RuntimeError(
        "No active model selected."
    )

    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.chat_completions(request)

    assert exc_info.value.status_code == 503

    assert (
        "No active model selected."
        in exc_info.value.detail
    )


def test_chat_completions_returns_500_for_inference_error(
    mock_runtime,
):
    mock_runtime.generate.side_effect = ValueError(
        "Inference failure"
    )

    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.chat_completions(request)

    assert exc_info.value.status_code == 500

    assert (
        "Inference failed"
        in exc_info.value.detail
    )

    # ============================================================
# HTTP Integration Tests
# ============================================================


@pytest.fixture
def api_client(mock_runtime):
    """Create a FastAPI test client with the mocked runtime."""
    return TestClient(app)


def test_http_health(api_client, mock_runtime):
    response = api_client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["running"] is True
    assert data["loaded"] is True


def test_http_list_models(api_client, mock_runtime):
    model = MagicMock()
    model.name = "test-model.gguf"

    mock_runtime.list_models.return_value = [model]

    response = api_client.get("/v1/models")

    assert response.status_code == 200

    data = response.json()

    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model.gguf"
    assert data["data"][0]["object"] == "model"
    assert data["data"][0]["owned_by"] == "qair"


def test_http_chat_completions(api_client, mock_runtime):
    mock_runtime.generate.return_value = "Hello from QAIR."

    response = api_client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            "temperature": 0.3,
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert len(data["choices"]) == 1

    assert data["choices"][0]["index"] == 0
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == (
        "Hello from QAIR."
    )

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


def test_http_chat_completions_with_knowledge(
    api_client,
    mock_runtime,
):
    mock_runtime.generate.return_value = "RAG response."

    response = api_client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "What is QAIR?",
                }
            ],
            "use_knowledge": True,
            "knowledge_limit": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["choices"][0]["message"]["content"] == (
        "RAG response."
    )

    mock_runtime.generate.assert_called_once_with(
        [
            {
                "role": "user",
                "content": "What is QAIR?",
            }
        ],
        max_tokens=None,
        temperature=None,
        top_p=None,
        use_knowledge=True,
        knowledge_limit=3,
    )


def test_http_chat_completions_runtime_error(
    api_client,
    mock_runtime,
):
    mock_runtime.generate.side_effect = RuntimeError(
        "No active model selected."
    )

    response = api_client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        },
    )

    assert response.status_code == 503

    data = response.json()

    assert "QAIR runtime unavailable" in data["detail"]
    assert "No active model selected." in data["detail"]


def test_http_chat_completions_inference_error(
    api_client,
    mock_runtime,
):
    mock_runtime.generate.side_effect = ValueError(
        "Inference failure"
    )

    response = api_client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        },
    )

    assert response.status_code == 500

    data = response.json()

    assert "Inference failed" in data["detail"]
    assert "Inference failure" in data["detail"]


def test_http_chat_completions_rejects_invalid_knowledge_limit(
    api_client,
    mock_runtime,
):
    response = api_client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            "use_knowledge": True,
            "knowledge_limit": 0,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert "knowledge_limit" in str(data["detail"])

    mock_runtime.generate.assert_not_called()