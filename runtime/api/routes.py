"""
QAIR API Routes
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from runtime.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    HealthResponse,
    ModelsResponse,
)
from runtime.core.runtime import QAIRRuntime


router = APIRouter()


# A single runtime instance is shared by the API process.
runtime = QAIRRuntime()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Return QAIR runtime health."""

    return HealthResponse(
        status="ok",
        running=runtime.running,
        loaded=runtime.loaded,
    )


@router.get(
    "/v1/models",
    response_model=ModelsResponse,
)
def list_models() -> ModelsResponse:
    """Return models discovered by QAIR."""

    models = runtime.list_models()

    data = []

    for model in models:
        data.append(
            {
                "id": model.name,
                "object": "model",
                "owned_by": "qair",
            }
        )

    return ModelsResponse(data=data)


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """
    Generate a chat completion through QAIR.

    RAG can be explicitly enabled through:
        use_knowledge=true

    Legacy request objects without the RAG fields remain
    supported for backward compatibility.
    """

    messages = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.messages
    ]

    # --------------------------------------------------
    # Backward-compatible RAG options
    # --------------------------------------------------

    use_knowledge = getattr(
        request,
        "use_knowledge",
        False,
    )

    knowledge_limit = getattr(
        request,
        "knowledge_limit",
        5,
    )

    # --------------------------------------------------
    # Build runtime arguments.
    #
    # Do not pass RAG arguments for normal requests.
    # This preserves the existing QAIR API contract.
    # --------------------------------------------------

    runtime_kwargs = {
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
    }

    if use_knowledge:
        runtime_kwargs["use_knowledge"] = True
        runtime_kwargs["knowledge_limit"] = knowledge_limit

    try:
        reply = runtime.generate(
            messages,
            **runtime_kwargs,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"QAIR runtime unavailable: {exc}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=reply,
                ),
            )
        ],
    )