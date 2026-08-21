"""
QAIR API Schemas

Request and response models for the QAIR HTTP API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """QAIR health status."""

    status: str
    running: bool
    loaded: bool


class ModelResponse(BaseModel):
    """Information about an available model."""

    name: str
    path: str | None = None


class ModelsResponse(BaseModel):
    """Collection of available models."""

    object: str = "list"
    data: list[dict[str, Any]]


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    choices: list[ChatCompletionChoice]