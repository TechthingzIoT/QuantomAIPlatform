"""
=========================================================
QAIR llama.cpp Inference Backend
=========================================================

Concrete inference backend for local GGUF models using
llama.cpp through llama-cpp-python.

Author:
    TIOTAIROBOTIX
=========================================================
"""

from __future__ import annotations

import json
import re

from llama_cpp import Llama

from runtime.config.settings import QAIRSettings, settings
from runtime.inference.backend import InferenceBackend
from runtime.inference.response import (
    InferenceResponse,
    ToolCallRequest,
)
from runtime.models.model import Model


class LlamaCppBackend(InferenceBackend):
    """
    QAIR inference backend powered by llama.cpp.

    This class owns the llama.cpp model instance and
    translates the QAIR inference contract into the
    llama-cpp-python API.
    """

    def __init__(self, runtime_settings: QAIRSettings | None = None) -> None:
        self.settings = runtime_settings or settings
        self._model: Llama | None = None
        self._model_info: Model | None = None

    def load(self, model: Model) -> None:
        """
        Load a GGUF model using llama.cpp.
        """
        self._model = Llama(
            model_path=str(model.path),
            n_ctx=self.settings.context_size,
            n_gpu_layers=self.settings.gpu_layers,
            verbose=self.settings.verbose,
        )
        self._model_info = model

    def unload(self) -> None:
        """
        Release the loaded llama.cpp model.
        """
        self._model = None
        self._model_info = None

    @property
    def loaded(self) -> bool:
        """
        Whether a model is currently loaded.
        """
        return self._model is not None

    @property
    def model(self) -> Model | None:
        """
        Return metadata for the currently loaded model.
        """
        return self._model_info

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using the loaded llama.cpp tokenizer.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not self.loaded:
            raise RuntimeError("No model is loaded.")

        assert self._model is not None

        return len(
            self._model.tokenize(
                text.encode("utf-8"),
                add_bos=False,
            )
        )

    def _parse_text_tool_calls(
        self,
        content: str | None,
    ) -> tuple[str | None, list[ToolCallRequest]]:
        """
        Parse Qwen-style textual tool calls.

        Some GGUF/chat-template combinations return tool calls as:

            <tool_call>
            {"name": "tool_name", "arguments": {...}}
            </tool_call>

        instead of OpenAI-compatible structured tool_calls.
        """

        if not content:
            return content, []

        pattern = re.compile(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            re.DOTALL,
        )

        matches = pattern.findall(content)

        if not matches:
            return content, []

        tool_calls: list[ToolCallRequest] = []

        for index, raw_call in enumerate(matches, start=1):
            try:
                payload = json.loads(raw_call)
            except json.JSONDecodeError:
                continue

            name = payload.get("name")

            arguments = payload.get(
                "arguments",
                {},
            )

            if not isinstance(name, str):
                continue

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue

            if not isinstance(arguments, dict):
                continue

            tool_calls.append(
                ToolCallRequest(
                    id=f"qair_text_call_{index}",
                    name=name,
                    arguments=arguments,
                )
            )

        if not tool_calls:
            return content, []

        cleaned_content = pattern.sub("", content).strip()

        return (
            cleaned_content or None,
            tool_calls,
        )

    def _normalize_messages(
        self,
        messages: list[dict],
    ) -> list[dict]:
        """
        Convert QAIR conversation messages into the format expected
        by llama.cpp's OpenAI-compatible chat completion API.

        QAIR stores tool calls in a provider-neutral representation:

            {
                "id": "...",
                "name": "...",
                "arguments": {...},
            }

        llama.cpp expects OpenAI-compatible function call objects.
        """

        normalized: list[dict] = []

        for message in messages:
            item = dict(message)

            tool_calls = item.get("tool_calls")

            if tool_calls:
                normalized_calls = []

                for tool_call in tool_calls:
                    arguments = tool_call.get("arguments", {})

                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)

                    normalized_calls.append(
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": arguments,
                            },
                        }
                    )

                item["tool_calls"] = normalized_calls

            normalized.append(item)

        return normalized

    def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> InferenceResponse:
        """
        Generate a response using llama.cpp chat completion.
        """
        if not self.loaded:
            raise RuntimeError("No model is loaded.")

        assert self._model is not None

        request = {
            "messages": self._normalize_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if tools:
            request["tools"] = tools

        response = self._model.create_chat_completion(**request)

        message = response["choices"][0]["message"]

        content = message.get("content")

        raw_tool_calls = message.get("tool_calls") or []

        tool_calls: list[ToolCallRequest] = []

        for raw_tool_call in raw_tool_calls:

            function = raw_tool_call["function"]

            arguments = function.get("arguments", {})

            if isinstance(arguments, str):

                arguments = json.loads(arguments)

            tool_calls.append(

                ToolCallRequest(

                    id=raw_tool_call["id"],

                    name=function["name"],

                    arguments=arguments,

                )

            )

        if not tool_calls:

            content, tool_calls = self._parse_text_tool_calls(
                content
            )

        return InferenceResponse(
            content=content.strip() if content else None,
            tool_calls=tool_calls,
        )
