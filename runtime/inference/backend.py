"""
=========================================================
QAIR Inference Backend Contract
=========================================================

Defines the interface implemented by QAIR inference
backends.

Backends are responsible for model loading, unloading,
tokenization, and text generation.

Author:
    TIOTAIROBOTIX
=========================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from runtime.models.model import Model


class InferenceBackend(ABC):
    """
    Abstract interface for QAIR inference backends.

    The inference engine depends on this contract rather
    than on a specific inference implementation such as
    llama.cpp, Ollama, OpenAI, or vLLM.
    """

    @abstractmethod
    def load(self, model: Model) -> None:
        """
        Load the supplied model.
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """
        Release the currently loaded model.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def loaded(self) -> bool:
        """
        Whether a model is currently loaded.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> Model | None:
        """
        Return metadata for the currently loaded model.
        """
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using the backend's tokenizer.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """
        Generate a response from conversational messages.
        """
        raise NotImplementedError
