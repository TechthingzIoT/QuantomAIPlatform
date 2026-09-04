"""
=========================================================
QAIR Inference Engine
=========================================================

High-level inference orchestration for QAIR.

Responsibilities
----------------

• Select the active model
• Coordinate the inference backend
• Generate responses
• Reload models
• Expose runtime information

The engine intentionally does not depend on a specific
inference implementation. Backend-specific behavior is
provided through the InferenceBackend contract.

Author:
    TIOTAIROBOTIX
=========================================================
"""

from __future__ import annotations

from runtime.config.settings import settings
from runtime.inference.backend import InferenceBackend
from runtime.inference.llama_cpp import LlamaCppBackend
from runtime.models.manager import ModelManager
from runtime.models.model import Model


class InferenceEngine:
    """
    QAIR inference engine.

    The engine coordinates model selection and delegates
    inference operations to an InferenceBackend.

    By default, QAIR uses LlamaCppBackend so existing
    local GGUF inference behavior remains unchanged.
    """

    def __init__(
        self,
        *,
        model_manager: ModelManager | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        self.settings = settings

        # Preserve an explicitly injected model manager.
        self.manager = (
            model_manager if model_manager is not None else ModelManager()
        )

        # Preserve an explicitly injected backend.
        self.backend = (
            backend if backend is not None else LlamaCppBackend(self.settings)
        )

        self._model_info: Model | None = None

    # ==================================================
    # Loading
    # ==================================================

    def load(self) -> None:
        """
        Load the currently active model through the backend.
        """
        active = self.manager.active_model()

        if active is None:
            raise RuntimeError("No active model selected.")

        self.backend.load(active)
        self._model_info = active

    def reload(self) -> None:
        """
        Reload the active model.
        """
        self.unload()
        self.load()

    def unload(self) -> None:
        """
        Release the loaded model.
        """
        self.backend.unload()
        self._model_info = None

    # ==================================================
    # Status
    # ==================================================

    @property
    def loaded(self) -> bool:
        """
        Whether a model is currently loaded.
        """
        return self.backend.loaded

    @property
    def model(self) -> Model | None:
        """
        Return loaded model metadata.
        """
        return self._model_info

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using the active backend.

        The model is loaded lazily when necessary.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not self.loaded:
            self.load()

        return self.backend.count_tokens(text)

    # ==================================================
    # Generation
    # ==================================================

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        Generate a response through the active backend.
        """
        if not self.loaded:
            self.load()

        actual_max_tokens = (
            self.settings.max_tokens if max_tokens is None else max_tokens
        )

        actual_temperature = (
            self.settings.temperature
            if temperature is None
            else temperature
        )

        actual_top_p = self.settings.top_p if top_p is None else top_p

        return self.backend.generate(
            messages,
            max_tokens=actual_max_tokens,
            temperature=actual_temperature,
            top_p=actual_top_p,
        )

    # ==================================================
    # Runtime Information
    # ==================================================

    def summary(self) -> dict:
        """
        Runtime summary.
        """
        return {
            "loaded": self.loaded,
            "model": self.model.name if self.model else None,
            "context": self.settings.context_size,
            "gpu_layers": self.settings.gpu_layers,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
        }
