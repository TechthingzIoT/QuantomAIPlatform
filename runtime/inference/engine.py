"""
=========================================================
QAIR Inference Engine
=========================================================

Thin abstraction over the underlying language model.

Responsibilities
----------------
• Load the active model
• Generate responses
• Reload models
• Expose runtime information

Author:
    TIOTAIROBOTIX
=========================================================
"""

from __future__ import annotations

from llama_cpp import Llama

from runtime.config.settings import settings
from runtime.models.manager import ModelManager
from runtime.models.model import Model


class InferenceEngine:
    """
    QAIR inference engine.

    This class owns the loaded language model and provides
    a simple interface for text generation.
    """

    def __init__(
        self,
        *,
        model_manager: ModelManager | None = None,
    ) -> None:
        self.settings = settings

        # Preserve an explicitly injected model manager.
        # Only create the default manager when none was supplied.
        self.manager = (
            model_manager
            if model_manager is not None
            else ModelManager()
        )

        self._model: Llama | None = None
        self._model_info: Model | None = None

    # ==================================================
    # Loading
    # ==================================================

    def load(self) -> None:
        """
        Load the currently active model.
        """

        active = self.manager.active_model()

        if active is None:
            raise RuntimeError("No active model selected.")

        self._model_info = active

        self._model = Llama(
            model_path=str(active.path),
            n_ctx=self.settings.context_size,
            n_gpu_layers=self.settings.gpu_layers,
            verbose=self.settings.verbose,
        )

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

        self._model = None
        self._model_info = None

    # ==================================================
    # Status
    # ==================================================

    @property
    def loaded(self) -> bool:
        """
        Whether a model is currently loaded.
        """

        return self._model is not None

    @property
    def model(self) -> Model | None:
        """
        Return loaded model metadata.
        """

        return self._model_info

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
        Generate a response using the model's native
        chat completion API.
        """

        if not self.loaded:
            self.load()

        assert self._model is not None

        actual_max_tokens = (
            self.settings.max_tokens
            if max_tokens is None
            else max_tokens
        )

        actual_temperature = (
            self.settings.temperature
            if temperature is None
            else temperature
        )

        actual_top_p = (
            self.settings.top_p
            if top_p is None
            else top_p
        )

        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=actual_max_tokens,
            temperature=actual_temperature,
            top_p=actual_top_p,
        )

        return response["choices"][0]["message"]["content"].strip()

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