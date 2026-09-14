"""
Provider-neutral inference orchestration for QAIR.

The engine coordinates model selection, backend resolution, loading,
token counting, and generation without depending on a specific
inference provider.
"""

from __future__ import annotations

from runtime.config.settings import settings
from runtime.inference.backend import InferenceBackend
from runtime.inference.registry import BackendRegistry, backend_registry
from runtime.inference.response import InferenceResponse
from runtime.models.manager import ModelManager
from runtime.models.model import Model


class InferenceEngine:
    """Provider-neutral inference orchestration layer."""

    def __init__(
        self,
        *,
        model_manager: ModelManager | None = None,
        backend: InferenceBackend | None = None,
        registry: BackendRegistry | None = None,
    ) -> None:
        self.settings = settings

        self.manager = (
            model_manager
            if model_manager is not None
            else ModelManager()
        )

        self.registry = (
            registry
            if registry is not None
            else backend_registry
        )

        # An explicitly supplied backend is primarily useful for tests
        # and advanced embedding of QAIR.
        self.backend = (
            backend
            if backend is not None
            else self.registry.resolve(
                self.settings.inference_backend,
                self.settings,
            )
        )

        self._model_info: Model | None = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the currently active model through the configured backend."""
        active = self.manager.active_model()

        if active is None:
            raise RuntimeError("No active model selected.")

        self.backend.load(active)
        self._model_info = active

    def reload(self) -> None:
        """Unload and reload the currently active model."""
        self.unload()
        self.load()

    def unload(self) -> None:
        """Unload the active model from the inference backend."""
        self.backend.unload()
        self._model_info = None

    # ------------------------------------------------------------------
    # Runtime state
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        """Return whether the inference backend currently has a model loaded."""
        return self.backend.loaded

    @property
    def model(self) -> Model | None:
        """Return the model currently tracked by the engine."""
        return self._model_info

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Return the backend token count for ``text``."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not self.loaded:
            self.load()

        return self.backend.count_tokens(text)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> InferenceResponse:
        """Generate an inference response through the configured backend."""
        if not self.loaded:
            self.load()

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

        return self.backend.generate(
            messages,
            tools=tools,
            max_tokens=actual_max_tokens,
            temperature=actual_temperature,
            top_p=actual_top_p,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a concise snapshot of the inference configuration/state."""
        return {
            "loaded": self.loaded,
            "model": self.model.name if self.model else None,
            "backend": self.settings.inference_backend,
            "context": self.settings.context_size,
            "gpu_layers": self.settings.gpu_layers,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
        }
