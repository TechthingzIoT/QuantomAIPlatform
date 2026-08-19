"""
QAIR Core Runtime

Central lifecycle controller for the Quantom AI Runtime.

Responsibilities:
- Initialize QAIR runtime components
- Discover and manage models
- Load and unload the inference engine
- Expose runtime state
- Provide controlled shutdown
"""

from __future__ import annotations

from runtime.inference.engine import InferenceEngine
from runtime.models.manager import ModelManager
from runtime.prompts.selection import PromptSelector


class QAIRRuntime:
    """
    Central runtime orchestration layer.

    QAIRRuntime coordinates model management, inference,
    and prompt selection without owning the interactive CLI.
    """

    def __init__(
        self,
        *,
        model_manager: ModelManager | None = None,
        engine: InferenceEngine | None = None,
        prompt_selector: PromptSelector | None = None,
    ) -> None:
        # Use explicitly supplied dependencies.
        # Only create defaults when None was supplied.
        self.model_manager = (
            model_manager
            if model_manager is not None
            else ModelManager()
        )

        self.engine = (
            engine
            if engine is not None
            else InferenceEngine(
                model_manager=self.model_manager
            )
        )

        self.prompt_selector = (
            prompt_selector
            if prompt_selector is not None
            else PromptSelector()
        )

        self.running = False

    # ==================================================
    # Lifecycle
    # ==================================================

    def start(self) -> None:
        """
        Start the QAIR runtime.

        Discovers models and loads the active model.
        """

        if self.running:
            return

        models = self.model_manager.list_models()

        if not models:
            raise RuntimeError("No AI models discovered.")

        active = self.model_manager.active_model()

        if active is None:
            raise RuntimeError("No active model selected.")

        self.engine.load()
        self.running = True

    def stop(self) -> None:
        """
        Stop QAIR and release inference resources.
        """

        if not self.running:
            return

        self.engine.unload()
        self.running = False

    # ==================================================
    # Runtime State
    # ==================================================

    @property
    def loaded(self) -> bool:
        """Return whether the inference engine is loaded."""

        return self.engine.loaded

    @property
    def active_model(self):
        """Return the currently active model."""

        return self.model_manager.active_model()

    @property
    def active_prompt(self) -> str:
        """Return the name of the active prompt."""

        return self.prompt_selector.DEFAULT_PROMPT

    # ==================================================
    # Model Management
    # ==================================================

    def refresh_models(self):
        """Rediscover available AI models."""

        return self.model_manager.refresh()

    def list_models(self):
        """Return discovered AI models."""

        return self.model_manager.list_models()

    def activate_model(self, model_name: str):
        """
        Validate and activate a model.

        If QAIR is running, reload the inference engine
        so the newly activated model becomes live.
        """

        model = self.model_manager.activate(model_name)

        if self.running:
            self.engine.reload()

        return model

    # ==================================================
    # Prompt Management
    # ==================================================

    def available_prompts(self) -> list[str]:
        """Return all available system prompts."""

        return self.prompt_selector.available()

    def get_prompt(self, name: str) -> str:
        """Return the contents of a named prompt."""

        return self.prompt_selector.select(name)

    # ==================================================
    # Runtime Information
    # ==================================================

    def summary(self) -> dict:
        """
        Return a complete runtime summary.
        """

        model_summary = self.model_manager.summary()
        engine_summary = self.engine.summary()

        return {
            "running": self.running,
            "loaded": self.loaded,
            "active_model": model_summary["active_model"],
            "installed_models": model_summary["installed_models"],
            "total_model_size": model_summary["total_size"],
            "model": engine_summary["model"],
            "context": engine_summary["context"],
            "gpu_layers": engine_summary["gpu_layers"],
            "temperature": engine_summary["temperature"],
            "top_p": engine_summary["top_p"],
            "max_tokens": engine_summary["max_tokens"],
        }

    # ==================================================
    # Context Manager
    # ==================================================

    def __enter__(self) -> "QAIRRuntime":
        """Start QAIR when entering a context."""

        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Stop QAIR when leaving a context."""

        self.stop()