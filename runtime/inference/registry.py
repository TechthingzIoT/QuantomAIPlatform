"""
QAIR inference backend registry.

The registry maps backend names to provider implementations.
It provides a single extension point for future backends such
as Ollama, vLLM, remote APIs, or other inference providers.
"""

from __future__ import annotations

from collections.abc import Callable

from runtime.config.settings import QAIRSettings
from runtime.inference.backend import InferenceBackend
from runtime.inference.llama_cpp import LlamaCppBackend

BackendFactory = Callable[[QAIRSettings], InferenceBackend]


class BackendRegistry:
    """Registry used to resolve QAIR inference backends."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(
        self,
        name: str,
        factory: BackendFactory,
    ) -> None:
        """Register a backend factory."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        normalized_name = name.strip().lower()

        if not normalized_name:
            raise ValueError("name cannot be empty.")

        if not callable(factory):
            raise TypeError("factory must be callable.")

        self._factories[normalized_name] = factory

    def resolve(
        self,
        name: str,
        runtime_settings: QAIRSettings,
    ) -> InferenceBackend:
        """Create the backend registered under ``name``."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        normalized_name = name.strip().lower()

        if not normalized_name:
            raise ValueError("name cannot be empty.")

        factory = self._factories.get(normalized_name)

        if factory is None:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"Unknown inference backend '{normalized_name}'. "
                f"Available backends: {available}"
            )

        backend = factory(runtime_settings)

        if not isinstance(backend, InferenceBackend):
            raise TypeError(
                f"Backend factory '{normalized_name}' returned "
                "an object that does not implement InferenceBackend."
            )

        return backend

    def contains(self, name: str) -> bool:
        """Return whether a backend is registered."""
        if not isinstance(name, str):
            return False

        return name.strip().lower() in self._factories

    def names(self) -> tuple[str, ...]:
        """Return registered backend names."""
        return tuple(sorted(self._factories))


def create_default_backend_registry() -> BackendRegistry:
    """Create the registry containing QAIR's built-in backends."""
    registry = BackendRegistry()
    registry.register("llama_cpp", LlamaCppBackend)
    return registry


backend_registry = create_default_backend_registry()
