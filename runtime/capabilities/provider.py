"""QAIR capability provider."""

from __future__ import annotations

from runtime.capabilities.capability import Capability
from runtime.core.runtime import QAIRRuntime


class CapabilityProvider:
    """Expose capabilities derived from the current QAIR runtime."""

    INFERENCE_GENERATE = "inference.generate"
    KNOWLEDGE_RETRIEVE = "knowledge.retrieve"

    def __init__(self, runtime: QAIRRuntime) -> None:
        if not isinstance(runtime, QAIRRuntime):
            raise TypeError("runtime must be a QAIRRuntime.")

        self.runtime = runtime

    def list(self) -> list[Capability]:
        """Return all capabilities and their current availability."""

        return [
            Capability(
                name=self.INFERENCE_GENERATE,
                available=self._inference_available(),
                description=(
                    "Generate responses using the configured "
                    "QAIR inference backend."
                ),
            ),
            Capability(
                name=self.KNOWLEDGE_RETRIEVE,
                available=self._knowledge_available(),
                description=(
                    "Retrieve relevant information from the "
                    "QAIR knowledge system."
                ),
            ),
        ]

    def get(self, name: str) -> Capability | None:
        """Return a capability by name, or None when unknown."""

        if not isinstance(name, str):
            return None

        normalized_name = name.strip()

        for capability in self.list():
            if capability.name == normalized_name:
                return capability

        return None

    def can(self, name: str) -> bool:
        """Return whether a named capability is currently available."""

        capability = self.get(name)
        return capability is not None and capability.available

    def _inference_available(self) -> bool:
        """Return whether inference is currently available."""

        return self.runtime.running and self.runtime.loaded

    def _knowledge_available(self) -> bool:
        """Return whether knowledge retrieval is currently available."""

        return (
            self.runtime.running
            and self.runtime.knowledge_retriever is not None
        )
