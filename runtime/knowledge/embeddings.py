"""
QAIR Embedding Provider

Defines the abstraction used by QAIR to convert text into
numeric vector representations.

Concrete embedding implementations can later use local
models, remote providers, or specialized hardware without
changing the knowledge layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Abstract interface for text embedding providers.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Convert a single text string into an embedding vector.
        """

        raise NotImplementedError

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Convert multiple text strings into embedding vectors.
        """

        return [
            self.embed(text)
            for text in texts
        ]