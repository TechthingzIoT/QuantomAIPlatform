"""
QAIR llama.cpp Embedding Provider

Concrete embedding implementation backed by llama.cpp.

This adapter implements the generic EmbeddingProvider interface
without exposing llama.cpp details to the rest of the knowledge
layer.
"""

from __future__ import annotations

from typing import Any

from llama_cpp import Llama

from runtime.knowledge.embeddings import EmbeddingProvider


class LlamaEmbeddingProvider(EmbeddingProvider):
    """
    Generate embeddings using a llama.cpp embedding model.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 512,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path

        self._model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text string.
        """

        if not isinstance(text, str):
            raise TypeError("Embedding input must be a string.")

        if not text.strip():
            raise ValueError("Embedding input cannot be empty.")

        response = self._model.create_embedding(text)

        return self._extract_embedding(response)

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        """

        if not isinstance(texts, list):
            raise TypeError("Embedding inputs must be a list.")

        if not texts:
            return []

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Every embedding input must be a string.")

            if not text.strip():
                raise ValueError("Embedding input cannot be empty.")

        response = self._model.create_embedding(texts)

        return self._extract_embeddings(response)

    @staticmethod
    def _extract_embedding(
        response: Any,
    ) -> list[float]:
        """
        Extract a single embedding vector from a llama.cpp
        embedding response.
        """

        data = response["data"]

        if not data:
            raise ValueError("Embedding response contains no data.")

        embedding = data[0]["embedding"]

        return [float(value) for value in embedding]

    @staticmethod
    def _extract_embeddings(
        response: Any,
    ) -> list[list[float]]:
        """
        Extract multiple embedding vectors from a llama.cpp
        embedding response.
        """

        data = response["data"]

        return [[float(value) for value in item["embedding"]] for item in data]
