"""
QAIR Knowledge Retriever

Hybrid retrieval implementation.

Supports:
- deterministic keyword retrieval
- semantic vector retrieval
- hybrid keyword + semantic ranking

The embedding provider is optional so the knowledge layer
remains usable without an embedding model.
"""

from __future__ import annotations

import re

from runtime.knowledge.document import KnowledgeDocument
from runtime.knowledge.embeddings import EmbeddingProvider
from runtime.knowledge.similarity import cosine_similarity
from runtime.knowledge.store import KnowledgeStore


class KnowledgeRetriever:
    """
    Retrieve relevant documents from a KnowledgeStore.

    When an embedding provider is supplied and documents
    contain embeddings, semantic similarity is incorporated
    into the ranking.

    Without an embedding provider, retrieval falls back to
    deterministic keyword matching.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        embedding_provider: EmbeddingProvider | None = None,
        *,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
        min_score: float = 0.0,
    ) -> None:
        if min_score < 0:
            raise ValueError("min_score cannot be negative.")
        if keyword_weight < 0:
            raise ValueError("keyword_weight cannot be negative.")

        if semantic_weight < 0:
            raise ValueError("semantic_weight cannot be negative.")

        if keyword_weight == 0 and semantic_weight == 0:
            raise ValueError("At least one retrieval weight must be greater than zero.")

        self.store = store
        self.embedding_provider = embedding_provider
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.min_score = min_score

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[KnowledgeDocument]:
        """
        Retrieve documents relevant to a query.

        Keyword-only retrieval is used when no embedding
        provider is configured.

        Hybrid retrieval is used when an embedding provider
        is available and at least one document has an embedding.
        """

        if not query.strip():
            return []

        if limit <= 0:
            return []

        documents = self.store.all()

        if not documents:
            return []

        query_embedding = self._query_embedding(query)

        scored: list[tuple[float, KnowledgeDocument]] = []

        for document in documents:
            keyword_score = self._keyword_score(
                document,
                query,
            )

            semantic_score = 0.0

            if query_embedding is not None and document.embedding is not None:
                try:
                    semantic_score = cosine_similarity(
                        query_embedding,
                        document.embedding,
                    )
                except ValueError:
                    semantic_score = 0.0

            if query_embedding is not None:
                score = (
                    self.keyword_weight * keyword_score
                    + self.semantic_weight * semantic_score
                )
            else:
                score = keyword_score

            if score > 0 and score >= self.min_score:
                scored.append((score, document))

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [document for _, document in scored[:limit]]

    def _query_embedding(
        self,
        query: str,
    ) -> list[float] | None:
        """Generate an embedding for the query when possible."""

        if self.embedding_provider is None:
            return None

        return self.embedding_provider.embed(query)

    @classmethod
    def _keyword_score(
        cls,
        document: KnowledgeDocument,
        query: str,
    ) -> float:
        """Calculate bounded keyword relevance from query-term coverage."""
        searchable = " ".join(
            [
                document.title or "",
                document.source or "",
                document.content,
            ]
        )

        searchable_terms = set(cls._tokenize(searchable))
        query_terms = set(cls._tokenize(query))

        if not query_terms:
            return 0.0

        matched_terms = searchable_terms.intersection(query_terms)

        return len(matched_terms) / len(query_terms)

    @staticmethod
    def _tokenize(query: str) -> list[str]:
        """Convert a query into normalized search terms."""

        return re.findall(
            r"\b[a-zA-Z0-9_]+\b",
            query.lower(),
        )
