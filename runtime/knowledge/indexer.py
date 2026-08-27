"""
QAIR Knowledge Indexer

Enriches knowledge documents with vector embeddings.

The indexer is intentionally separate from the knowledge loader:
the loader parses source files, while the indexer performs
embedding generation.
"""

from __future__ import annotations

from runtime.knowledge.document import KnowledgeDocument
from runtime.knowledge.embeddings import EmbeddingProvider


class KnowledgeIndexer:
    """
    Generate embeddings for knowledge documents.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.embedding_provider = embedding_provider

    def index(
        self,
        documents: list[KnowledgeDocument],
    ) -> list[KnowledgeDocument]:
        """
        Generate embeddings and attach them to documents.

        Documents are modified in place and returned for
        convenient chaining.
        """
        if not documents:
            return []

        texts = [self._embedding_text(document) for document in documents]

        embeddings = self.embedding_provider.embed_many(texts)

        if len(embeddings) != len(documents):
            raise ValueError(
                "Embedding provider returned a different number "
                "of embeddings than documents."
            )

        for document, embedding in zip(
            documents,
            embeddings,
        ):
            document.embedding = embedding

        return documents

    @staticmethod
    def _embedding_text(
        document: KnowledgeDocument,
    ) -> str:
        """
        Build the text representation used for embedding.
        """
        title = document.title or ""

        return f"Title: {title}\n" f"Content:\n" f"{document.content}"
