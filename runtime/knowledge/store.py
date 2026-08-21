"""
QAIR Knowledge Store

Minimal in-memory knowledge repository.

The storage interface is intentionally simple so it can
later be backed by a vector database without changing
the rest of QAIR.
"""

from __future__ import annotations

from runtime.knowledge.document import KnowledgeDocument


class KnowledgeStore:
    """
    Store and retrieve knowledge documents.

    This implementation uses an in-memory dictionary.
    """

    def __init__(self) -> None:
        self._documents: dict[str, KnowledgeDocument] = {}

    def add(self, document: KnowledgeDocument) -> None:
        """Add or replace a document."""

        self._documents[document.id] = document

    def add_many(
        self,
        documents: list[KnowledgeDocument],
    ) -> None:
        """Add multiple documents."""

        for document in documents:
            self.add(document)

    def get(
        self,
        document_id: str,
    ) -> KnowledgeDocument | None:
        """Return a document by ID."""

        return self._documents.get(document_id)

    def remove(self, document_id: str) -> None:
        """Remove a document if it exists."""

        self._documents.pop(document_id, None)

    def clear(self) -> None:
        """Remove all documents."""

        self._documents.clear()

    def all(self) -> list[KnowledgeDocument]:
        """Return all stored documents."""

        return list(self._documents.values())

    def __len__(self) -> int:
        """Return the number of stored documents."""

        return len(self._documents)