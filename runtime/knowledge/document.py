"""
QAIR Knowledge Document

Represents a single piece of knowledge available to
the QAIR retrieval layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class KnowledgeDocument:
    """
    Represents one knowledge document.

    A document contains source text, lightweight metadata,
    and optionally a precomputed embedding vector.
    """

    id: str
    content: str
    title: str | None = None
    source: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict:
        """Convert the document to a serializable dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "source": self.source,
            "metadata": self.metadata.copy(),
            "embedding": (
                self.embedding.copy()
                if self.embedding is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> KnowledgeDocument:
        """Create a document from a dictionary."""
        if "id" not in data:
            raise ValueError(
                "Document data missing required field: 'id'."
            )
        if "content" not in data:
            raise ValueError(
                "Document data missing required field: 'content'."
            )
        embedding = data.get("embedding")
        return cls(
            id=str(data["id"]),
            content=str(data["content"]),
            title=data.get("title"),
            source=data.get("source"),
            metadata=dict(data.get("metadata", {})),
            embedding=(
                [float(value) for value in embedding]
                if embedding is not None
                else None
            ),
        )
