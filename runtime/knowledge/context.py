"""
QAIR Knowledge Context

Builds controlled context from retrieved knowledge documents.
"""

from __future__ import annotations

from runtime.knowledge.document import KnowledgeDocument


class KnowledgeContextBuilder:
    """
    Convert retrieved knowledge documents into model-ready context.
    """

    def __init__(
        self,
        *,
        max_characters: int = 12000,
    ) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero.")

        self.max_characters = max_characters

    def build(
        self,
        documents: list[KnowledgeDocument],
    ) -> str:
        """
        Build a structured context block from documents.
        """

        if not documents:
            return ""

        sections: list[str] = []
        total_length = 0

        for index, document in enumerate(documents, start=1):
            section = self._format_document(
                index,
                document,
            )

            separator_length = 2 if sections else 0

            if total_length + separator_length + len(section) > self.max_characters:
                remaining = self.max_characters - total_length - separator_length

                if remaining > 0:
                    sections.append(section[:remaining])

                break

            sections.append(section)

            total_length += separator_length + len(section)

        return "\n\n".join(sections)

    @staticmethod
    def _format_document(
        index: int,
        document: KnowledgeDocument,
    ) -> str:
        """Format one knowledge document."""

        title = document.title or "Untitled"
        source = document.source or "Unknown"

        return (
            f"[Knowledge {index}]\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Content:\n"
            f"{document.content}"
        )
