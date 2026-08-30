"""
QAIR Knowledge Context

Builds controlled context from retrieved knowledge documents.
"""

from __future__ import annotations

from collections.abc import Callable

from runtime.knowledge.document import KnowledgeDocument


class KnowledgeContextBuilder:
    """
    Convert retrieved knowledge documents into model-ready context.

    The builder supports both:
    - a legacy character budget
    - an optional token budget using the actual model tokenizer
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
        *,
        max_tokens: int | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> str:
        """
        Build a structured context block from documents.

        The existing character limit remains the hard safety limit.
        When ``max_tokens`` and ``token_counter`` are supplied,
        the generated context is additionally constrained by the
        model token budget.
        """
        if not documents:
            return ""

        if max_tokens is not None and max_tokens <= 0:
            return ""

        if max_tokens is not None and token_counter is None:
            raise ValueError("token_counter is required when max_tokens is provided.")

        sections: list[str] = []
        total_length = 0
        total_tokens = 0

        for index, document in enumerate(documents, start=1):
            full_section = self._format_document(
                index,
                document,
            )

            separator = "\n\n" if sections else ""
            separator_length = len(separator)

            remaining_characters = self.max_characters - total_length - separator_length

            if remaining_characters <= 0:
                break

            section = full_section[:remaining_characters]

            if max_tokens is not None:
                assert token_counter is not None

                separator_tokens = token_counter(separator) if separator else 0

                remaining_tokens = max_tokens - total_tokens - separator_tokens

                if remaining_tokens <= 0:
                    break

                section = self._fit_text_to_tokens(
                    section,
                    remaining_tokens,
                    token_counter,
                )

                if not section:
                    break

                section_tokens = token_counter(section)

                if section_tokens > remaining_tokens:
                    break

                total_tokens += separator_tokens + section_tokens

            sections.append(section)
            total_length += separator_length + len(section)

            if len(section) < len(full_section):
                break

        return "\n\n".join(sections)

    @staticmethod
    def _fit_text_to_tokens(
        text: str,
        max_tokens: int,
        token_counter: Callable[[str], int],
    ) -> str:
        """
        Fit text to a token budget using binary search.

        This avoids repeatedly guessing tokenizer behavior.
        """
        if max_tokens <= 0 or not text:
            return ""

        if token_counter(text) <= max_tokens:
            return text

        low = 0
        high = len(text)
        best = ""

        while low <= high:
            middle = (low + high) // 2
            candidate = text[:middle]

            if token_counter(candidate) <= max_tokens:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1

        return best.rstrip()

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
