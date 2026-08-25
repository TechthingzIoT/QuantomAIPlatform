"""
QAIR Knowledge Loader

Loads local Markdown knowledge files into KnowledgeDocument objects.

The loader deliberately performs no inference and no embedding generation.
It is responsible only for turning a local knowledge directory into
structured QAIR knowledge documents.
"""

from __future__ import annotations

from pathlib import Path

from runtime.knowledge.document import KnowledgeDocument


class KnowledgeLoader:
    """Load Markdown knowledge documents from a local directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def load(self) -> list[KnowledgeDocument]:
        """
        Load all Markdown files from the configured directory.

        Files are processed in deterministic filename order.
        """
        if not self.directory.exists():
            raise FileNotFoundError(f"Knowledge directory not found: {self.directory}")

        if not self.directory.is_dir():
            raise NotADirectoryError(
                f"Knowledge path is not a directory: {self.directory}"
            )

        documents: list[KnowledgeDocument] = []

        for path in sorted(self.directory.glob("*.md")):
            documents.append(self._load_file(path))

        return documents

    @staticmethod
    def _load_file(path: Path) -> KnowledgeDocument:
        """Convert one Markdown file into a KnowledgeDocument."""
        content = path.read_text(encoding="utf-8").strip()

        if not content:
            raise ValueError(f"Knowledge document is empty: {path}")

        title = KnowledgeLoader._extract_title(
            content,
            fallback=path.stem.replace("_", " ").title(),
        )

        return KnowledgeDocument(
            id=path.stem,
            title=title,
            source=str(path),
            content=content,
            metadata={
                "format": "markdown",
                "filename": path.name,
            },
        )

    @staticmethod
    def _extract_title(
        content: str,
        *,
        fallback: str,
    ) -> str:
        """Extract the first Markdown H1 heading."""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

        return fallback
