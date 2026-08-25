from pathlib import Path

import pytest

from runtime.knowledge import KnowledgeLoader


def test_loader_reads_markdown_documents(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    (knowledge_dir / "rwanda_ai.md").write_text(
        "# Rwanda AI\n\nRwanda is developing AI capability.",
        encoding="utf-8",
    )

    loader = KnowledgeLoader(knowledge_dir)

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].id == "rwanda_ai"
    assert documents[0].title == "Rwanda AI"
    assert documents[0].content.startswith("# Rwanda AI")
    assert documents[0].metadata["format"] == "markdown"


def test_loader_processes_files_deterministically(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    (knowledge_dir / "zulu.md").write_text(
        "# Zulu",
        encoding="utf-8",
    )
    (knowledge_dir / "alpha.md").write_text(
        "# Alpha",
        encoding="utf-8",
    )

    documents = KnowledgeLoader(knowledge_dir).load()

    assert [document.id for document in documents] == [
        "alpha",
        "zulu",
    ]


def test_loader_rejects_missing_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        KnowledgeLoader(tmp_path / "missing").load()


def test_loader_rejects_file_path(tmp_path: Path):
    path = tmp_path / "knowledge.md"
    path.write_text("# Knowledge", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        KnowledgeLoader(path).load()


def test_loader_rejects_empty_document(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    (knowledge_dir / "empty.md").write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="empty"):
        KnowledgeLoader(knowledge_dir).load()


def test_loader_uses_filename_when_title_is_missing(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    (knowledge_dir / "ai_sovereignty.md").write_text(
        "AI sovereignty enables local capability.",
        encoding="utf-8",
    )

    documents = KnowledgeLoader(knowledge_dir).load()

    assert documents[0].title == "Ai Sovereignty"
