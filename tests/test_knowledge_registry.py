from importlib import reload
from pathlib import Path

import pytest
import yaml

from runtime.knowledge import registry


@pytest.fixture
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide a completely isolated registry for each test."""
    module = reload(registry)

    registry_dir = tmp_path / ".qair"
    registry_file = registry_dir / "knowledge.yaml"

    monkeypatch.setattr(module, "REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(module, "REGISTRY_FILE", registry_file)

    return module


def test_load_registry_creates_default_registry(
    isolated_registry,
):
    registry = isolated_registry

    data = registry.load_registry()

    assert data == {"sources": []}
    assert registry.REGISTRY_FILE.exists()


def test_load_registry_reads_persisted_sources(
    tmp_path: Path,
    isolated_registry,
):
    registry = isolated_registry

    registry.REGISTRY_DIR.mkdir(parents=True)
    registry.REGISTRY_FILE.write_text(
        yaml.safe_dump(
            {
                "sources": ["/tmp/knowledge"],
            }
        ),
        encoding="utf-8",
    )

    data = registry.load_registry()

    assert data["sources"] == ["/tmp/knowledge"]


def test_add_source_persists_normalized_path(
    tmp_path: Path,
    isolated_registry,
):
    registry = isolated_registry
    source = tmp_path / "knowledge"

    registry.add_source(source)

    assert registry.list_sources() == [str(source.resolve())]


def test_add_source_does_not_duplicate_source(
    tmp_path: Path,
    isolated_registry,
):
    registry = isolated_registry
    source = tmp_path / "knowledge"

    registry.add_source(source)
    registry.add_source(source)

    assert registry.list_sources() == [str(source.resolve())]


def test_remove_source_removes_registered_source(
    tmp_path: Path,
    isolated_registry,
):
    registry = isolated_registry

    first = tmp_path / "first"
    second = tmp_path / "second"

    registry.add_source(first)
    registry.add_source(second)

    registry.remove_source(first)

    assert registry.list_sources() == [str(second.resolve())]


def test_remove_source_is_safe_when_source_is_missing(
    tmp_path: Path,
    isolated_registry,
):
    registry = isolated_registry
    source = tmp_path / "knowledge"

    registry.remove_source(source)

    assert registry.list_sources() == []


def test_clear_sources_removes_all_registered_sources(
    tmp_path: Path,
    isolated_registry,
):
    registry = isolated_registry

    registry.add_source(tmp_path / "first")
    registry.add_source(tmp_path / "second")

    registry.clear_sources()

    assert registry.list_sources() == []
    assert registry.load_registry() == {"sources": []}


def test_load_registry_recovers_invalid_sources_value(
    isolated_registry,
):
    registry = isolated_registry

    registry.REGISTRY_DIR.mkdir(parents=True)
    registry.REGISTRY_FILE.write_text(
        "sources: invalid\n",
        encoding="utf-8",
    )

    data = registry.load_registry()

    assert data["sources"] == []


def test_save_registry_creates_parent_directory(
    isolated_registry,
):
    registry = isolated_registry

    registry.save_registry({"sources": ["/tmp/knowledge"]})

    assert registry.REGISTRY_FILE.exists()
    assert registry.load_registry()["sources"] == ["/tmp/knowledge"]
