"""
QAIR Knowledge Registry

Persists local knowledge source directories.

The registry stores knowledge locations, not copies of the
knowledge documents themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REGISTRY_DIR = Path.home() / ".qair"
REGISTRY_FILE = REGISTRY_DIR / "knowledge.yaml"

DEFAULT_REGISTRY: dict[str, Any] = {
    "sources": [],
}


def _default_registry() -> dict[str, Any]:
    """Return a fresh default registry."""
    return {
        "sources": [],
    }


def load_registry() -> dict[str, Any]:
    """Load the knowledge registry from disk."""
    if not REGISTRY_FILE.exists():
        save_registry(_default_registry())
        return _default_registry()

    with REGISTRY_FILE.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        data = _default_registry()

    sources = data.get("sources")
    if not isinstance(sources, list):
        data["sources"] = []

    return data


def save_registry(data: dict[str, Any]) -> None:
    """Save the knowledge registry to disk."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    with REGISTRY_FILE.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            default_flow_style=False,
        )


def list_sources() -> list[str]:
    """Return registered knowledge source directories."""
    return list(load_registry()["sources"])


def add_source(path: str | Path) -> None:
    """Register a knowledge source directory."""
    source = str(Path(path).expanduser().resolve())

    registry = load_registry()

    if source not in registry["sources"]:
        registry["sources"].append(source)
        save_registry(registry)


def remove_source(path: str | Path) -> None:
    """Remove a registered knowledge source directory."""
    source = str(Path(path).expanduser().resolve())

    registry = load_registry()
    registry["sources"] = [item for item in registry["sources"] if item != source]

    save_registry(registry)


def clear_sources() -> None:
    """Remove all registered knowledge sources."""
    save_registry(_default_registry())
