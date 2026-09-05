"""
=========================================================
QAIR Model Registry
=========================================================

Stores information about installed models and
the currently active model.

Registry location:
    ~/.qair/registry.yaml
"""

from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------
# Registry Paths
# --------------------------------------------------------

REGISTRY_DIR = Path.home() / ".qair"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = REGISTRY_DIR / "registry.yaml"


# --------------------------------------------------------
# Default Registry
# --------------------------------------------------------

DEFAULT_REGISTRY = {
    "active_model": None,
    "installed_models": [],
}


# --------------------------------------------------------
# Registry Functions
# --------------------------------------------------------

def load_registry() -> dict[str, Any]:
    """
    Load the registry.

    Creates a default registry if one does not exist.
    """

    if not REGISTRY_FILE.exists():
        save_registry(DEFAULT_REGISTRY)
        return DEFAULT_REGISTRY.copy()

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = DEFAULT_REGISTRY.copy()

    data.setdefault("active_model", None)
    data.setdefault("installed_models", [])

    return data


def save_registry(data: dict[str, Any]) -> None:
    """
    Save registry to disk.
    """

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
        )


def get_active_model() -> str | None:
    """
    Return the active model name.
    """

    return load_registry()["active_model"]


def set_active_model(model_name: str) -> None:
    """
    Set the active model.
    """

    registry = load_registry()
    registry["active_model"] = model_name
    save_registry(registry)


def list_registered_models() -> list:
    """
    Return installed models from registry.
    """

    return load_registry()["installed_models"]


def register_model(model_info: dict[str, Any]) -> None:
    """
    Register a model if it is not already present.
    """

    registry = load_registry()

    models = registry["installed_models"]

    for model in models:
        if model["name"] == model_info["name"]:
            return

    models.append(model_info)

    save_registry(registry)


def unregister_model(model_name: str) -> None:
    """
    Remove a model from registry.
    """

    registry = load_registry()

    registry["installed_models"] = [
        model
        for model in registry["installed_models"]
        if model["name"] != model_name
    ]

    if registry["active_model"] == model_name:
        registry["active_model"] = None

    save_registry(registry)