"""
=========================================================
QAIR Model Discovery
=========================================================

Discovers GGUF models from one or more directories.

Author:
    TIOTAIROBOTIX
=========================================================
"""

from pathlib import Path

from runtime.models.model import Model

SUPPORTED_EXTENSIONS = {".gguf"}


def discover_models(*directories: str | Path) -> list[Model]:
    """
    Discover all supported GGUF models recursively.

    Args:
        directories:
            One or more directories to scan.

    Returns:
        Sorted list of unique Model objects.
    """

    discovered: dict[Path, Model] = {}

    for directory in directories:

        root = Path(directory).expanduser()

        if not root.exists():
            continue

        for file in root.rglob("*"):

            if not file.is_file():
                continue

            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Ignore Hugging Face blob storage
            if "blobs" in file.parts:
                continue

            model = Model(
    name=file.name,
    path=file.absolute(),
    size=file.stat().st_size,
    extension=file.suffix.lower(),
)


            # Prevent duplicates caused by symbolic links
            discovered[model.path] = model

    return sorted(
        discovered.values(),
        key=lambda model: model.name.lower(),
    )


def default_search_paths() -> list[Path]:
    """
    Default QAIR model search locations.
    """

    return [
        Path.cwd() / "models",
        Path.home() / "Models",
        Path.home() / ".cache" / "huggingface",
    ]


def discover_default_models() -> list[Model]:
    """
    Discover models using QAIR's default search paths.
    """

    return discover_models(*default_search_paths())