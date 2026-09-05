"""
QAIR Model Validator
"""

from pathlib import Path


def validate_model(path: Path):

    if not path.exists():
        return False, "Model does not exist."

    if not path.is_file():
        return False, "Not a file."

    if path.suffix != ".gguf":
        return False, "Not a GGUF model."

    if path.stat().st_size == 0:
        return False, "Empty model."

    return True, "OK"