"""
=========================================================
QAIR Model Metadata
=========================================================
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Model:
    """
    Represents one discovered AI model.
    """

    name: str
    path: Path
    size: int
    extension: str

    @property
    def size_mb(self) -> float:
        return self.size / (1024 * 1024)