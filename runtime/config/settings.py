"""
=========================================================
QAIR Settings
=========================================================
Defines the QAIR configuration model and exposes
the global runtime settings instance.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel


class QAIRSettings(BaseModel):
    model: str
    model_path: str

    # Optional local embedding model used by the knowledge layer.
    embedding_model: str | None = None
    embedding_model_path: str | None = None
    embedding_context_size: int = 512
    temperature: float
    top_p: float
    max_tokens: int
    context_size: int
    gpu_layers: int
    verbose: bool


CONFIG_FILE = Path(__file__).parent / "default.yaml"


def load_settings() -> QAIRSettings:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return QAIRSettings(**data)


# Global settings singleton
settings = load_settings()
