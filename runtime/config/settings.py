"""
=========================================================
QAIR Settings Model
=========================================================
"""

from pydantic import BaseModel


class QAIRSettings(BaseModel):

    model: str

    model_path: str

    temperature: float

    top_p: float

    max_tokens: int

    context_size: int

    gpu_layers: int

    verbose: bool