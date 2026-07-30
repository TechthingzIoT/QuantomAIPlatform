from pathlib import Path

from runtime.config.settings import MODELS_DIR


class ModelManager:
    def __init__(self):
        MODELS_DIR.mkdir(exist_ok=True)

    def list_models(self):
        """Return all GGUF models."""
        return sorted(MODELS_DIR.glob("*.gguf"))

    def exists(self, model_name: str) -> bool:
        return (MODELS_DIR / model_name).exists()

    def get(self, model_name: str) -> Path:
        path = MODELS_DIR / model_name

        if not path.exists():
            raise FileNotFoundError(f"{model_name} not found")

        return path
