from pathlib import Path
import yaml


class Settings:
    def __init__(self):
        config_path = Path(__file__).parent / "default.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def __getattr__(self, key):
        return self.config.get(key)


settings = Settings()
