"""
=========================================================
QAIR Configuration Loader
=========================================================
"""

from pathlib import Path

import yaml

from runtime.config.settings import QAIRSettings

CONFIG_FILE = Path(__file__).parent / "default.yaml"


def load_settings() -> QAIRSettings:
    """
    Load and validate QAIR configuration.
    """

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return QAIRSettings(**data)