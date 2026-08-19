from unittest.mock import MagicMock

from runtime.inference.engine import InferenceEngine


def test_engine_accepts_custom_model_manager():
    custom_manager = MagicMock()

    engine = InferenceEngine(model_manager=custom_manager)

    assert engine.manager is custom_manager