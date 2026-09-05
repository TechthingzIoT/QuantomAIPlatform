from runtime.models.manager import ModelManager


def test_model_manager_creation():
    manager = ModelManager()
    assert manager is not None


def test_list_models():
    manager = ModelManager()

    models = manager.list_models()

    assert isinstance(models, list)


def test_activate_validates_model_path(monkeypatch):
    from pathlib import Path
    from unittest.mock import MagicMock

    from runtime.models.manager import ModelManager
    from runtime.models.model import Model

    model = Model(
        name="test-model.gguf",
        path=Path("/tmp/test-model.gguf"),
        size=1024,
        extension=".gguf",
    )

    manager = ModelManager()
    manager._models = [model]

    validate_model = MagicMock(return_value=(True, "OK"))
    monkeypatch.setattr(
        "runtime.models.manager.validate_model",
        validate_model,
    )
    monkeypatch.setattr(
        "runtime.models.manager.set_active_model",
        lambda name: None,
    )

    result = manager.activate("test-model.gguf")

    validate_model.assert_called_once_with(model.path)
    assert result is model
