from runtime.models.manager import ModelManager


def test_model_manager_creation():
    manager = ModelManager()
    assert manager is not None


def test_list_models():
    manager = ModelManager()

    models = manager.list_models()

    assert isinstance(models, list)