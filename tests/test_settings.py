from runtime.config.settings import QAIRSettings, load_settings, settings


def test_settings_instance():
    assert isinstance(settings, QAIRSettings)


def test_settings_model():
    assert settings.model


def test_settings_model_path():
    assert settings.model_path


def test_settings_embedding_model_defaults_to_none():
    assert settings.embedding_model is None


def test_settings_embedding_model_path_defaults_to_none():
    assert settings.embedding_model_path is None


def test_settings_temperature():
    assert settings.temperature == 0.3


def test_settings_top_p():
    assert settings.top_p == 0.95


def test_settings_max_tokens():
    assert settings.max_tokens == 256


def test_settings_context_size():
    assert settings.context_size == 2048


def test_settings_gpu_layers():
    assert settings.gpu_layers == 18


def test_settings_verbose():
    assert settings.verbose is False


def test_load_settings():
    loaded = load_settings()

    assert isinstance(loaded, QAIRSettings)
    assert loaded.model == settings.model
    assert loaded.model_path == settings.model_path
    assert loaded.embedding_model == settings.embedding_model
    assert loaded.embedding_model_path == settings.embedding_model_path
