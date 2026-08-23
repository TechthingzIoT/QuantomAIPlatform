"""
Tests for the QAIR command-line interface.
"""

from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from runtime.cli import app

runner = CliRunner()


# =========================================================
# ROOT / HELP
# =========================================================


def test_cli_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Quantom AI Runtime" in result.stdout
    assert "models" in result.stdout
    assert "version" in result.stdout
    assert "serve" in result.stdout


# =========================================================
# SERVER
# =========================================================


def test_serve_help():
    result = runner.invoke(
        app,
        ["serve", "--help"],
    )

    assert result.exit_code == 0
    assert "Launch the QAIR HTTP API server" in result.stdout
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--reload" in result.stdout


@patch("uvicorn.run")
def test_serve_defaults(mock_uvicorn_run):
    result = runner.invoke(
        app,
        ["serve"],
    )

    assert result.exit_code == 0

    mock_uvicorn_run.assert_called_once_with(
        "runtime.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


@patch("uvicorn.run")
def test_serve_custom_host_port_and_reload(
    mock_uvicorn_run,
):
    result = runner.invoke(
        app,
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--reload",
        ],
    )

    assert result.exit_code == 0

    mock_uvicorn_run.assert_called_once_with(
        "runtime.api.app:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
    )


def test_serve_rejects_zero_port():
    result = runner.invoke(
        app,
        [
            "serve",
            "--port",
            "0",
        ],
    )

    assert result.exit_code != 0


def test_serve_rejects_negative_port():
    result = runner.invoke(
        app,
        [
            "serve",
            "--port",
            "-1",
        ],
    )

    assert result.exit_code != 0


def test_serve_rejects_port_above_maximum():
    result = runner.invoke(
        app,
        [
            "serve",
            "--port",
            "65536",
        ],
    )

    assert result.exit_code != 0


# =========================================================
# MODELS HELP
# =========================================================


def test_models_help():
    result = runner.invoke(
        app,
        ["models", "--help"],
    )

    assert result.exit_code == 0
    assert "Manage local AI models" in result.stdout
    assert "list" in result.stdout
    assert "active" in result.stdout
    assert "use" in result.stdout
    assert "info" in result.stdout
    assert "refresh" in result.stdout


# =========================================================
# VERSION
# =========================================================


def test_version():
    result = runner.invoke(
        app,
        ["version"],
    )

    assert result.exit_code == 0
    assert "QAIR v0.6.0" in result.stdout


# =========================================================
# MODELS LIST
# =========================================================


@patch("runtime.cli.ModelManager")
def test_models_list(mock_manager):
    mock_manager.return_value.list_models.return_value = [
        SimpleNamespace(
            name="test-model.gguf",
            size=1024 * 1024 * 100,
            extension=".gguf",
        )
    ]

    mock_manager.return_value.active_model.return_value = SimpleNamespace(
        name="test-model.gguf",
    )

    result = runner.invoke(
        app,
        ["models", "list"],
    )

    assert result.exit_code == 0
    assert "Installed Models" in result.stdout
    assert "test-model.gguf" in result.stdout
    assert "100.0" in result.stdout


# =========================================================
# ACTIVE MODEL
# =========================================================


@patch("runtime.cli.ModelManager")
def test_models_active(mock_manager):
    mock_manager.return_value.active_model.return_value = SimpleNamespace(
        name="test-model.gguf",
    )

    result = runner.invoke(
        app,
        ["models", "active"],
    )

    assert result.exit_code == 0
    assert "Active Model" in result.stdout
    assert "test-model.gguf" in result.stdout


@patch("runtime.cli.ModelManager")
def test_models_active_none(mock_manager):
    mock_manager.return_value.active_model.return_value = None

    result = runner.invoke(
        app,
        ["models", "active"],
    )

    assert result.exit_code == 0
    assert "No active model selected." in result.stdout


# =========================================================
# MODEL INFO
# =========================================================


@patch("runtime.cli.ModelManager")
def test_models_info(mock_manager):
    mock_manager.return_value.summary.return_value = {
        "installed_models": 1,
        "active_model": "test-model.gguf",
        "total_size": 104857600,
    }

    result = runner.invoke(
        app,
        ["models", "info"],
    )

    assert result.exit_code == 0
    assert "QAIR Model Summary" in result.stdout
    assert "installed_models" in result.stdout
    assert "test-model.gguf" in result.stdout


# =========================================================
# MODEL REFRESH
# =========================================================


@patch("runtime.cli.ModelManager")
def test_models_refresh(mock_manager):
    mock_manager.return_value.refresh.return_value = [
        SimpleNamespace(
            name="test-model.gguf",
        )
    ]

    result = runner.invoke(
        app,
        ["models", "refresh"],
    )

    assert result.exit_code == 0
    assert "Discovered 1 model" in result.stdout


# =========================================================
# MODEL USE
# =========================================================


def test_models_use_missing_argument():
    result = runner.invoke(
        app,
        ["models", "use"],
    )

    assert result.exit_code != 0


@patch("runtime.cli.ModelManager")
def test_models_use_invalid_model(mock_manager):
    mock_manager.return_value.set_current_model.side_effect = ValueError(
        "Model 'missing.gguf' not found."
    )

    result = runner.invoke(
        app,
        [
            "models",
            "use",
            "missing.gguf",
        ],
    )

    assert result.exit_code != 0
    assert "missing.gguf" in result.stdout
