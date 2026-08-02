"""
=========================================================
QAIR Command Line Interface
=========================================================

Professional CLI for the Quantom AI Runtime.

Author:
    TIOTAIROBOTIX
=========================================================
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from runtime.models.manager import ModelManager
from runtime.models.registry import set_active_model

app = typer.Typer(
    name="qair",
    help="Quantom AI Runtime",
    no_args_is_help=False,
)

models_app = typer.Typer(help="Manage local AI models")

app.add_typer(models_app, name="models")

console = Console()


# =========================================================
# Root command
# =========================================================

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    Launch QAIR interactive mode when no subcommand is supplied.
    """

    if ctx.invoked_subcommand is not None:
        return

    from runtime.chat.session import ChatSession

    ChatSession().run()


# =========================================================
# MODELS
# =========================================================

@models_app.command("list")
def list_models():
    """
    List installed GGUF models.
    """

    manager = ModelManager()

    models = manager.list_models()

    if not models:
        console.print("[yellow]No models discovered.[/yellow]")
        raise typer.Exit()

    table = Table(title="Installed Models")

    table.add_column("Name", style="cyan")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Extension")

    for model in models:
        table.add_row(
            model.name,
            f"{model.size / (1024 * 1024):.1f}",
            model.extension,
        )

    console.print(table)


@models_app.command("active")
def active_model():
    """
    Show active model.
    """

    manager = ModelManager()

    model = manager.active_model()

    if model is None:
        console.print("[yellow]No active model selected.[/yellow]")
        raise typer.Exit()

    console.print(f"[green]Active Model:[/green] {model.name}")


@models_app.command("use")
def use_model(model_name: str):
    """
    Set active model.
    """

    manager = ModelManager()

    manager.set_current_model(model_name)

    console.print(f"[green]✓ Active model changed to[/green] {model_name}")


@models_app.command("info")
def model_info():
    """
    Display model summary.
    """

    manager = ModelManager()

    summary = manager.summary()

    table = Table(title="QAIR Model Summary")

    table.add_column("Property")
    table.add_column("Value")

    for key, value in summary.items():
        table.add_row(str(key), str(value))

    console.print(table)


@models_app.command("refresh")
def refresh():
    """
    Rediscover installed models.
    """

    manager = ModelManager()

    models = manager.list_models()

    console.print(
        f"[green]✓[/green] Discovered {len(models)} model(s)."
    )


# =========================================================
# VERSION
# =========================================================

@app.command()
def version():
    """
    Show QAIR version.
    """

    console.print("QAIR v0.4.0")


# =========================================================
# ENTRY POINT
# =========================================================

# =========================================================
# ENTRY POINT
# =========================================================

def run():
    """
    Console entry point for the qair executable.
    """

    app()


if __name__ == "__main__":
    run()