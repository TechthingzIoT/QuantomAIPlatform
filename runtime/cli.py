"""
QAIR Command Line Interface

Professional CLI for the Quantom AI Runtime.

Responsibilities
----------------
- Launch interactive QAIR sessions
- Select system prompts
- Manage local GGUF models
- Display runtime/model information
- Provide version information
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from runtime.models.manager import ModelManager


# =========================================================
# APPLICATION
# =========================================================

app = typer.Typer(
    name="qair",
    help="Quantom AI Runtime",
    no_args_is_help=False,
)

models_app = typer.Typer(
    name="models",
    help="Manage local AI models",
)

app.add_typer(
    models_app,
    name="models",
)

console = Console()


# =========================================================
# ROOT COMMAND
# =========================================================

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: str = typer.Option(
        "assistant",
        "--prompt",
        "-p",
        help="System prompt/domain to use.",
    ),
) -> None:
    """
    Launch QAIR interactive mode when no subcommand
    is supplied.
    """

    # -----------------------------------------------------
    # IMPORTANT:
    # If a subcommand such as `qair models list` or
    # `qair version` was supplied, do not launch ChatSession.
    # -----------------------------------------------------

    if ctx.invoked_subcommand is not None:
        return

    from runtime.chat.session import ChatSession

    try:
        ChatSession(prompt_name=prompt).run()

    except ValueError as exc:
        console.print(
            f"[red]✗ {exc}[/red]"
        )

        raise typer.Exit(code=1)

    except RuntimeError as exc:
        console.print(
            f"[red]✗ QAIR runtime error:[/red] {exc}"
        )

        raise typer.Exit(code=1)


# =========================================================
# MODELS
# =========================================================

@models_app.command("list")
def list_models() -> None:
    """
    List installed GGUF models.
    """

    manager = ModelManager()

    models = manager.list_models()

    if not models:
        console.print(
            "[yellow]No models discovered.[/yellow]"
        )
        raise typer.Exit()

    table = Table(
        title="Installed Models",
        show_header=True,
    )

    table.add_column(
        "Name",
        style="cyan",
    )

    table.add_column(
        "Size (MB)",
        justify="right",
    )

    table.add_column(
        "Extension",
    )

    table.add_column(
        "Active",
        justify="center",
    )

    active = manager.active_model()

    active_name = (
        active.name
        if active is not None
        else None
    )

    for model in models:

        is_active = (
            "✓"
            if model.name == active_name
            else ""
        )

        table.add_row(
            model.name,
            f"{model.size / (1024 * 1024):.1f}",
            model.extension,
            is_active,
        )

    console.print(table)


# =========================================================
# ACTIVE MODEL
# =========================================================

@models_app.command("active")
def active_model() -> None:
    """
    Show the currently active model.
    """

    manager = ModelManager()

    model = manager.active_model()

    if model is None:
        console.print(
            "[yellow]No active model selected.[/yellow]"
        )
        raise typer.Exit()

    console.print(
        f"[green]Active Model:[/green] "
        f"{model.name}"
    )


# =========================================================
# USE MODEL
# =========================================================

@models_app.command("use")
def use_model(
    model_name: str = typer.Argument(
        ...,
        help="GGUF model filename to activate.",
    ),
) -> None:
    """
    Set the active GGUF model.
    """

    manager = ModelManager()

    try:
        manager.set_current_model(model_name)

    except (ValueError, FileNotFoundError) as exc:
        console.print(
            f"[red]✗ {exc}[/red]"
        )

        raise typer.Exit(code=1)

    console.print(
        f"[green]✓ Active model changed to:[/green] "
        f"{model_name}"
    )


# =========================================================
# MODEL INFO
# =========================================================

@models_app.command("info")
def model_info() -> None:
    """
    Display model manager summary.
    """

    manager = ModelManager()

    summary = manager.summary()

    table = Table(
        title="QAIR Model Summary",
    )

    table.add_column(
        "Property",
        style="cyan",
    )

    table.add_column(
        "Value",
    )

    for key, value in summary.items():

        table.add_row(
            str(key),
            str(value),
        )

    console.print(table)


# =========================================================
# REFRESH MODELS
# =========================================================

@models_app.command("refresh")
def refresh() -> None:
    """
    Rediscover installed GGUF models.
    """

    manager = ModelManager()

    models = manager.refresh()

    console.print(
        f"[green]✓[/green] Discovered "
        f"{len(models)} model(s)."
    )


# =========================================================
# VERSION
# =========================================================

@app.command()
def version() -> None:
    """
    Show QAIR version.
    """

    console.print(
        "[bold cyan]QAIR v0.5.0[/bold cyan]"
    )


# =========================================================
# ENTRY POINT
# =========================================================

def run() -> None:
    """
    Console entry point for the `qair` executable.
    """

    app()


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":
    run()
