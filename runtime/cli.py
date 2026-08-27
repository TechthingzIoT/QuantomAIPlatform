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
- Launch the QAIR HTTP API server
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from runtime.knowledge.registry import (
    add_source,
    clear_sources,
    list_sources,
    remove_source,
)
from runtime.models.manager import ModelManager

# =========================================================
# CONSTANTS
# =========================================================

QAIR_VERSION = "0.6.0"


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
knowledge_app = typer.Typer(
    name="knowledge",
    help="Manage local QAIR knowledge sources",
)

app.add_typer(
    knowledge_app,
    name="knowledge",
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
    if ctx.invoked_subcommand is not None:
        return

    from runtime.chat.session import ChatSession

    try:
        ChatSession(prompt_name=prompt).run()
    except ValueError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        console.print(f"[red]✗ QAIR runtime error:[/red] {exc}")
        raise typer.Exit(code=1)


# =========================================================
# SERVER
# =========================================================


@app.command("serve")
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host interface to bind.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        min=1,
        max=65535,
        help="TCP port to bind.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable Uvicorn auto-reload for development.",
    ),
) -> None:
    """
    Launch the QAIR HTTP API server.
    """
    import uvicorn

    uvicorn.run(
        "runtime.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


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
        console.print("[yellow]No models discovered.[/yellow]")
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
    table.add_column("Extension")
    table.add_column(
        "Active",
        justify="center",
    )

    active = manager.active_model()
    active_name = active.name if active is not None else None

    for model in models:
        is_active = "✓" if model.name == active_name else ""

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
        console.print("[yellow]No active model selected.[/yellow]")
        raise typer.Exit()

    console.print(f"[green]Active Model:[/green] " f"{model.name}")


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
        console.print(f"[red]✗ {exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]✓ Active model changed to:[/green] " f"{model_name}")


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
    table.add_column("Value")

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

    console.print(f"[green]✓[/green] Discovered " f"{len(models)} model(s).")


# =========================================================
# KNOWLEDGE
# =========================================================


@knowledge_app.command("sources")
def knowledge_sources() -> None:
    """
    List registered local knowledge sources.
    """

    sources = list_sources()

    if not sources:
        console.print("[yellow]No knowledge sources registered.[/yellow]")
        return

    console.print("Registered Knowledge Sources")

    table = Table(
        show_header=True,
    )

    table.add_column(
        "Source",
        style="cyan",
    )

    for source in sources:
        table.add_row(source)

    console.print(table)


@knowledge_app.command("add")
def knowledge_add(
    path: str = typer.Argument(
        ...,
        help="Local directory containing Markdown knowledge files.",
    ),
) -> None:
    """
    Register a local knowledge source directory.
    """

    source = Path(path).expanduser().resolve()

    if not source.exists():
        console.print(f"[red]✗ Knowledge directory not found:[/red] {source}")
        raise typer.Exit(code=1)

    if not source.is_dir():
        console.print(f"[red]✗ Knowledge path is not a directory:[/red] {source}")
        raise typer.Exit(code=1)

    add_source(source)

    console.print(f"[green]✓ Knowledge source registered:[/green] {source}")


@knowledge_app.command("remove")
def knowledge_remove(
    path: str = typer.Argument(
        ...,
        help="Knowledge source directory to unregister.",
    ),
) -> None:
    """
    Remove a registered local knowledge source.
    """

    source = Path(path).expanduser().resolve()

    remove_source(source)

    console.print(f"[green]✓ Knowledge source removed:[/green] {source}")


@knowledge_app.command("clear")
def knowledge_clear() -> None:
    """
    Remove all registered knowledge sources.
    """

    clear_sources()

    console.print("[green]✓ All knowledge sources cleared.[/green]")


# =========================================================
# VERSION
# =========================================================


@app.command()
def version() -> None:
    """
    Show QAIR version.
    """
    console.print(f"[bold cyan]QAIR v{QAIR_VERSION}[/bold cyan]")


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
