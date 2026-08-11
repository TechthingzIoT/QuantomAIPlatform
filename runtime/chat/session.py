"""
QAIR Chat Session

Interactive runtime controller for QAIR.

Responsibilities:
- Initialize the runtime
- Load the active GGUF model
- Maintain conversation history
- Coordinate inference
- Provide an interactive REPL
- Gracefully shutdown
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from runtime.chat.history import ConversationHistory
from runtime.chat.message import ChatMessage, MessageRole
from runtime.inference.engine import InferenceEngine


class ChatSession:
    """
    Interactive QAIR runtime.

    Coordinates the inference engine, conversation history,
    and terminal interface.
    """

    def __init__(self):
        self.console = Console()
        self.engine = InferenceEngine()
        self.history = ConversationHistory()

        # --------------------------------------------------
        # QAIR system identity
        # --------------------------------------------------

        self.system_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are QAIR, the Quantom AI Runtime assistant "
                "developed by TIOTAIROBOTIX. "
                "You are a local AI assistant running through QAIR. "
                "Answer clearly, accurately, and concisely."
            ),
        )

        # System identity is always the first message.
        self.history.add(self.system_message)

        self.running = False

    # ==================================================
    # Banner
    # ==================================================

    def banner(self) -> None:
        """Display the QAIR startup banner."""

        self.console.print()

        self.console.print(
            Panel.fit(
                "[bold cyan]QAIR v0.5.0[/bold cyan]\n"
                "Quantom AI Runtime\n\n"
                "[green]Powered by TIOTAIROBOTIX[/green]",
                border_style="cyan",
            )
        )

        self.console.print()

    # ==================================================
    # Startup / Shutdown
    # ==================================================

    def startup(self) -> None:
        """Initialize the runtime."""

        self.banner()

        self.console.print(
            "[cyan]Loading active model...[/cyan]"
        )

        self.engine.load()

        model_name = (
            self.engine.model.name
            if self.engine.model
            else "Unknown"
        )

        self.console.print(
            f"[green]✓ Loaded:[/green] {model_name}"
        )

        self.console.print()

        self.console.print(
            "[dim]Type '/help' for commands.[/dim]"
        )

        self.console.print(
            "[dim]Type 'exit' or 'quit' to leave QAIR.[/dim]"
        )

        self.console.print()

        self.running = True

    def shutdown(self) -> None:
        """Shutdown QAIR cleanly."""

        self.console.print()

        self.console.print(
            "[yellow]Shutting down QAIR...[/yellow]"
        )

        self.engine.unload()

        self.running = False

        self.console.print(
            "[green]Session closed.[/green]"
        )

    # ==================================================
    # Conversation
    # ==================================================

    def ask(self, prompt: str) -> str:
        """
        Send a user message to the inference engine.

        The complete structured conversation history,
        including the QAIR system identity, is passed
        to llama.cpp.
        """

        user_message = ChatMessage(
            role=MessageRole.USER,
            content=prompt,
        )

        self.history.add(user_message)

        messages = self.history.to_messages()

        reply = self.engine.generate(messages)

        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=reply,
        )

        self.history.add(assistant_message)

        return reply

    # ==================================================
    # Utilities
    # ==================================================

    def clear(self) -> None:
        """
        Clear conversation history while preserving
        the QAIR system identity.
        """

        self.history.clear()

        self.history.add(self.system_message)

        self.console.print(
            "[green]Conversation cleared.[/green]"
        )

    def stats(self) -> None:
        """Display runtime statistics."""

        summary = self.engine.summary()

        self.console.print()

        self.console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"Model       : {summary['model']}",
                        f"Loaded      : {summary['loaded']}",
                        f"Context     : {summary['context']}",
                        f"GPU Layers  : {summary['gpu_layers']}",
                        f"Temperature : {summary['temperature']}",
                        f"Top P       : {summary['top_p']}",
                        f"Messages    : {len(self.history)}",
                    ]
                ),
                title="QAIR Runtime",
                border_style="cyan",
            )
        )

    def show_history(self) -> None:
        """Display conversation history."""

        if self.history.empty():
            self.console.print(
                "[yellow]History is empty.[/yellow]"
            )
            return

        self.console.print()

        for message in self.history:
            self.console.print(str(message))

        self.console.print()

    # ==================================================
    # Runtime Commands
    # ==================================================

    def command(self, text: str) -> bool:
        """
        Execute built-in commands.

        Returns True if the command was handled.
        """

        cmd = text.strip().lower()

        if cmd in {"exit", "quit"}:
            self.running = False
            return True

        if cmd == "/clear":
            self.clear()
            return True

        if cmd == "/history":
            self.show_history()
            return True

        if cmd == "/stats":
            self.stats()
            return True

        if cmd == "/help":
            self.console.print()

            self.console.print(
                Panel.fit(
                    "\n".join(
                        [
                            "/help      Show commands",
                            "/history   Show conversation",
                            "/stats     Runtime information",
                            "/clear     Clear conversation",
                            "exit       Quit QAIR",
                        ]
                    ),
                    title="QAIR Commands",
                    border_style="green",
                )
            )

            return True

        return False

    # ==================================================
    # Interactive Runtime
    # ==================================================

    def run(self) -> None:
        """Launch the interactive QAIR runtime."""

        self.startup()

        try:
            while self.running:

                prompt = Prompt.ask(
                    "[bold cyan]You[/bold cyan]"
                ).strip()

                if not prompt:
                    continue

                if self.command(prompt):
                    continue

                response = self.ask(prompt)

                self.console.print()

                self.console.print(
                    Panel(
                        response,
                        title="QAIR",
                        border_style="green",
                    )
                )

                self.console.print()

        except KeyboardInterrupt:

            self.console.print()

            self.console.print(
                "[yellow]Interrupted by user.[/yellow]"
            )

        finally:
            self.shutdown()


if __name__ == "__main__":
    ChatSession().run()