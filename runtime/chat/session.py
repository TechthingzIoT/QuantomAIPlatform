"""
QAIR Chat Session

Interactive runtime controller for QAIR.

Responsibilities:
- Initialize the runtime
- Load the active GGUF model
- Maintain conversation history
- Manage system prompts
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
from runtime.prompts.selection import PromptSelector


class ChatSession:
    """
    Interactive QAIR runtime.

    Coordinates the inference engine, conversation history,
    prompt selection, and terminal interface.
    """

    def __init__(
        self,
        prompt_selector: PromptSelector | None = None,
        prompt_name: str | None = None,
    ):
        self.console = Console()

        self.engine = InferenceEngine()

        self.history = ConversationHistory()

        # --------------------------------------------------
        # Prompt selection
        # --------------------------------------------------

        self.prompt_selector = (
            prompt_selector or PromptSelector()
        )

        self.active_prompt = (
            prompt_name or self.prompt_selector.DEFAULT_PROMPT
        )

        # Validate and load selected prompt.
        system_prompt = self.prompt_selector.select(
            self.active_prompt
        )

        self.system_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt,
        )

        # System prompt is always the first message.
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
                f"[yellow]Prompt:[/yellow] {self.active_prompt}\n"
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
    # Prompt Management
    # ==================================================

    def set_prompt(self, name: str) -> None:
        """
        Switch the active system prompt.

        Switching prompts resets the conversation because
        previous messages may have been generated under a
        different domain/system instruction.
        """

        name = name.strip().lower()

        if not name:
            raise ValueError("Prompt name cannot be empty.")

        if not self.prompt_selector.exists(name):
            available = ", ".join(
                self.prompt_selector.available()
            )

            raise ValueError(
                f"Unknown prompt '{name}'. "
                f"Available prompts: {available}"
            )

        system_prompt = self.prompt_selector.select(name)

        self.active_prompt = name

        self.system_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt,
        )

        self.history.clear()

        self.history.add(self.system_message)

    def show_prompts(self) -> None:
        """Display available system prompts."""

        available = self.prompt_selector.available()

        self.console.print()

        self.console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"{'* ' if name == self.active_prompt else '  '}"
                        f"{name}"
                        for name in available
                    ]
                ),
                title="QAIR Prompts",
                border_style="cyan",
            )
        )

        self.console.print()

    # ==================================================
    # Conversation
    # ==================================================

    def ask(self, prompt: str) -> str:
        """
        Send a user message to the inference engine.

        The complete structured conversation history,
        including the active system prompt, is passed
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
        the active system prompt.
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
                        f"Prompt      : {self.active_prompt}",
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

        raw = text.strip()

        if not raw:
            return False

        cmd = raw.lower()

        # --------------------------------------------------
        # Exit
        # --------------------------------------------------

        if cmd in {"exit", "quit"}:
            self.running = False
            return True

        # --------------------------------------------------
        # Clear
        # --------------------------------------------------

        if cmd == "/clear":
            self.clear()
            return True

        # --------------------------------------------------
        # History
        # --------------------------------------------------

        if cmd == "/history":
            self.show_history()
            return True

        # --------------------------------------------------
        # Stats
        # --------------------------------------------------

        if cmd == "/stats":
            self.stats()
            return True

        # --------------------------------------------------
        # Prompt
        # --------------------------------------------------

        if cmd == "/prompt":
            self.show_prompts()
            return True

        if cmd.startswith("/prompt "):
            prompt_name = raw.split(maxsplit=1)[1]

            try:
                self.set_prompt(prompt_name)

                self.console.print(
                    f"[green]✓ Prompt switched to:[/green] "
                    f"{self.active_prompt}"
                )

            except ValueError as exc:
                self.console.print(
                    f"[red]✗ {exc}[/red]"
                )

            return True

        # --------------------------------------------------
        # Help
        # --------------------------------------------------

        if cmd == "/help":
            self.console.print()

            self.console.print(
                Panel.fit(
                    "\n".join(
                        [
                            "/help              Show commands",
                            "/prompt            List available prompts",
                            "/prompt <name>     Switch system prompt",
                            "/history           Show conversation",
                            "/stats             Runtime information",
                            "/clear             Clear conversation",
                            "exit               Quit QAIR",
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
