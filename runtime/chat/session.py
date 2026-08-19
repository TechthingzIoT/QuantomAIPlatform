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

# IMPORTANT:
# These imports intentionally live at module level.
# tests/test_session.py patches these symbols directly
# using runtime.chat.session.InferenceEngine, etc.
from runtime.inference.engine import InferenceEngine
from runtime.chat.history import ConversationHistory
from runtime.chat.message import ChatMessage, MessageRole
from runtime.prompts.selection import PromptSelector


class ChatSession:
    """
    Interactive QAIR runtime session.

    Coordinates:
    - InferenceEngine
    - ConversationHistory
    - PromptSelector
    - Terminal UI
    """

    VERSION = "0.5.0"

    def __init__(
        self,
        prompt_selector: PromptSelector | None = None,
        prompt_name: str | None = None,
    ) -> None:
        self.console = Console()

        # --------------------------------------------------
        # Runtime engine
        # --------------------------------------------------

        self.engine = InferenceEngine()

        # --------------------------------------------------
        # Conversation history
        # --------------------------------------------------

        self.history = ConversationHistory()

        # --------------------------------------------------
        # Prompt management
        # --------------------------------------------------

        self.prompt_selector = (
            prompt_selector
            if prompt_selector is not None
            else PromptSelector()
        )

        self.active_prompt = (
            prompt_name
            if prompt_name is not None
            else self.prompt_selector.DEFAULT_PROMPT
        )

        self.active_prompt = self.active_prompt.strip().lower()

        # Validate and load the selected system prompt.
        system_prompt = self.prompt_selector.select(
            self.active_prompt
        )

        self.system_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt,
        )

        # System message must always be first.
        self.history.add(self.system_message)

        # --------------------------------------------------
        # Lifecycle state
        # --------------------------------------------------

        self.running = False

    # ======================================================
    # Banner
    # ======================================================

    def banner(self) -> None:
        """Display the QAIR startup banner."""

        self.console.print()

        self.console.print(
            Panel.fit(
                f"[bold cyan]QAIR v{self.VERSION}[/bold cyan]\n"
                "Quantom AI Runtime\n\n"
                f"[yellow]Prompt:[/yellow] "
                f"{self.active_prompt}\n"
                "[green]Powered by TIOTAIROBOTIX[/green]",
                border_style="cyan",
            )
        )

        self.console.print()

    # ======================================================
    # Startup / Shutdown
    # ======================================================

    def startup(self) -> None:
        """
        Initialize the runtime.

        Startup is idempotent.
        """

        if self.running:
            return

        self.banner()

        self.console.print(
            "[cyan]Loading active model...[/cyan]"
        )

        try:
            self.engine.load()

            model_name = (
                self.engine.model.name
                if getattr(self.engine, "model", None)
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

        except Exception:
            self.running = False
            raise

    def shutdown(self) -> None:
        """
        Shutdown QAIR cleanly.

        Shutdown is idempotent.
        """

        if not self.running:
            return

        self.console.print()

        self.console.print(
            "[yellow]Shutting down QAIR...[/yellow]"
        )

        try:
            self.engine.unload()
        finally:
            self.running = False

        self.console.print(
            "[green]Session closed.[/green]"
        )

    # ======================================================
    # Prompt Management
    # ======================================================

    def set_prompt(self, name: str) -> None:
        """
        Switch the active system prompt.

        Switching prompts resets the conversation because
        previous messages may have been generated under
        different system instructions.
        """

        name = name.strip().lower()

        if not name:
            raise ValueError(
                "Prompt name cannot be empty."
            )

        if not self.prompt_selector.exists(name):
            available = ", ".join(
                self.prompt_selector.available()
            )

            raise ValueError(
                f"Unknown prompt '{name}'. "
                f"Available prompts: {available}"
            )

        # Select first so an invalid prompt cannot mutate
        # the current session state.
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

        if not available:
            self.console.print(
                "[yellow]No prompts available.[/yellow]"
            )
            self.console.print()
            return

        lines = []

        for name in available:
            marker = (
                "* "
                if name == self.active_prompt
                else "  "
            )

            lines.append(
                f"{marker}{name}"
            )

        self.console.print(
            Panel.fit(
                "\n".join(lines),
                title="QAIR Prompts",
                border_style="cyan",
            )
        )

        self.console.print()

    # ======================================================
    # Conversation
    # ======================================================

    def ask(self, prompt: str) -> str:
        """
        Send a user message to the inference engine.

        The complete structured conversation history,
        including the system message, is sent to the
        inference engine.
        """

        # Preserve the historical behavior expected by
        # the existing test suite: even an empty string is
        # recorded as a user message.
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

    # ======================================================
    # Utilities
    # ======================================================

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

        Returns:
            True  - input was handled as a QAIR command.
            False - input is not a QAIR command and should
                    be passed to the inference engine.
        """

        raw = text.strip()

        if not raw:
            return False

        cmd = raw.lower()
        # --------------------------------------------------
        # Exit
        # --------------------------------------------------

        if cmd in {
            "exit",
            "quit",
            "exit()",
            "quit()",
            "/exit",
            "/quit",
            "/exit()",
            "/quit()",
        }:
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

        # --------------------------------------------------
        # Unknown input
        # --------------------------------------------------

        return False

        # ======================================================
    # Interactive REPL
    # ======================================================

    def run(self) -> None:
        """
        Start and run the interactive QAIR chat session.

        The session:
        - Starts the inference engine
        - Accepts interactive user input
        - Handles built-in commands
        - Sends normal input to the model
        - Displays model responses
        - Shuts down cleanly on exit or interruption
        """

        try:
            self.startup()

            while self.running:
                try:
                    user_input = Prompt.ask(
                        "[bold cyan]You[/bold cyan]"
                    )

                except (EOFError, KeyboardInterrupt):
                    self.console.print()
                    self.console.print(
                        "[yellow]Exiting QAIR...[/yellow]"
                    )
                    break

                # Handle built-in QAIR commands.
                if self.command(user_input):
                    continue

                # Ignore completely empty interactive input.
                if not user_input.strip():
                    continue

                try:
                    reply = self.ask(user_input)

                    self.console.print(
                        Panel(
                            reply,
                            title="QAIR",
                            border_style="green",
                        )
                    )

                    self.console.print()

                except Exception as exc:
                    self.console.print(
                        f"[red]Inference error: {exc}[/red]"
                    )

        finally:
            self.shutdown()