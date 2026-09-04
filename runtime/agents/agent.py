"""
QAIR Agent.

Provides a deterministic, single-step orchestration layer above
QAIRRuntime and ConversationHistory.

The Agent owns task execution and conversation state, while
QAIRRuntime remains responsible for inference and knowledge
augmentation.
"""

from __future__ import annotations

from runtime.chat.history import ConversationHistory
from runtime.chat.message import ChatMessage, MessageRole
from runtime.core.runtime import QAIRRuntime


class Agent:
    """Deterministic QAIR agent orchestrator."""

    DEFAULT_NAME = "qair-agent"

    def __init__(
        self,
        *,
        runtime: QAIRRuntime | None = None,
        history: ConversationHistory | None = None,
        name: str = DEFAULT_NAME,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        name = name.strip()
        if not name:
            raise ValueError("name cannot be empty.")

        self.name = name
        self.runtime = runtime if runtime is not None else QAIRRuntime()
        self.history = history if history is not None else ConversationHistory()
        self.running = False

    # ==================================================
    # Lifecycle
    # ==================================================

    def start(self) -> None:
        """Start the agent and its underlying runtime."""
        if self.running:
            return
        self.runtime.start()
        self.running = True

    def stop(self) -> None:
        """Stop the agent and its underlying runtime."""
        if not self.running:
            return
        self.runtime.stop()
        self.running = False

    # Conversation
    # ==================================================

    def reset(self) -> None:
        """Clear the agent conversation history."""
        self.history.clear()

    # ==================================================
    # Execution
    # ==================================================

    def step(self, prompt: str) -> str:
        """
        Execute one deterministic agent step.

        A step records the user message, delegates inference to
        QAIRRuntime, and records the assistant response only when
        inference succeeds.
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        if not prompt.strip():
            raise ValueError("prompt cannot be empty.")

        if not self.running:
            self.start()

        user_message = ChatMessage(
            role=MessageRole.USER,
            content=prompt,
        )
        self.history.add(user_message)

        messages = self.history.to_messages()

        response = self.runtime.generate(
            messages,
            use_knowledge=True,
        )

        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response,
        )
        self.history.add(assistant_message)

        return response

    def run(self, prompt: str) -> str:
        """
        Execute a user task.

        M13 initially uses a single-step execution model. The
        public boundary is intentionally separated from ``step()``
        so future multi-step planning and tool execution can be
        introduced without changing callers.
        """
        return self.step(prompt)
