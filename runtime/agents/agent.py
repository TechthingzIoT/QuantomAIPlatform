"""
QAIR Agent.

Provides a deterministic, single-step orchestration layer above
QAIRRuntime and ConversationHistory.

The Agent owns task execution and conversation state, while
QAIRRuntime remains responsible for inference and knowledge
augmentation.
"""

from __future__ import annotations

import json

from runtime.chat.history import ConversationHistory
from runtime.chat.message import ChatMessage, MessageRole
from runtime.core.runtime import QAIRRuntime
from runtime.inference.response import ToolCallRequest
from runtime.tools.protocol import parse_tool_call
from runtime.tools.registry import ToolRegistry
from runtime.tools.validation import validate_tool_arguments


class Agent:
    """Deterministic QAIR agent orchestrator."""

    DEFAULT_NAME = "qair-agent"

    MAX_TOOL_ITERATIONS = 8

    def __init__(
        self,
        *,
        runtime: QAIRRuntime | None = None,
        history: ConversationHistory | None = None,
        name: str = DEFAULT_NAME,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        name = name.strip()
        if not name:
            raise ValueError("name cannot be empty.")

        self.name = name
        self.runtime = runtime if runtime is not None else QAIRRuntime()
        self.history = history if history is not None else ConversationHistory()
        self.tool_registry = (
            tool_registry if tool_registry is not None else ToolRegistry()
        )
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

        content = response.content

        if content is None:
            raise RuntimeError(
                "Inference response did not contain assistant content."
            )

        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
        )

        self.history.add(assistant_message)

        return content

    def execute_tool(self, payload: object) -> object:
        """Parse, validate, and execute a registered tool call."""

        tool_call = parse_tool_call(payload)

        tool = self.tool_registry.get(tool_call.name)

        if tool is None:
            raise ValueError(f"Unknown tool: {tool_call.name}")

        validate_tool_arguments(tool, tool_call.arguments)

        return tool.execute(tool_call.arguments)

    def _record_tool_calls(
        self,
        tool_calls: list[ToolCallRequest],
    ) -> None:
        """Record assistant-requested tool calls in conversation history."""

        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=tool_calls,
        )

        self.history.add(assistant_message)

    def _execute_tool_call(
        self,
        tool_call: ToolCallRequest,
    ) -> object:
        """Execute a structured inference tool request."""

        payload = {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
        }

        return self.execute_tool(payload)

    def _record_tool_result(
        self,
        tool_call: ToolCallRequest,
        result: object,
    ) -> None:
        """Record a tool execution result in conversation history."""

        if isinstance(result, str):
            content = result
        else:
            try:
                content = json.dumps(result)
            except (TypeError, ValueError):
                content = str(result)

        tool_message = ChatMessage(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call.id,
        )

        self.history.add(tool_message)

    def run(self, prompt: str) -> str:
        """
        Execute a user task with bounded tool orchestration.

        The agent performs inference and executes requested tools
        until the model returns assistant content without additional
        tool calls.

        A bounded iteration limit prevents infinite tool loops.
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

        for _ in range(self.MAX_TOOL_ITERATIONS):
            messages = self.history.to_messages()
            tools = self.tool_registry.definitions() or None

            response = self.runtime.generate(
                messages,
                tools=tools,
                use_knowledge=True,
            )

            if response.tool_calls:
                self._record_tool_calls(response.tool_calls)

                for tool_call in response.tool_calls:
                    result = self._execute_tool_call(tool_call)
                    self._record_tool_result(tool_call, result)

                continue

            content = response.content

            if content is None:
                raise RuntimeError(
                    "Inference response did not contain assistant "
                    "content or tool calls."
                )

            assistant_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
            )

            self.history.add(assistant_message)

            return content

        raise RuntimeError(
            "Maximum tool execution iterations exceeded."
        )
