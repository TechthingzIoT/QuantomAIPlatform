"""
=========================================================
QAIR Chat Session

File:
    runtime/chat/session.py

Purpose:
    Maintains conversation history for QAIR.

Responsibilities:
    - Store conversation history
    - Build prompts for the inference engine
    - Preserve the system prompt
    - Trim old history
    - Provide session statistics

Author:
    TIOTAIROBOTIX
=========================================================
"""

from typing import Dict, List, Optional


class ChatSession:
    """
    QAIR conversation manager.

    This class owns the conversation state while the
    inference engine focuses only on generating responses.
    """

    def __init__(
        self,
        engine,
        system_prompt: Optional[str] = None,
        max_history: int = 20,
    ):

        self.engine = engine
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []

        if system_prompt:
            self.messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

    # --------------------------------------------------

    def ask(self, prompt: str) -> str:

        self.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        self._trim_history()

        conversation = self._build_prompt()

        reply = self.engine.chat(conversation)

        self.messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        return reply

    # --------------------------------------------------

    def _build_prompt(self) -> str:

        prompt = ""

        for msg in self.messages:

            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"

            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"

            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"

        prompt += "Assistant:"

        return prompt

    # --------------------------------------------------

    def _trim_history(self):

        if len(self.messages) <= self.max_history:
            return

        system = []

        if self.messages and self.messages[0]["role"] == "system":
            system = [self.messages[0]]

        history = self.messages[-(self.max_history - len(system)):]

        self.messages = system + history

    # --------------------------------------------------

    def clear(self):

        system = []

        if self.messages and self.messages[0]["role"] == "system":
            system = [self.messages[0]]

        self.messages = system

    # --------------------------------------------------

    def history(self):

        return self.messages

    # --------------------------------------------------

    def stats(self):

        users = sum(
            1 for m in self.messages
            if m["role"] == "user"
        )

        assistants = sum(
            1 for m in self.messages
            if m["role"] == "assistant"
        )

        return {
            "messages": len(self.messages),
            "user_messages": users,
            "assistant_messages": assistants,
            "max_history": self.max_history,
        }
