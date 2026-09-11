from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from runtime.tools.context import ToolExecutionContext
from runtime.tools.protocol import ToolCall


@dataclass(frozen=True, slots=True)
class ToolPolicyDecision:
    """Decision returned by a tool execution policy."""

    allowed: bool
    reason: str | None = None


class ToolPolicy(ABC):
    """Abstract policy for authorizing tool execution."""

    @abstractmethod
    def evaluate(
        self,
        tool_call: ToolCall,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolPolicyDecision:
        """Evaluate whether a tool call is allowed."""
        raise NotImplementedError


class AllowAllToolPolicy(ToolPolicy):
    """Policy that allows every tool execution."""

    def evaluate(
        self,
        tool_call: ToolCall,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolPolicyDecision:
        return ToolPolicyDecision(allowed=True)


class AllowListToolPolicy(ToolPolicy):
    """Policy that allows only explicitly approved tool names."""

    def __init__(self, allowed_tools: set[str]) -> None:
        if not isinstance(allowed_tools, set):
            raise TypeError("allowed_tools must be a set.")

        if not all(
            isinstance(name, str)
            for name in allowed_tools
        ):
            raise TypeError(
                "allowed_tools must contain only strings."
            )

        self._allowed_tools = frozenset(
            name.strip()
            for name in allowed_tools
            if name.strip()
        )

    def evaluate(
        self,
        tool_call: ToolCall,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolPolicyDecision:
        if tool_call.name in self._allowed_tools:
            return ToolPolicyDecision(allowed=True)

        return ToolPolicyDecision(
            allowed=False,
            reason=(
                f"Tool execution denied by policy: "
                f"{tool_call.name}"
            ),
        )
