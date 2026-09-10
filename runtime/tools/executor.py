from __future__ import annotations

import time

from typing import Any

from runtime.tools.protocol import ToolCall
from runtime.tools.registry import ToolRegistry
from runtime.tools.result import ToolExecutionResult
from runtime.tools.validation import validate_tool_arguments


class ToolExecutor:
    """Execute registered QAIR tools safely."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, tool_call: ToolCall) -> ToolExecutionResult:
        """Execute a validated tool call and return a structured result."""

        started_at = time.perf_counter()

        try:
            tool = self.registry.get(tool_call.name)

            if tool is None:
                raise ValueError(
                    f"Unknown tool: {tool_call.name}"
                )

            validate_tool_arguments(
                tool,
                tool_call.arguments,
            )

            result = tool.execute(tool_call.arguments)

            return ToolExecutionResult.success(result)

        except Exception as exc:
            return ToolExecutionResult.failure(exc)

        finally:
            elapsed_ms = (
                time.perf_counter() - started_at
            ) * 1000

            # Reserved for future execution telemetry.
            _ = elapsed_ms
