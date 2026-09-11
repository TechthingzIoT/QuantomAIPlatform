from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from runtime.tools.config import ToolExecutionConfig
from runtime.tools.context import ToolExecutionContext
from runtime.tools.errors import ToolExecutionTimeoutError
from runtime.tools.event import ToolExecutionEvent
from runtime.tools.outcome import ToolExecutionOutcome
from runtime.tools.policy import (
    AllowAllToolPolicy,
    ToolPolicy,
)
from runtime.tools.protocol import ToolCall
from runtime.tools.registry import ToolRegistry
from runtime.tools.result import ToolExecutionResult
from runtime.tools.validation import validate_tool_arguments


class ToolExecutor:
    """Execute registered QAIR tools safely."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        policy: ToolPolicy | None = None,
        config: ToolExecutionConfig | None = None,
    ) -> None:
        self.registry = registry
        self.policy = (
            policy
            if policy is not None
            else AllowAllToolPolicy()
        )
        self.config = (
            config
            if config is not None
            else ToolExecutionConfig()
        )

    def execute(
        self,
        tool_call: ToolCall,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolExecutionResult:
        """Execute a tool and return its structured result."""
        return self.execute_with_outcome(
            tool_call,
            context=context,
        ).result

    def execute_with_outcome(
        self,
        tool_call: ToolCall,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolExecutionOutcome:
        """
        Execute a tool and return both its result and
        operational execution event.
        """
        started_at = time.perf_counter()

        try:
            decision = self.policy.evaluate(
                tool_call,
                context=context,
            )

            if not decision.allowed:
                raise PermissionError(
                    decision.reason
                    or (
                        "Tool execution denied by policy: "
                        f"{tool_call.name}"
                    )
                )

            tool = self.registry.get(tool_call.name)

            if tool is None:
                raise ValueError(
                    f"Unknown tool: {tool_call.name}"
                )

            validate_tool_arguments(
                tool,
                tool_call.arguments,
            )

            result = self._execute_with_retries(
                tool,
                tool_call.arguments,
                context=context,
            )

            execution_result = ToolExecutionResult.success(
                result
            )

        except Exception as exc:  # noqa: BLE001
            execution_result = ToolExecutionResult.failure(
                exc
            )

        elapsed_ms = (
            time.perf_counter() - started_at
        ) * 1000

        metadata = (
            context.metadata
            if context is not None
            else {}
        )

        event = ToolExecutionEvent(
            tool_name=tool_call.name,
            elapsed_ms=elapsed_ms,
            ok=execution_result.ok,
            tool_call_id=(
                context.tool_call_id
                if context is not None
                else None
            ),
            agent_name=(
                context.agent_name
                if context is not None
                else None
            ),
            iteration=metadata.get(
                "iteration"
            ),
            error_type=execution_result.error_type,
            error_message=execution_result.error_message,
        )

        return ToolExecutionOutcome(
            result=execution_result,
            event=event,
        )

    def _execute_with_retries(
        self,
        tool: object,
        arguments: dict,
        *,
        context: ToolExecutionContext | None = None,
    ) -> object:
        """
        Execute a tool and retry execution failures when the
        tool explicitly supports retries.
        """
        max_retries = self.config.max_retries
        retry_delay_seconds = (
            self.config.retry_delay_seconds
        )

        supports_retry = getattr(
            tool,
            "supports_retry",
            False,
        )

        attempts = 1

        if supports_retry:
            attempts += max_retries

        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                return self._execute_tool(
                    tool,
                    arguments,
                    context=context,
                )

            except Exception as exc:
                last_error = exc

                if attempt == attempts - 1:
                    raise

                if retry_delay_seconds > 0:
                    time.sleep(retry_delay_seconds)

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "Tool execution failed without an error."
        )

    def _execute_tool(
        self,
        tool: object,
        arguments: dict,
        *,
        context: ToolExecutionContext | None = None,
    ) -> object:
        """Execute a tool, applying the configured timeout."""
        timeout_seconds = self.config.timeout_seconds

        if timeout_seconds is None:
            return self._invoke_tool(
                tool,
                arguments,
                context=context,
            )

        executor = ThreadPoolExecutor(
            max_workers=1
        )

        future = executor.submit(
            self._invoke_tool,
            tool,
            arguments,
            context=context,
        )

        try:
            return future.result(
                timeout=timeout_seconds
            )

        except FutureTimeoutError as exc:
            future.cancel()

            raise ToolExecutionTimeoutError(
                "Tool execution exceeded "
                f"{timeout_seconds} seconds."
            ) from exc

        finally:
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

    @staticmethod
    def _invoke_tool(
        tool: object,
        arguments: dict,
        *,
        context: ToolExecutionContext | None = None,
    ) -> object:
        """Invoke a tool while preserving legacy tool compatibility."""

        supports_context = getattr(
            tool,
            "supports_context",
            False,
        )

        if context is not None and supports_context:
            return tool.execute(
                arguments,
                context=context,
            )

        return tool.execute(arguments)
