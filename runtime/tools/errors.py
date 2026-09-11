from __future__ import annotations


class ToolExecutionTimeoutError(TimeoutError):
    """Raised when a tool execution exceeds its configured timeout."""
