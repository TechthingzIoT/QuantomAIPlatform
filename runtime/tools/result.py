from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Structured result of a QAIR tool execution."""

    ok: bool
    result: Any = None
    error_type: str | None = None
    error_message: str | None = None

    @classmethod
    def success(cls, result: Any) -> "ToolExecutionResult":
        """Create a successful tool execution result."""

        return cls(
            ok=True,
            result=result,
        )

    @classmethod
    def failure(
        cls,
        error: Exception,
    ) -> "ToolExecutionResult":
        """Create a failed tool execution result."""

        return cls(
            ok=False,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        if self.ok:
            return {
                "ok": True,
                "result": self.result,
            }

        return {
            "ok": False,
            "error": {
                "type": self.error_type,
                "message": self.error_message,
            },
        }
