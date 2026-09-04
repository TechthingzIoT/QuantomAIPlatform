from __future__ import annotations

from typing import Any

from runtime.tools.base import Tool


def validate_tool_arguments(
    tool: Tool,
    arguments: Any,
) -> None:
    """Validate tool arguments against the tool's input schema."""
    if not isinstance(arguments, dict):
        raise TypeError("Tool arguments must be a dictionary.")

    schema = tool.input_schema
    if not isinstance(schema, dict):
        raise TypeError("Tool input schema must be a dictionary.")
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for name in required:
        if name not in arguments:
            raise ValueError(f"Missing required argument: {name}")

    unexpected = set(arguments) - set(properties)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"unexpected argument(s): {names}")

    for name, value in arguments.items():
        expected_type = properties[name].get("type")

        if expected_type == "string" and not isinstance(value, str):
            raise TypeError(f"Argument {name} must be a string.")

        if expected_type == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise TypeError(f"Argument {name} must be an integer.")

        if expected_type == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise TypeError(f"Argument {name} must be a number.")

        if expected_type == "boolean" and not isinstance(value, bool):
            raise TypeError(f"Argument {name} must be a boolean.")

        if expected_type == "array" and not isinstance(value, list):
            raise TypeError(f"Argument {name} must be an array.")

        if expected_type == "object" and not isinstance(value, dict):
            raise TypeError(f"Argument {name} must be an object.")
