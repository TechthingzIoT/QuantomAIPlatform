from dataclasses import FrozenInstanceError

import pytest

from runtime.tools.protocol import ToolCall


def test_tool_call_stores_name_and_arguments():
    call = ToolCall(
        name="echo",
        arguments={"message": "hello"},
    )

    assert call.name == "echo"
    assert call.arguments == {"message": "hello"}


def test_tool_call_is_immutable():
    call = ToolCall(
        name="echo",
        arguments={"message": "hello"},
    )

    with pytest.raises(FrozenInstanceError):
        call.name = "other"


def test_tool_call_rejects_non_string_name():
    with pytest.raises(TypeError, match="name must be a string"):
        ToolCall(
            name=123,
            arguments={},
        )


def test_tool_call_rejects_empty_name():
    with pytest.raises(ValueError, match="name cannot be empty"):
        ToolCall(
            name="   ",
            arguments={},
        )


def test_tool_call_rejects_non_dictionary_arguments():
    with pytest.raises(TypeError, match="arguments must be a dictionary"):
        ToolCall(
            name="echo",
            arguments="hello",
        )


def test_tool_call_copies_arguments():
    arguments = {"message": "hello"}

    call = ToolCall(
        name="echo",
        arguments=arguments,
    )

    arguments["message"] = "changed"

    assert call.arguments == {"message": "hello"}


def test_tool_call_is_exported_from_tools_package():
    from runtime.tools import ToolCall as ExportedToolCall

    assert ExportedToolCall is ToolCall


def test_tool_protocol_exports_only_tool_call():
    from runtime.tools import __all__

    assert __all__ == ["ToolCall"]


def test_parse_tool_call_from_dictionary():
    from runtime.tools.protocol import parse_tool_call

    call = parse_tool_call(
        {
            "name": "echo",
            "arguments": {"message": "hello"},
        }
    )

    assert isinstance(call, ToolCall)
    assert call.name == "echo"
    assert call.arguments == {"message": "hello"}


def test_parse_tool_call_rejects_non_dictionary_payload():
    from runtime.tools.protocol import parse_tool_call

    with pytest.raises(TypeError, match="Tool call payload must be a dictionary"):
        parse_tool_call("not a tool call")


def test_parse_tool_call_rejects_missing_name():
    from runtime.tools.protocol import parse_tool_call

    with pytest.raises(
        ValueError, match="Tool call payload missing required field: 'name'"
    ):
        parse_tool_call({"arguments": {}})


def test_parse_tool_call_rejects_missing_arguments():
    from runtime.tools.protocol import parse_tool_call

    with pytest.raises(
        ValueError,
        match="Tool call payload missing required field: 'arguments'",
    ):
        parse_tool_call({"name": "echo"})


def test_parse_tool_call_rejects_extra_fields():
    from runtime.tools.protocol import parse_tool_call

    with pytest.raises(ValueError, match="unexpected field"):
        parse_tool_call(
            {
                "name": "echo",
                "arguments": {},
                "extra": "not allowed",
            }
        )


def test_parse_tool_call_from_json_string():
    from runtime.tools.protocol import parse_tool_call

    call = parse_tool_call('{"name": "echo", "arguments": {"message": "hello"}}')

    assert isinstance(call, ToolCall)
    assert call.name == "echo"
    assert call.arguments == {"message": "hello"}


def test_parse_tool_call_rejects_malformed_json():
    from runtime.tools.protocol import parse_tool_call

    with pytest.raises(ValueError, match="Invalid tool call JSON"):
        parse_tool_call('{"name": "echo", "arguments": ')


def test_parse_tool_call_rejects_json_non_dictionary():
    from runtime.tools.protocol import parse_tool_call

    with pytest.raises(TypeError, match="Tool call payload must be a dictionary"):
        parse_tool_call('["echo", {"message": "hello"}]')


def test_parse_tool_call_rejects_json_missing_fields():
    from runtime.tools.protocol import parse_tool_call

    with pytest.raises(
        ValueError,
        match="Tool call payload missing required field: 'arguments'",
    ):
        parse_tool_call('{"name": "echo"}')


@pytest.mark.parametrize("payload", ["null", "true", "42"])
def test_parse_tool_call_rejects_json_primitive(payload):
    from runtime.tools.protocol import parse_tool_call

    with pytest.raises(TypeError, match="Tool call payload must be a dictionary"):
        parse_tool_call(payload)
