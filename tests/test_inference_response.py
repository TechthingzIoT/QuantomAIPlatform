import pytest

from runtime.inference.response import (
    InferenceResponse,
    ToolCallRequest,
)


def test_tool_call_request_accepts_valid_values():

    request = ToolCallRequest(
        id="call-1",
        name="echo",
        arguments={"message": "hello"},
    )

    assert request.id == "call-1"
    assert request.name == "echo"
    assert request.arguments == {"message": "hello"}


def test_tool_call_request_normalizes_id_and_name():

    request = ToolCallRequest(
        id="  call-1  ",
        name="  echo  ",
        arguments={},
    )

    assert request.id == "call-1"
    assert request.name == "echo"


def test_tool_call_request_rejects_non_string_id():

    with pytest.raises(TypeError, match="id must be a string"):

        ToolCallRequest(
            id=123,
            name="echo",
            arguments={},
        )


def test_tool_call_request_rejects_empty_id():

    with pytest.raises(ValueError, match="id cannot be empty"):

        ToolCallRequest(
            id="   ",
            name="echo",
            arguments={},
        )


def test_tool_call_request_rejects_non_string_name():

    with pytest.raises(TypeError, match="name must be a string"):

        ToolCallRequest(
            id="call-1",
            name=123,
            arguments={},
        )


def test_tool_call_request_rejects_empty_name():

    with pytest.raises(ValueError, match="name cannot be empty"):

        ToolCallRequest(
            id="call-1",
            name="   ",
            arguments={},
        )


def test_tool_call_request_rejects_non_dictionary_arguments():

    with pytest.raises(TypeError, match="arguments must be a dictionary"):

        ToolCallRequest(
            id="call-1",
            name="echo",
            arguments=["hello"],
        )


def test_tool_call_request_copies_arguments():

    arguments = {"message": "hello"}

    request = ToolCallRequest(
        id="call-1",
        name="echo",
        arguments=arguments,
    )

    arguments["message"] = "changed"

    assert request.arguments == {"message": "hello"}


def test_inference_response_accepts_content():

    response = InferenceResponse(
        content="Hello",
    )

    assert response.content == "Hello"
    assert response.tool_calls == []


def test_inference_response_accepts_tool_calls():

    tool_call = ToolCallRequest(
        id="call-1",
        name="echo",
        arguments={"message": "hello"},
    )

    response = InferenceResponse(
        tool_calls=[tool_call],
    )

    assert response.content is None
    assert response.tool_calls == [tool_call]


def test_inference_response_rejects_invalid_content():

    with pytest.raises(
        TypeError,
        match="content must be a string or None",
    ):

        InferenceResponse(
            content=123,
        )


def test_inference_response_rejects_non_list_tool_calls():

    with pytest.raises(
        TypeError,
        match="tool_calls must be a list",
    ):

        InferenceResponse(
            tool_calls="invalid",
        )


def test_inference_response_rejects_invalid_tool_call_items():

    with pytest.raises(
        TypeError,
        match="tool_calls must contain ToolCallRequest objects",
    ):

        InferenceResponse(
            tool_calls=["invalid"],
        )


def test_inference_response_copies_tool_calls():

    tool_call = ToolCallRequest(
        id="call-1",
        name="echo",
        arguments={},
    )

    tool_calls = [tool_call]

    response = InferenceResponse(
        tool_calls=tool_calls,
    )

    tool_calls.clear()

    assert response.tool_calls == [tool_call]
