from runtime.tools.result import ToolExecutionResult


def test_successful_tool_execution_result():

    result = ToolExecutionResult.success(
        {
            "status": "operational",
            "temperature": 24.5,
        }
    )

    assert result.ok is True

    assert result.to_dict() == {
        "ok": True,
        "result": {
            "status": "operational",
            "temperature": 24.5,
        },
    }


def test_failed_tool_execution_result():

    error = ValueError("Missing required argument: location")

    result = ToolExecutionResult.failure(error)

    assert result.ok is False

    assert result.to_dict() == {
        "ok": False,
        "error": {
            "type": "ValueError",
            "message": "Missing required argument: location",
        },
    }
