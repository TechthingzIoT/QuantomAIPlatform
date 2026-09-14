import pytest

from runtime.context.execution import ExecutionContext


def test_execution_context_requires_run_id():

    context = ExecutionContext(
        run_id="run-1",
    )

    assert context.run_id == "run-1"


def test_execution_context_accepts_agent_name():

    context = ExecutionContext(
        run_id="run-1",
        agent_name="jafar",
    )

    assert context.agent_name == "jafar"


def test_execution_context_accepts_iteration():

    context = ExecutionContext(
        run_id="run-1",
        iteration=3,
    )

    assert context.iteration == 3


def test_execution_context_copies_metadata():

    metadata = {
        "device": "robot-01",
    }

    context = ExecutionContext(
        run_id="run-1",
        metadata=metadata,
    )

    metadata["device"] = "robot-02"

    assert context.metadata["device"] == "robot-01"


def test_execution_context_rejects_non_string_run_id():

    with pytest.raises(
        TypeError,
        match="run_id must be a string",
    ):

        ExecutionContext(
            run_id=123,
        )


def test_execution_context_rejects_empty_run_id():

    with pytest.raises(
        ValueError,
        match="run_id cannot be empty",
    ):

        ExecutionContext(
            run_id="   ",
        )


def test_execution_context_rejects_non_string_agent_name():

    with pytest.raises(
        TypeError,
        match="agent_name must be a string or None",
    ):

        ExecutionContext(
            run_id="run-1",
            agent_name=123,
        )


def test_execution_context_rejects_empty_agent_name():

    with pytest.raises(
        ValueError,
        match="agent_name cannot be empty",
    ):

        ExecutionContext(
            run_id="run-1",
            agent_name="   ",
        )


def test_execution_context_rejects_negative_iteration():

    with pytest.raises(
        ValueError,
        match="iteration cannot be negative",
    ):

        ExecutionContext(
            run_id="run-1",
            iteration=-1,
        )


def test_execution_context_rejects_non_integer_iteration():

    with pytest.raises(
        TypeError,
        match="iteration must be an integer or None",
    ):

        ExecutionContext(
            run_id="run-1",
            iteration="one",
        )


def test_execution_context_rejects_non_dictionary_metadata():

    with pytest.raises(
        TypeError,
        match="metadata must be a dictionary",
    ):

        ExecutionContext(
            run_id="run-1",
            metadata=[],
        )
