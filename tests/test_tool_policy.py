import pytest

from runtime.tools.policy import (
    AllowAllToolPolicy,
    AllowListToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
)
from runtime.tools.protocol import ToolCall


def test_policy_decision_allows_execution():
    decision = ToolPolicyDecision(allowed=True)

    assert decision.allowed is True
    assert decision.reason is None


def test_policy_decision_can_deny_execution():
    decision = ToolPolicyDecision(
        allowed=False,
        reason="Denied.",
    )

    assert decision.allowed is False
    assert decision.reason == "Denied."


def test_tool_policy_is_abstract():
    with pytest.raises(TypeError):
        ToolPolicy()


def test_allow_all_policy_allows_any_tool():
    policy = AllowAllToolPolicy()

    decision = policy.evaluate(
        ToolCall(
            name="anything",
            arguments={},
        )
    )

    assert decision.allowed is True


def test_allow_list_policy_allows_registered_name():
    policy = AllowListToolPolicy(
        {"echo", "weather"}
    )

    decision = policy.evaluate(
        ToolCall(
            name="echo",
            arguments={},
        )
    )

    assert decision.allowed is True


def test_allow_list_policy_denies_unapproved_name():
    policy = AllowListToolPolicy({"echo"})

    decision = policy.evaluate(
        ToolCall(
            name="shell",
            arguments={},
        )
    )

    assert decision.allowed is False
    assert decision.reason == (
        "Tool execution denied by policy: shell"
    )


def test_allow_list_policy_requires_set():
    with pytest.raises(
        TypeError,
        match="allowed_tools must be a set",
    ):
        AllowListToolPolicy(["echo"])


def test_allow_list_policy_requires_string_names():
    with pytest.raises(
        TypeError,
        match="allowed_tools must contain only strings",
    ):
        AllowListToolPolicy({"echo", 123})
