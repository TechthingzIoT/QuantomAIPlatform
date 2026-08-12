import pytest

from runtime.prompts.selection import PromptSelector


@pytest.fixture
def selector():
    return PromptSelector()


def test_default_prompt(selector):
    prompt = selector.default()

    assert isinstance(prompt, str)
    assert prompt.strip() != ""
    assert "QAIR" in prompt


def test_select_embedded_prompt(selector):
    prompt = selector.select("embedded")

    assert isinstance(prompt, str)
    assert prompt.strip() != ""
    assert "Embedded Systems Engineer" in prompt


def test_select_robotics_prompt(selector):
    prompt = selector.select("robotics")

    assert isinstance(prompt, str)
    assert prompt.strip() != ""


def test_select_agriculture_prompt(selector):
    prompt = selector.select("agriculture")

    assert isinstance(prompt, str)
    assert prompt.strip() != ""


def test_select_coding_prompt(selector):
    prompt = selector.select("coding")

    assert isinstance(prompt, str)
    assert prompt.strip() != ""


def test_selector_exists(selector):
    assert selector.exists("assistant") is True
    assert selector.exists("embedded") is True
    assert selector.exists("robotics") is True
    assert selector.exists("agriculture") is True
    assert selector.exists("coding") is True


def test_selector_missing_prompt(selector):
    assert selector.exists("does_not_exist") is False

    with pytest.raises(FileNotFoundError):
        selector.select("does_not_exist")


def test_available_prompts(selector):
    prompts = selector.available()

    assert isinstance(prompts, list)

    expected = {
        "assistant",
        "embedded",
        "robotics",
        "agriculture",
        "coding",
    }

    assert expected.issubset(set(prompts))


def test_default_matches_assistant(selector):
    assert selector.default() == selector.select("assistant")
