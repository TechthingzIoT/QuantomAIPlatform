import pytest

from runtime.prompts.manager import PromptManager


@pytest.fixture
def prompt_manager():
    return PromptManager()


def test_prompt_directory_exists(prompt_manager):
    assert prompt_manager.prompt_dir.exists()
    assert prompt_manager.prompt_dir.is_dir()


def test_list_prompts(prompt_manager):
    prompts = prompt_manager.list_prompts()

    assert isinstance(prompts, list)
    assert prompts == sorted(prompts)

    assert "assistant" in prompts
    assert "embedded" in prompts
    assert "robotics" in prompts
    assert "agriculture" in prompts
    assert "coding" in prompts


def test_load_existing_prompt(prompt_manager):
    prompt = prompt_manager.load("embedded")

    assert isinstance(prompt, str)
    assert prompt.strip() != ""


def test_load_prompt_contains_expected_content(prompt_manager):
    prompt = prompt_manager.load("embedded")

    assert len(prompt) > 20


def test_load_missing_prompt(prompt_manager):
    with pytest.raises(FileNotFoundError, match="Prompt 'does_not_exist' not found"):
        prompt_manager.load("does_not_exist")


def test_list_prompts_only_returns_txt_files(prompt_manager):
    prompts = prompt_manager.list_prompts()

    for name in prompts:
        assert (prompt_manager.prompt_dir / f"{name}.txt").exists()

def test_prompt_exists(prompt_manager):
    assert prompt_manager.exists("embedded") is True


def test_prompt_does_not_exist(prompt_manager):
    assert prompt_manager.exists("does_not_exist") is False


def test_get_prompt(prompt_manager):
    prompt = prompt_manager.get("embedded")

    assert isinstance(prompt, str)
    assert prompt.strip() != ""
