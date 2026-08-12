"""
QAIR Prompt Selection

Provides controlled selection of system prompts for
different QAIR operating domains.
"""

from runtime.prompts.manager import PromptManager


class PromptSelector:
    """
    Selects and loads QAIR system prompts.
    """

    DEFAULT_PROMPT = "assistant"

    DOMAIN_PROMPTS = {
        "embedded",
        "robotics",
        "agriculture",
        "coding",
    }

    def __init__(self, manager: PromptManager | None = None):
        self.manager = manager or PromptManager()

    def default(self) -> str:
        """
        Return the default QAIR assistant prompt.
        """
        return self.manager.get(self.DEFAULT_PROMPT)

    def select(self, name: str) -> str:
        """
        Load a prompt by name.

        Raises
        ------
        FileNotFoundError
            If the requested prompt does not exist.
        """
        return self.manager.get(name)

    def exists(self, name: str) -> bool:
        """
        Check whether a prompt exists.
        """
        return self.manager.exists(name)

    def available(self) -> list[str]:
        """
        Return all available prompts.
        """
        return self.manager.list_prompts()