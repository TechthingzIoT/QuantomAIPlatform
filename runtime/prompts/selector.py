from runtime.prompts.manager import PromptManager


class PromptSelector:
    """
    Selects and loads QAIR system prompts by domain.
    """

    DEFAULT_PROMPT = "assistant"

    def __init__(self, manager: PromptManager | None = None):
        self.manager = manager or PromptManager()

    def select(self, name: str | None = None) -> str:
        """
        Return the prompt for the requested domain.

        If no domain is specified, the default assistant
        prompt is returned.
        """

        prompt_name = name or self.DEFAULT_PROMPT

        return self.manager.get(prompt_name)

    def available(self) -> list[str]:
        """
        Return all available prompt domains.
        """

        return self.manager.list_prompts()
