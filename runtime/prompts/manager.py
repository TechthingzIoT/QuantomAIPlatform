from pathlib import Path


class PromptManager:
    """
    Loads and manages QAIR system prompts.
    """

    def __init__(self):
        self.prompt_dir = Path(__file__).parent

    def exists(self, name: str) -> bool:
        """
        Return True if a named prompt exists.
        """
        return (self.prompt_dir / f"{name}.txt").is_file()

    def load(self, name: str) -> str:
        """
        Load a named prompt.
        """

        path = self.prompt_dir / f"{name}.txt"

        if not path.exists():
            raise FileNotFoundError(
                f"Prompt '{name}' not found: {path}"
            )

        return path.read_text(encoding="utf-8").strip()

    def get(self, name: str) -> str:
        """
        Alias for load().
        """
        return self.load(name)

    def list_prompts(self) -> list[str]:
        """
        Return all available prompt names.
        """

        return sorted(
            path.stem
            for path in self.prompt_dir.glob("*.txt")
            if path.is_file()
        )
