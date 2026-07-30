from pathlib import Path


class PromptManager:
    def __init__(self):
        self.prompt_dir = Path(__file__).parent

    def load(self, name: str) -> str:
        path = self.prompt_dir / f"{name}.txt"

        if not path.exists():
            raise FileNotFoundError(
                f"Prompt '{name}' not found: {path}"
            )

        return path.read_text(encoding="utf-8")

    def list_prompts(self):
        return sorted(
            p.stem
            for p in self.prompt_dir.glob("*.txt")
        )
