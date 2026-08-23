"""Inspect models discovered by QAIR."""

from runtime.models.manager import ModelManager


def main() -> None:
    """Print installed QAIR models."""
    manager = ModelManager()

    print("Installed Models")
    print("=" * 30)

    models = manager.list_models()

    if not models:
        print("No models found.")
        return

    for model in models:
        print(model.name)


if __name__ == "__main__":
    main()
