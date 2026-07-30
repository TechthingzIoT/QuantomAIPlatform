from runtime.models.model_manager import ModelManager

manager = ModelManager()

print("Installed Models")
print("=" * 30)

models = manager.list_models()

if not models:
    print("No models found.")
else:
    for model in models:
        print(model.name)
