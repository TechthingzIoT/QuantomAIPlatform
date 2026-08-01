from runtime.models.manager import ModelManager

manager = ModelManager("models")

print("Installed Models")

for model in manager.list_models():
    print("-", model.name)