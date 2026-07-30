from runtime.prompts.manager import PromptManager

pm = PromptManager()

print("Available Prompts")
print("-----------------")
print(pm.list_prompts())

print("\nEmbedded Prompt")
print("-----------------")
print(pm.load("embedded"))
