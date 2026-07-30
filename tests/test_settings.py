from runtime.config.settings import settings

print("Current Configuration")
print("=====================")

print("Model:", settings.model)
print("Temperature:", settings.temperature)
print("Top P:", settings.top_p)
print("Max Tokens:", settings.max_tokens)
print("Context:", settings.context_size)
print("GPU Layers:", settings.gpu_layers)
print("Verbose:", settings.verbose)
