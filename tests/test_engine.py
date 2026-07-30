from runtime.inference.engine import LlamaEngine

MODEL = "/Users/cashamerica/.cache/huggingface/hub/models--bartowski--Qwen2.5-3B-Instruct-GGUF/snapshots/f302c64a2269a69fb27b2f9473b362f5bb8e78d8/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

print("Starting QAIR...")

engine = LlamaEngine(MODEL)

print("Asking model...")

response = engine.chat("Introduce yourself in one sentence.")

print("\nAssistant:\n")
print(response)
