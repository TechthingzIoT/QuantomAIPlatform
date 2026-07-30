from llama_cpp import Llama
import time

MODEL = "/Users/cashamerica/.cache/huggingface/hub/models--bartowski--Qwen2.5-3B-Instruct-GGUF/snapshots/f302c64a2269a69fb27b2f9473b362f5bb8e78d8/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

print("Loading model...")

llm = Llama(
    model_path=MODEL,
    n_ctx=2048,
    n_gpu_layers=0,   # CPU only for debugging
    verbose=True,
)

print("Model loaded.")

start = time.time()

response = llm(
    "What is ESP32?",
    max_tokens=64,
    stop=["</s>"],
)

print(f"\nInference time: {time.time()-start:.2f} sec\n")
print(response["choices"][0]["text"])
