"""Optional real-model llama.cpp smoke test."""

import os
import time
from pathlib import Path

import pytest
from llama_cpp import Llama

MODEL_ENV_VAR = "QAIR_SMOKE_MODEL"


def test_llama_cpp_smoke():
    """Run a real llama.cpp inference smoke test when a model is configured."""
    model_path = os.getenv(MODEL_ENV_VAR)

    if not model_path:
        pytest.skip(
            f"{MODEL_ENV_VAR} is not set; skipping real-model smoke test."
        )

    model = Path(model_path).expanduser()

    if not model.is_file():
        pytest.fail(f"Smoke-test model does not exist: {model}")

    print(f"Loading model: {model}")

    llm = Llama(
        model_path=str(model),
        n_ctx=2048,
        n_gpu_layers=0,
        verbose=True,
    )

    print("Model loaded.")

    start = time.time()

    response = llm(
        "What is ESP32?",
        max_tokens=64,
        stop=["</s>"],
    )

    elapsed = time.time() - start

    print(f"\nInference time: {elapsed:.2f} sec\n")
    print(response["choices"][0]["text"])
