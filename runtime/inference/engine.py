"""
=========================================================
QAIR (Quantom AI Runtime)

File:
    runtime/inference/engine.py

Purpose:
    Core inference engine responsible for loading local
    GGUF language models and generating responses.

Author:
    TIOTAIROBOTIX
=========================================================
"""

import time
from pathlib import Path
from typing import Optional, List, Dict

from llama_cpp import Llama

from runtime.config.loader import load_settings


class LlamaEngine:
    """
    QAIR Local Language Model Engine.

    Loads a GGUF model using llama.cpp and exposes a simple
    chat interface for the rest of QAIR.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
    ):

        # Load runtime configuration
        self.settings = load_settings()

        self.model_path = model_path or self.settings.model_path
        self.n_ctx = n_ctx or self.settings.context_size
        self.n_gpu_layers = (
            n_gpu_layers
            if n_gpu_layers is not None
            else self.settings.gpu_layers
        )

        model = Path(self.model_path)

        if not model.exists():
            raise FileNotFoundError(model)

        print("\n============================================================")
        print("QAIR Runtime")
        print("============================================================")
        print(f"Model      : {model.name}")
        print(f"Context    : {self.n_ctx}")
        print(f"GPU Layers : {self.n_gpu_layers}")
        print()

        start = time.time()

        self.llm = Llama(
            model_path=str(model),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.settings.verbose,
        )

        elapsed = time.time() - start

        print(f"✓ Model loaded in {elapsed:.2f} sec\n")

    def chat(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a chat response.

        Supports either:

            prompt="Hello"

        or

            messages=[
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
            ]
        """

        if messages is None:

            messages = []

            if system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                )

            if prompt:
                messages.append(
                    {
                        "role": "user",
                        "content": prompt,
                    }
                )

        print("Generating...\n")

        start = time.time()

        output = self.llm.create_chat_completion(
            messages=messages,
            temperature=(
                temperature
                if temperature is not None
                else self.settings.temperature
            ),
            top_p=(
                top_p
                if top_p is not None
                else self.settings.top_p
            ),
            max_tokens=(
                max_tokens
                if max_tokens is not None
                else self.settings.max_tokens
            ),
        )

        elapsed = time.time() - start

        print("------------------------------------------------------------")
        print(f"Inference Time : {elapsed:.2f} sec")
        print("------------------------------------------------------------")

        return output["choices"][0]["message"]["content"]

    def info(self):
        """
        Return runtime information.
        """

        return {
            "model": Path(self.model_path).name,
            "context": self.n_ctx,
            "gpu_layers": self.n_gpu_layers,
        }