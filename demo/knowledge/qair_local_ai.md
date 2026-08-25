# QAIR — Quantom AI Runtime

## What is QAIR?

QAIR is a local AI runtime designed to run language models on local computing hardware.

The current prototype uses GGUF model files and llama.cpp-based inference.

## Current Demonstration

The current QAIR prototype successfully demonstrates:

- Local model discovery.
- Local model selection.
- Local model loading.
- Local inference.
- OpenAI-compatible chat completion API.
- Runtime health monitoring.
- Readiness monitoring.
- Local knowledge storage.
- Local knowledge retrieval.
- Retrieval-augmented generation.
- Operation without requiring a remote AI inference API.

## Current Model

The demonstration model is:

Qwen2.5-3B-Instruct-Q4_K_M.gguf

## Local RAG

QAIR can retrieve relevant documents from a local knowledge store and inject the retrieved context into the model's conversation.

The retrieval layer currently supports:

- Deterministic keyword retrieval.
- Semantic vector retrieval when embeddings are available.
- Hybrid retrieval.
- Controlled context construction.

## Why Local Inference Matters

Local inference can provide:

- Greater control over where inference occurs.
- Reduced dependence on external AI inference services.
- Ability to operate in disconnected or constrained environments.
- A foundation for private AI applications.
- A platform for experimentation with sovereign AI infrastructure.

## Demonstration Principle

The important QAIR demonstration is not simply:

"Here is a chatbot."

The demonstration is:

"Here is an AI runtime running the model, knowledge and inference locally."

## Limitations

Local inference does not automatically make an AI system sovereign.

A complete sovereign AI ecosystem also involves:

- Computing infrastructure.
- Energy.
- Data ownership and governance.
- Model development.
- AI talent.
- Cybersecurity.
- Hardware supply chains.
- Software infrastructure.
- Responsible AI governance.

QAIR demonstrates one technical layer of that larger ecosystem: local AI inference and knowledge retrieval.
