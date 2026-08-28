"""
===========================================================
QAIR Runtime Tests
===========================================================

Test suite for the central QAIR runtime orchestration layer.

Covers:
- Runtime initialization
- Dependency injection
- Lifecycle management
- Model management
- Prompt management
- Runtime state and summary
- Context manager behavior
- Knowledge / RAG integration
- Knowledge-aware inference

Author:
    TIOTAIROBOTIX
===========================================================
"""

from unittest.mock import MagicMock

import pytest

from runtime.core.runtime import QAIRRuntime

# ============================================================
# Test Fixtures / Helpers
# ============================================================


def make_runtime():
    """Create an isolated QAIRRuntime with mocked dependencies."""

    model_manager = MagicMock()
    engine = MagicMock()
    prompt_selector = MagicMock()

    # --------------------------------------------------------
    # Mock model
    # --------------------------------------------------------

    model = MagicMock()
    model.name = "test-model.gguf"
    model.path = "/tmp/test-model.gguf"

    model_manager.list_models.return_value = [model]
    model_manager.active_model.return_value = model
    model_manager.activate.return_value = model

    model_manager.summary.return_value = {
        "installed_models": 1,
        "active_model": "test-model.gguf",
        "total_size": 123456789,
    }

    # --------------------------------------------------------
    # Mock inference engine
    # --------------------------------------------------------

    engine.loaded = False

    engine.summary.return_value = {
        "loaded": False,
        "model": "test-model.gguf",
        "context": 2048,
        "gpu_layers": 18,
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 256,
    }

    engine.generate.return_value = "Generated response."

    # --------------------------------------------------------
    # Mock prompt selector
    # --------------------------------------------------------

    prompt_selector.DEFAULT_PROMPT = "assistant"

    prompt_selector.available.return_value = [
        "assistant",
        "embedded",
        "robotics",
        "agriculture",
        "coding",
    ]

    prompt_selector.select.return_value = "Assistant system prompt"

    # --------------------------------------------------------
    # Mock knowledge dependencies
    # --------------------------------------------------------

    knowledge_store = MagicMock()
    knowledge_retriever = MagicMock()
    knowledge_context_builder = MagicMock()

    knowledge_store.get.return_value = None
    knowledge_retriever.search.return_value = []
    knowledge_context_builder.build.return_value = ""

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    runtime = QAIRRuntime(
        model_manager=model_manager,
        engine=engine,
        prompt_selector=prompt_selector,
        knowledge_store=knowledge_store,
        knowledge_retriever=knowledge_retriever,
        knowledge_context_builder=knowledge_context_builder,
    )

    return (
        runtime,
        model_manager,
        engine,
        prompt_selector,
    )


# ============================================================
# Initialization & Architecture
# ============================================================


def test_runtime_initialization():
    runtime, _, _, _ = make_runtime()

    assert runtime.running is False


def test_runtime_shares_model_manager_with_engine():
    custom_manager = MagicMock()

    runtime = QAIRRuntime(
        model_manager=custom_manager,
    )

    assert runtime.model_manager is custom_manager
    assert runtime.engine.manager is custom_manager


# ============================================================
# Startup
# ============================================================


def test_runtime_start():
    runtime, manager, engine, _ = make_runtime()

    runtime.start()

    manager.list_models.assert_called_once()
    manager.active_model.assert_called()
    engine.load.assert_called_once()

    assert runtime.running is True


def test_runtime_start_is_idempotent():
    runtime, _, engine, _ = make_runtime()

    runtime.start()
    runtime.start()

    engine.load.assert_called_once()


def test_runtime_start_without_models_fails():
    runtime, manager, _, _ = make_runtime()

    manager.list_models.return_value = []

    with pytest.raises(
        RuntimeError,
        match="No AI models discovered",
    ):
        runtime.start()


def test_runtime_start_without_active_model_fails():
    runtime, manager, _, _ = make_runtime()

    manager.active_model.return_value = None

    with pytest.raises(
        RuntimeError,
        match="No active model selected",
    ):
        runtime.start()


# ============================================================
# Shutdown
# ============================================================


def test_runtime_stop():
    runtime, _, engine, _ = make_runtime()

    runtime.start()
    runtime.stop()

    engine.unload.assert_called_once()

    assert runtime.running is False


def test_runtime_stop_is_idempotent():
    runtime, _, engine, _ = make_runtime()

    runtime.stop()

    engine.unload.assert_not_called()


# ============================================================
# Runtime State
# ============================================================


def test_runtime_loaded():
    runtime, _, engine, _ = make_runtime()

    engine.loaded = True

    assert runtime.loaded is True


def test_runtime_active_model():
    runtime, manager, _, _ = make_runtime()

    assert runtime.active_model is manager.active_model.return_value


# ============================================================
# Model Management
# ============================================================


def test_runtime_refresh_models():
    runtime, manager, _, _ = make_runtime()

    runtime.refresh_models()

    manager.refresh.assert_called_once()


def test_runtime_list_models():
    runtime, manager, _, _ = make_runtime()

    models = runtime.list_models()

    assert models == manager.list_models.return_value


def test_runtime_activate_model():
    runtime, manager, _, _ = make_runtime()

    model = runtime.activate_model("test-model.gguf")

    manager.activate.assert_called_once_with(
        "test-model.gguf",
    )

    assert model is manager.activate.return_value


def test_runtime_activate_model_reloads_when_running():
    runtime, manager, engine, _ = make_runtime()

    runtime.start()

    # Clear calls generated by startup.
    engine.reset_mock()

    runtime.activate_model("test-model.gguf")

    manager.activate.assert_called_once_with(
        "test-model.gguf",
    )

    engine.reload.assert_called_once()


# ============================================================
# Prompt Management
# ============================================================


def test_runtime_available_prompts():
    runtime, _, _, selector = make_runtime()

    prompts = runtime.available_prompts()

    assert prompts == selector.available.return_value


def test_runtime_get_prompt():
    runtime, _, _, selector = make_runtime()

    selector.select.return_value = "Embedded systems prompt"

    prompt = runtime.get_prompt("embedded")

    selector.select.assert_called_once_with(
        "embedded",
    )

    assert prompt == "Embedded systems prompt"


# ============================================================
# Runtime Summary
# ============================================================


def test_runtime_summary():
    runtime, manager, engine, _ = make_runtime()

    summary = runtime.summary()

    # Runtime state
    assert summary["running"] is False
    assert summary["loaded"] is False

    # Model manager information
    assert summary["active_model"] == "test-model.gguf"
    assert summary["installed_models"] == 1
    assert summary["total_model_size"] == 123456789

    # Inference engine information
    assert summary["model"] == "test-model.gguf"
    assert summary["context"] == 2048
    assert summary["gpu_layers"] == 18
    assert summary["temperature"] == 0.3
    assert summary["top_p"] == 0.95
    assert summary["max_tokens"] == 256

    manager.summary.assert_called_once()
    engine.summary.assert_called_once()


# ============================================================
# Inference
# ============================================================


def test_runtime_generate_without_knowledge_preserves_messages():
    runtime, _, engine, _ = make_runtime()

    runtime.start()

    messages = [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    runtime.generate(messages)

    engine.generate.assert_called_once_with(
        messages,
        max_tokens=None,
        temperature=None,
        top_p=None,
    )

    assert messages == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]


# ============================================================
# Knowledge / RAG
# ============================================================


def test_runtime_accepts_knowledge_dependencies():
    knowledge_store = MagicMock()
    knowledge_retriever = MagicMock()
    context_builder = MagicMock()

    runtime, _, engine, selector = make_runtime()

    runtime = QAIRRuntime(
        model_manager=runtime.model_manager,
        engine=engine,
        prompt_selector=selector,
        knowledge_store=knowledge_store,
        knowledge_retriever=knowledge_retriever,
        knowledge_context_builder=context_builder,
    )

    assert runtime.knowledge_store is knowledge_store
    assert runtime.knowledge_retriever is knowledge_retriever
    assert runtime.knowledge_context_builder is context_builder


def test_runtime_add_knowledge():
    runtime, _, _, _ = make_runtime()

    document = MagicMock()
    document.id = "doc-1"

    runtime.add_knowledge(document)

    runtime.knowledge_store.add.assert_called_once_with(
        document,
    )


def test_runtime_add_knowledge_many():
    runtime, _, _, _ = make_runtime()

    documents = [
        MagicMock(),
        MagicMock(),
    ]

    runtime.add_knowledge_many(documents)

    runtime.knowledge_store.add_many.assert_called_once_with(
        documents,
    )


def test_runtime_clear_knowledge():
    runtime, _, _, _ = make_runtime()

    runtime.clear_knowledge()

    runtime.knowledge_store.clear.assert_called_once()


def test_runtime_search_knowledge():
    runtime, _, _, _ = make_runtime()

    runtime.knowledge_retriever.search.return_value = ["result"]

    results = runtime.search_knowledge(
        "Rwanda AI",
        limit=3,
    )

    runtime.knowledge_retriever.search.assert_called_once_with(
        "Rwanda AI",
        limit=3,
    )

    assert results == ["result"]


def test_runtime_generate_with_knowledge_retrieves_latest_user_message():
    runtime, _, _, _ = make_runtime()

    runtime.start()

    runtime.knowledge_retriever.search.return_value = [MagicMock()]

    runtime.knowledge_context_builder.build.return_value = "Retrieved knowledge."

    messages = [
        {
            "role": "system",
            "content": "You are QAIR.",
        },
        {
            "role": "user",
            "content": "Old question",
        },
        {
            "role": "assistant",
            "content": "Old answer",
        },
        {
            "role": "user",
            "content": "What is Rwanda AI?",
        },
    ]

    runtime.generate(
        messages,
        use_knowledge=True,
        knowledge_limit=3,
    )

    runtime.knowledge_retriever.search.assert_called_once_with(
        "What is Rwanda AI?",
        limit=3,
    )


def test_runtime_generate_with_knowledge_injects_context():
    runtime, _, engine, _ = make_runtime()

    runtime.start()

    document = MagicMock()

    runtime.knowledge_retriever.search.return_value = [document]

    runtime.knowledge_context_builder.build.return_value = "Retrieved knowledge."

    messages = [
        {
            "role": "system",
            "content": "You are QAIR.",
        },
        {
            "role": "user",
            "content": "What is QAIR?",
        },
    ]

    runtime.generate(
        messages,
        use_knowledge=True,
    )

    augmented = engine.generate.call_args.args[0]

    # RAG context is merged into the existing system message.
    # QAIR must not create a second system message.
    assert augmented[0]["role"] == "system"
    assert augmented[0]["content"].startswith("You are QAIR.")
    assert "Retrieved knowledge." in augmented[0]["content"]

    # The original user message remains immediately after
    # the augmented system message.
    assert augmented[1] == messages[1]

    # The original input messages must remain unchanged.
    assert messages[0]["content"] == "You are QAIR."
    assert messages[1]["content"] == "What is QAIR?"


def test_runtime_generate_with_knowledge_does_not_mutate_messages():
    runtime, _, _, _ = make_runtime()

    runtime.start()

    runtime.knowledge_retriever.search.return_value = []

    messages = [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    original = [message.copy() for message in messages]

    runtime.generate(
        messages,
        use_knowledge=True,
    )

    assert messages == original


def test_runtime_generate_with_knowledge_without_results_uses_original_messages():
    runtime, _, engine, _ = make_runtime()

    runtime.start()

    runtime.knowledge_retriever.search.return_value = []

    messages = [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    runtime.generate(
        messages,
        use_knowledge=True,
    )

    engine.generate.assert_called_once_with(
        messages,
        max_tokens=None,
        temperature=None,
        top_p=None,
    )


def test_runtime_generate_rejects_invalid_knowledge_limit():
    runtime, _, _, _ = make_runtime()

    with pytest.raises(
        ValueError,
        match="knowledge_limit must be greater than zero",
    ):
        runtime.generate(
            [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            use_knowledge=True,
            knowledge_limit=0,
        )


# ============================================================
# Context Manager
# ============================================================


def test_runtime_context_manager():
    runtime, _, engine, _ = make_runtime()

    with runtime:
        assert runtime.running is True

    engine.load.assert_called_once()
    engine.unload.assert_called_once()

    assert runtime.running is False
