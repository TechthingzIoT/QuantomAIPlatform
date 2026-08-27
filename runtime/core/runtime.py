"""
QAIR Core Runtime

Central lifecycle controller for the Quantom AI Runtime.

Responsibilities
----------------
- Initialize QAIR runtime components
- Discover and manage models
- Load and unload the inference engine
- Expose runtime state
- Provide controlled shutdown
- Optionally augment inference with retrieved knowledge
"""

from typing import Self

from runtime.config.settings import settings
from runtime.inference.engine import InferenceEngine
from runtime.knowledge.context import KnowledgeContextBuilder
from runtime.knowledge.indexer import KnowledgeIndexer
from runtime.knowledge.llama_embeddings import LlamaEmbeddingProvider
from runtime.knowledge.loader import KnowledgeLoader
from runtime.knowledge.registry import list_sources
from runtime.knowledge.retriever import KnowledgeRetriever
from runtime.knowledge.store import KnowledgeStore
from runtime.models.manager import ModelManager
from runtime.prompts.selection import PromptSelector


class QAIRRuntime:
    """
    Central runtime orchestration layer.

    QAIRRuntime coordinates model management, inference,
    prompt selection, and optional knowledge retrieval.
    """

    def __init__(
        self,
        *,
        model_manager: ModelManager | None = None,
        engine: InferenceEngine | None = None,
        prompt_selector: PromptSelector | None = None,
        knowledge_store: KnowledgeStore | None = None,
        knowledge_retriever: KnowledgeRetriever | None = None,
        knowledge_context_builder: KnowledgeContextBuilder | None = None,
    ) -> None:
        # --------------------------------------------------
        # Core dependencies
        # --------------------------------------------------

        self.model_manager = (
            model_manager if model_manager is not None else ModelManager()
        )

        self.engine = (
            engine
            if engine is not None
            else InferenceEngine(
                model_manager=self.model_manager,
            )
        )

        self.prompt_selector = (
            prompt_selector if prompt_selector is not None else PromptSelector()
        )

        # --------------------------------------------------
        # Knowledge dependencies
        # --------------------------------------------------

        self.knowledge_store = (
            knowledge_store if knowledge_store is not None else KnowledgeStore()
        )

        # --------------------------------------------------
        # Optional embedding infrastructure
        # --------------------------------------------------

        self.embedding_provider = self._create_embedding_provider()

        self.knowledge_indexer = (
            KnowledgeIndexer(self.embedding_provider)
            if self.embedding_provider is not None
            else None
        )

        self.knowledge_retriever = (
            knowledge_retriever
            if knowledge_retriever is not None
            else KnowledgeRetriever(
                self.knowledge_store,
            )
        )

        self.knowledge_context_builder = (
            knowledge_context_builder
            if knowledge_context_builder is not None
            else KnowledgeContextBuilder()
        )

        self.running = False

    @staticmethod
    def _create_embedding_provider():
        """
        Create the configured local embedding provider.

        Embeddings are optional. When no embedding model path
        is configured, QAIR falls back to the existing
        knowledge retrieval behavior.
        """

        if not settings.embedding_model_path:
            return None

        return LlamaEmbeddingProvider(
            settings.embedding_model_path,
            n_ctx=settings.embedding_context_size,
            n_gpu_layers=settings.gpu_layers,
            verbose=settings.verbose,
        )

    # ==================================================
    # Lifecycle
    # ==================================================

    def start(self) -> None:
        """
        Start the QAIR runtime.

        Discovers models, loads the active model, and loads
        registered local knowledge sources.
        """

        if self.running:
            return

        models = self.model_manager.list_models()

        if not models:
            raise RuntimeError("No AI models discovered.")

        active = self.model_manager.active_model()

        if active is None:
            raise RuntimeError("No active model selected.")

        self._load_registered_knowledge()

        self.engine.load()
        self.running = True

    def stop(self) -> None:
        """
        Stop QAIR and release inference resources.
        """

        if not self.running:
            return

        self.engine.unload()
        self.running = False

    # ==================================================
    # Runtime State
    # ==================================================

    @property
    def loaded(self) -> bool:
        """Return whether the inference engine is loaded."""

        return self.engine.loaded

    @property
    def active_model(self):
        """Return the currently active model."""

        return self.model_manager.active_model()

    @property
    def active_prompt(self) -> str:
        """Return the name of the active prompt."""

        return self.prompt_selector.DEFAULT_PROMPT

    # ==================================================
    # Model Management
    # ==================================================

    def refresh_models(self):
        """Rediscover available AI models."""

        return self.model_manager.refresh()

    def list_models(self):
        """Return discovered AI models."""

        return self.model_manager.list_models()

    def activate_model(self, model_name: str):
        """
        Validate and activate a model.

        If QAIR is running, reload the inference engine
        so the newly activated model becomes live.
        """

        model = self.model_manager.activate(model_name)

        if self.running:
            self.engine.reload()

        return model

    # ==================================================
    # Prompt Management
    # ==================================================

    def available_prompts(self) -> list[str]:
        """Return all available system prompts."""

        return self.prompt_selector.available()

    def get_prompt(self, name: str) -> str:
        """Return the contents of a named prompt."""

        return self.prompt_selector.select(name)

    # ==================================================
    # Knowledge Management
    # ==================================================

    def _load_registered_knowledge(self) -> None:
        """
        Load all registered local knowledge sources.

        Each registered source directory is loaded through
        KnowledgeLoader.

        When an embedding provider is configured, the loaded
        documents are indexed before being stored.
        """

        self.knowledge_store.clear()

        for source in list_sources():
            loader = KnowledgeLoader(source)
            documents = loader.load()

            if self.knowledge_indexer is not None:
                documents = self.knowledge_indexer.index(
                    documents,
                )

            self.knowledge_store.add_many(documents)

    def add_knowledge(self, document) -> None:
        """
        Add a knowledge document to the runtime knowledge store.
        """

        self.knowledge_store.add(document)

    def add_knowledge_many(self, documents) -> None:
        """
        Add multiple knowledge documents.
        """

        self.knowledge_store.add_many(documents)

    def clear_knowledge(self) -> None:
        """Remove all knowledge documents."""

        self.knowledge_store.clear()

    def search_knowledge(
        self,
        query: str,
        *,
        limit: int = 5,
    ):
        """
        Search the runtime knowledge base.
        """

        return self.knowledge_retriever.search(
            query,
            limit=limit,
        )

    # ==================================================
    # Runtime Information
    # ==================================================

    def summary(self) -> dict:
        """
        Return a complete runtime summary.
        """

        model_summary = self.model_manager.summary()
        engine_summary = self.engine.summary()

        return {
            "running": self.running,
            "loaded": self.loaded,
            "active_model": model_summary["active_model"],
            "installed_models": model_summary["installed_models"],
            "total_model_size": model_summary["total_size"],
            "model": engine_summary["model"],
            "context": engine_summary["context"],
            "gpu_layers": engine_summary["gpu_layers"],
            "temperature": engine_summary["temperature"],
            "top_p": engine_summary["top_p"],
            "max_tokens": engine_summary["max_tokens"],
        }

    # ==================================================
    # Inference
    # ==================================================

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        use_knowledge: bool = False,
        knowledge_limit: int = 5,
    ) -> str:
        """
        Generate a response through the QAIR runtime.

        When ``use_knowledge`` is False, inference follows
        the existing direct inference path.

        When ``use_knowledge`` is True, QAIR:

        1. Extracts the latest user message.
        2. Retrieves relevant knowledge.
        3. Builds a controlled context block.
        4. Injects that context into the conversation.
        5. Sends the augmented conversation to the engine.
        """

        if not self.running:
            self.start()

        inference_messages = list(messages)

        if use_knowledge:
            if knowledge_limit <= 0:
                raise ValueError("knowledge_limit must be greater than zero.")

            query = self._latest_user_message(
                inference_messages,
            )

            if query:
                documents = self.knowledge_retriever.search(
                    query,
                    limit=knowledge_limit,
                )

                context = self.knowledge_context_builder.build(
                    documents,
                )

                if context:
                    inference_messages = self._augment_messages_with_context(
                        inference_messages,
                        context,
                    )

        return self.engine.generate(
            inference_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    # ==================================================
    # RAG Helpers
    # ==================================================

    @staticmethod
    def _latest_user_message(
        messages: list[dict],
    ) -> str:
        """
        Return the latest user message content.

        Messages are expected to use the OpenAI-compatible
        ``role`` and ``content`` structure.
        """

        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")

                if isinstance(content, str):
                    return content

        return ""

    @staticmethod
    def _augment_messages_with_context(
        messages: list[dict],
        context: str,
    ) -> list[dict]:
        """
        Add retrieved knowledge as a system-level context message.

        Existing messages are copied rather than mutated.
        """

        augmented = list(messages)

        knowledge_message = {
            "role": "system",
            "content": (
                "Use the following retrieved knowledge to help "
                "answer the user's request. Treat it as reference "
                "material and do not invent facts not supported "
                "by the retrieved context.\n\n"
                f"{context}"
            ),
        }

        # Insert knowledge immediately before the first user
        # message when possible. This keeps the original system
        # prompt at the beginning of the conversation.
        for index, message in enumerate(augmented):
            if message.get("role") == "user":
                augmented.insert(
                    index,
                    knowledge_message,
                )
                return augmented

        augmented.append(knowledge_message)

        return augmented

    # ==================================================
    # Context Manager
    # ==================================================

    def __enter__(self) -> Self:
        """Start QAIR when entering a context."""

        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """Stop QAIR when leaving a context."""

        self.stop()
