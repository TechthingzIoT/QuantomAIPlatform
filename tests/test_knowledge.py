from unittest.mock import patch
from runtime.knowledge.similarity import cosine_similarity
import pytest

from runtime.knowledge import (
    EmbeddingProvider,
    KnowledgeDocument,
    KnowledgeRetriever,
    KnowledgeStore,
    LlamaEmbeddingProvider,
)


def test_document_serialization_round_trip():
    document = KnowledgeDocument(
        id="rwanda-ai",
        title="Rwanda AI Strategy",
        content="Rwanda is developing AI capabilities.",
        source="government",
        metadata={"country": "Rwanda"},
    )

    data = document.to_dict()

    restored = KnowledgeDocument.from_dict(data)

    assert restored.id == document.id
    assert restored.title == document.title
    assert restored.content == document.content
    assert restored.source == document.source
    assert restored.metadata == document.metadata


def test_document_requires_id():
    with pytest.raises(ValueError):
        KnowledgeDocument.from_dict(
            {
                "content": "Some knowledge.",
            }
        )


def test_document_requires_content():
    with pytest.raises(ValueError):
        KnowledgeDocument.from_dict(
            {
                "id": "doc-1",
            }
        )


def test_store_add_and_get():
    store = KnowledgeStore()

    document = KnowledgeDocument(
        id="doc-1",
        content="QAIR is a local AI runtime.",
    )

    store.add(document)

    assert len(store) == 1
    assert store.get("doc-1") is document


def test_store_replaces_existing_document():
    store = KnowledgeStore()

    first = KnowledgeDocument(
        id="doc-1",
        content="Version one.",
    )

    second = KnowledgeDocument(
        id="doc-1",
        content="Version two.",
    )

    store.add(first)
    store.add(second)

    assert len(store) == 1
    assert store.get("doc-1") is second


def test_store_add_many():
    store = KnowledgeStore()

    documents = [
        KnowledgeDocument(
            id="doc-1",
            content="AI infrastructure.",
        ),
        KnowledgeDocument(
            id="doc-2",
            content="Robotics infrastructure.",
        ),
    ]

    store.add_many(documents)

    assert len(store) == 2


def test_store_remove():
    store = KnowledgeStore()

    store.add(
        KnowledgeDocument(
            id="doc-1",
            content="Test.",
        )
    )

    store.remove("doc-1")

    assert len(store) == 0
    assert store.get("doc-1") is None


def test_store_clear():
    store = KnowledgeStore()

    store.add_many(
        [
            KnowledgeDocument(
                id="doc-1",
                content="One.",
            ),
            KnowledgeDocument(
                id="doc-2",
                content="Two.",
            ),
        ]
    )

    store.clear()

    assert len(store) == 0


def test_retriever_returns_relevant_documents():
    store = KnowledgeStore()

    store.add_many(
        [
            KnowledgeDocument(
                id="ai",
                title="Artificial Intelligence",
                content=(
                    "AI infrastructure supports "
                    "machine learning research."
                ),
            ),
            KnowledgeDocument(
                id="robotics",
                title="Robotics",
                content=(
                    "Robotics systems use sensors "
                    "and actuators."
                ),
            ),
        ]
    )

    retriever = KnowledgeRetriever(store)

    results = retriever.search("AI infrastructure")

    assert results
    assert results[0].id == "ai"


def test_retriever_respects_limit():
    store = KnowledgeStore()

    for index in range(5):
        store.add(
            KnowledgeDocument(
                id=f"doc-{index}",
                content="AI infrastructure.",
            )
        )

    retriever = KnowledgeRetriever(store)

    results = retriever.search(
        "AI infrastructure",
        limit=2,
    )

    assert len(results) == 2


def test_retriever_empty_query():
    store = KnowledgeStore()

    store.add(
        KnowledgeDocument(
            id="doc-1",
            content="AI.",
        )
    )

    retriever = KnowledgeRetriever(store)

    assert retriever.search("") == []


def test_retriever_invalid_limit():
    store = KnowledgeStore()

    retriever = KnowledgeRetriever(store)

    assert retriever.search(
        "AI",
        limit=0,
    ) == []


# ==========================================================
# Embedding Provider
# ==========================================================


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for testing."""

    def embed(self, text: str) -> list[float]:
        return [
            float(len(text)),
            float(len(text.split())),
        ]


def test_embedding_provider_embeds_text():
    provider = FakeEmbeddingProvider()

    vector = provider.embed("hello world")

    assert vector == [11.0, 2.0]


def test_embedding_provider_embeds_many_texts():
    provider = FakeEmbeddingProvider()

    vectors = provider.embed_many(
        [
            "hello",
            "hello world",
        ]
    )

    assert vectors == [
        [5.0, 1.0],
        [11.0, 2.0],
    ]


# ==========================================================
# llama.cpp Embedding Provider
# ==========================================================


def test_llama_embedding_provider_single_embedding():
    response = {
        "data": [
            {
                "embedding": [
                    0.1,
                    0.2,
                    0.3,
                ]
            }
        ]
    }

    with patch(
        "runtime.knowledge.llama_embeddings.Llama"
    ) as llama_class:
        model = llama_class.return_value
        model.create_embedding.return_value = response

        provider = LlamaEmbeddingProvider(
            "/tmp/embedding.gguf"
        )

        vector = provider.embed("hello")

        assert vector == [
            0.1,
            0.2,
            0.3,
        ]

        model.create_embedding.assert_called_once_with(
            "hello"
        )


def test_llama_embedding_provider_multiple_embeddings():
    response = {
        "data": [
            {
                "embedding": [
                    0.1,
                    0.2,
                ]
            },
            {
                "embedding": [
                    0.3,
                    0.4,
                ]
            },
        ]
    }

    with patch(
        "runtime.knowledge.llama_embeddings.Llama"
    ) as llama_class:
        model = llama_class.return_value
        model.create_embedding.return_value = response

        provider = LlamaEmbeddingProvider(
            "/tmp/embedding.gguf"
        )

        vectors = provider.embed_many(
            [
                "hello",
                "world",
            ]
        )

        assert vectors == [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        model.create_embedding.assert_called_once_with(
            [
                "hello",
                "world",
            ]
        )


def test_llama_embedding_provider_rejects_empty_text():
    with patch(
        "runtime.knowledge.llama_embeddings.Llama"
    ):
        provider = LlamaEmbeddingProvider(
            "/tmp/embedding.gguf"
        )

        with pytest.raises(
            ValueError,
            match="Embedding input cannot be empty.",
        ):
            provider.embed("")


def test_llama_embedding_provider_rejects_non_string():
    with patch(
        "runtime.knowledge.llama_embeddings.Llama"
    ):
        provider = LlamaEmbeddingProvider(
            "/tmp/embedding.gguf"
        )

        with pytest.raises(
            TypeError,
            match="Embedding input must be a string.",
        ):
            provider.embed(123)  # type: ignore[arg-type]


            from runtime.knowledge.similarity import cosine_similarity


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    ) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    ) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    ) == pytest.approx(-1.0)


def test_cosine_similarity_requires_same_dimension():
    with pytest.raises(ValueError):
        cosine_similarity(
            [1.0, 2.0],
            [1.0],
        )


def test_cosine_similarity_rejects_empty_vectors():
    with pytest.raises(ValueError):
        cosine_similarity([], [])


def test_cosine_similarity_rejects_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 0.0],
        )

# ==========================================================
# Hybrid Retrieval
# ==========================================================


def test_document_embedding_round_trip():
    document = KnowledgeDocument(
        id="rwanda-ai",
        content="Rwanda AI strategy.",
        embedding=[0.1, 0.2, 0.3],
    )

    restored = KnowledgeDocument.from_dict(
        document.to_dict()
    )

    assert restored.embedding == [
        0.1,
        0.2,
        0.3,
    ]


def test_retriever_without_embedding_provider_uses_keywords():
    store = KnowledgeStore()

    store.add_many(
        [
            KnowledgeDocument(
                id="ai",
                content="Artificial intelligence infrastructure.",
            ),
            KnowledgeDocument(
                id="robotics",
                content="Robotics and automation.",
            ),
        ]
    )

    retriever = KnowledgeRetriever(store)

    results = retriever.search("AI infrastructure")

    assert results
    assert results[0].id == "ai"


def test_retriever_uses_semantic_similarity():
    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed(self, text: str) -> list[float]:
            if "robot" in text.lower():
                return [0.0, 1.0]

            return [1.0, 0.0]

    store = KnowledgeStore()

    store.add_many(
        [
            KnowledgeDocument(
                id="ai",
                content="Artificial intelligence.",
                embedding=[1.0, 0.0],
            ),
            KnowledgeDocument(
                id="robotics",
                content="Robotics systems.",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    retriever = KnowledgeRetriever(
        store,
        FakeEmbeddingProvider(),
    )

    results = retriever.search("robotics")

    assert results
    assert results[0].id == "robotics"


def test_retriever_hybrid_ranking():
    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0]

    store = KnowledgeStore()

    store.add_many(
        [
            KnowledgeDocument(
                id="semantic",
                content="Machine learning systems.",
                embedding=[1.0, 0.0],
            ),
            KnowledgeDocument(
                id="keyword",
                content="Rwanda AI infrastructure.",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    retriever = KnowledgeRetriever(
        store,
        FakeEmbeddingProvider(),
    )

    results = retriever.search("Rwanda AI")

    assert results
    assert results[0].id in {
        "semantic",
        "keyword",
    }


def test_retriever_rejects_negative_keyword_weight():
    store = KnowledgeStore()

    with pytest.raises(ValueError):
        KnowledgeRetriever(
            store,
            keyword_weight=-0.1,
        )


def test_retriever_rejects_zero_weights():
    store = KnowledgeStore()

    with pytest.raises(ValueError):
        KnowledgeRetriever(
            store,
            keyword_weight=0.0,
            semantic_weight=0.0,
        )


# ==========================================================
# Knowledge Context Builder
# ==========================================================

from runtime.knowledge import KnowledgeContextBuilder


def test_context_builder_formats_documents():
    documents = [
        KnowledgeDocument(
            id="rwanda-ai",
            title="Rwanda AI Strategy",
            content="Rwanda is developing national AI capabilities.",
            source="Government of Rwanda",
        ),
    ]

    builder = KnowledgeContextBuilder()

    context = builder.build(documents)

    assert "[Knowledge 1]" in context
    assert "Title: Rwanda AI Strategy" in context
    assert "Source: Government of Rwanda" in context
    assert "Rwanda is developing national AI capabilities." in context


def test_context_builder_returns_empty_for_no_documents():
    builder = KnowledgeContextBuilder()

    assert builder.build([]) == ""


def test_context_builder_supports_multiple_documents():
    documents = [
        KnowledgeDocument(
            id="doc-1",
            title="AI",
            content="Artificial intelligence.",
        ),
        KnowledgeDocument(
            id="doc-2",
            title="Robotics",
            content="Robotics engineering.",
        ),
    ]

    builder = KnowledgeContextBuilder()

    context = builder.build(documents)

    assert "[Knowledge 1]" in context
    assert "[Knowledge 2]" in context
    assert "Artificial intelligence." in context
    assert "Robotics engineering." in context


def test_context_builder_enforces_character_limit():
    documents = [
        KnowledgeDocument(
            id="doc-1",
            content="A" * 1000,
        ),
    ]

    builder = KnowledgeContextBuilder(
        max_characters=100,
    )

    context = builder.build(documents)

    assert len(context) <= 100


def test_context_builder_rejects_invalid_character_limit():
    with pytest.raises(ValueError):
        KnowledgeContextBuilder(
            max_characters=0,
        )
