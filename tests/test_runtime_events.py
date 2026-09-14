from runtime.core.runtime import QAIRRuntime
from runtime.events.event_type import RuntimeEventType
from runtime.events.query import RuntimeEventQuery


def test_runtime_exposes_shared_event_store():

    runtime = QAIRRuntime()

    assert runtime.event_store is runtime.run_service.event_store


def test_runtime_can_list_events():

    runtime = QAIRRuntime()

    runtime.create_run("run-1")

    runtime.run_service.start("run-1")

    events = runtime.list_events()

    assert len(events) == 1

    assert events[0].type is RuntimeEventType.RUN_STARTED


def test_runtime_can_query_events():

    runtime = QAIRRuntime()

    runtime.create_run("run-1")

    runtime.run_service.start("run-1")

    result = runtime.query_events(
        RuntimeEventQuery(
            type=RuntimeEventType.RUN_STARTED,
        )
    )

    assert len(result) == 1

    assert result[0].run_id == "run-1"


def test_runtime_can_get_events_for_run():

    runtime = QAIRRuntime()

    run = runtime.create_run("run-1")

    runtime.run_service.start(run.id)

    runtime.run_service.complete(run.id)

    events = runtime.get_run_events(run.id)

    assert len(events) == 2

    assert events[0].type is RuntimeEventType.RUN_STARTED

    assert events[1].type is RuntimeEventType.RUN_COMPLETED


def test_runtime_can_get_events_for_agent():

    runtime = QAIRRuntime()

    run = runtime.create_run(
        "run-1",
        agent_name="jafar",
    )

    runtime.run_service.start(run.id)

    events = runtime.get_agent_events("jafar")

    assert len(events) == 1

    assert events[0].agent_name == "jafar"


def test_runtime_returns_empty_events_for_unknown_run():

    runtime = QAIRRuntime()

    assert runtime.get_run_events("unknown-run") == []


def test_runtime_returns_empty_events_for_unknown_agent():

    runtime = QAIRRuntime()

    assert runtime.get_agent_events("unknown-agent") == []


def test_runtime_records_knowledge_retrieved_event():
    runtime = QAIRRuntime()

    runtime.start()

    runtime.knowledge_retriever.search = lambda query, limit: [
        object()
    ]

    runtime.knowledge_context_builder.build = (
        lambda documents, **kwargs: "Knowledge context."
    )

    runtime.engine.generate = lambda *args, **kwargs: None

    runtime.generate(
        [
            {
                "role": "user",
                "content": "What is QAIR?",
            }
        ],
        use_knowledge=True,
    )

    events = runtime.query_events(
        RuntimeEventQuery(
            type=RuntimeEventType.KNOWLEDGE_RETRIEVED,
        )
    )

    assert len(events) == 1
    assert events[0].data["query"] == "What is QAIR?"
    assert events[0].data["document_count"] == 1


def test_runtime_records_empty_knowledge_retrieval_event():
    runtime = QAIRRuntime()

    runtime.start()

    runtime.knowledge_retriever.search = lambda query, limit: []

    runtime.engine.generate = lambda *args, **kwargs: None

    runtime.generate(
        [
            {
                "role": "user",
                "content": "Unknown knowledge",
            }
        ],
        use_knowledge=True,
    )

    events = runtime.query_events(
        RuntimeEventQuery(
            type=RuntimeEventType.KNOWLEDGE_RETRIEVED,
        )
    )

    assert len(events) == 1
    assert events[0].data["query"] == "Unknown knowledge"
    assert events[0].data["document_count"] == 0
