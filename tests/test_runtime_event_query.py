from runtime.events.query import RuntimeEventQuery


def test_runtime_event_query_creates_with_defaults():

    query = RuntimeEventQuery()

    assert query.type is None
    assert query.run_id is None
    assert query.agent_name is None
    assert query.iteration is None
