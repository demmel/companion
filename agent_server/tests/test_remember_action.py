"""Unit tests for the deliberate REMEMBER action.

Exercises the action's orchestration of the memory recall interface (pure `query` read +
explicit `reinforce` mutation) and its structured output, using a fake `IMemory` so no real
embedding/retrieval is needed.
"""

from datetime import datetime

from agent.chain_of_action.action.actions.remember_action import RememberAction
from agent.chain_of_action.action.data.remember_data import RememberInput
from agent.chain_of_action.context import ExecutionContext
from agent.chain_of_action.trigger import UserInputTrigger
from agent.llm.models import SupportedModel
from agent.memory.memory import MemoryQueries, MemoryQuery, QueryType, RetrievedMemories
from agent.memory.dag.models import ConfidenceLevel, MemoryElement
from agent.state import create_default_agent_state


class FakeMemory:
    """Records query/reinforce calls and returns a canned recall result."""

    def __init__(self, result: RetrievedMemories | None = None, raise_on_query=False):
        self._result = result if result is not None else RetrievedMemories()
        self._raise = raise_on_query
        self.query_calls: list[MemoryQueries] = []
        self.reinforce_calls: list[RetrievedMemories] = []

    def query(self, memory_queries, llm, model) -> RetrievedMemories:
        if self._raise:
            raise RuntimeError("boom")
        self.query_calls.append(memory_queries)
        return self._result

    def reinforce(self, retrieved, budget, llm, model) -> None:
        self.reinforce_calls.append(retrieved)

    def prune(self, budget, llm, model) -> None: ...
    def get_formatted_context(self) -> str:
        return ""

    def store(self, trigger_history_entry, state, llm, model) -> None: ...


def _context(memory: FakeMemory) -> ExecutionContext:
    model = SupportedModel.CLAUDE_HAIKU_4_5
    return ExecutionContext(
        trigger=UserInputTrigger(content="hi", user_name="U"),
        situation_analysis="",
        session_id="s",
        agent_capabilities_knowledge_prompt="",
        memory=memory,
        memory_token_budget=1000,
        memory_retrieval_model=model,
        think_action_model=model,
        speak_action_model=model,
        visual_action_model=model,
        fetch_url_action_model=model,
        evaluate_priorities_action_model=model,
    )


def _input() -> RememberInput:
    return RememberInput(
        reason="ground myself in our past",
        queries=[
            MemoryQuery(
                reasoning="need context",
                query_type=QueryType.RELATIONSHIP,
                query_text="what David does for work",
                importance=0.9,
            )
        ],
    )


def _element(mid: str, content: str) -> MemoryElement:
    return MemoryElement(
        id=mid,
        content=content,
        timestamp=datetime(2026, 6, 1, 12, 0),
        confidence_level=ConfidenceLevel.USER_CONFIRMED,
        container_id="c1",
    )


def test_execute_maps_retrieved_elements_and_reinforces():
    retrieved = RetrievedMemories(
        elements=[_element("m1", "David is a developer")],
        actions=[],
    )
    memory = FakeMemory(result=retrieved)
    action_input = _input()

    result = RememberAction().execute(
        action_input,
        _context(memory),
        create_default_agent_state(),
        llm=None,
        progress_callback=lambda _: None,
    )

    assert result.type == "success"
    assert len(result.content.memories) == 1
    item = result.content.memories[0]
    assert item.memory_id == "m1"
    assert item.content == "David is a developer"
    assert item.confidence == "user_confirmed"
    # pure read then explicit reinforce, each exactly once, reinforce gets the query result
    assert len(memory.query_calls) == 1
    assert memory.query_calls[0].queries == action_input.queries
    assert memory.reinforce_calls == [retrieved]
    assert "David is a developer" in result.content.result_summary()


def test_execute_empty_recall():
    memory = FakeMemory(result=RetrievedMemories())
    result = RememberAction().execute(
        _input(),
        _context(memory),
        create_default_agent_state(),
        llm=None,
        progress_callback=lambda _: None,
    )
    assert result.type == "success"
    assert result.content.memories == []
    assert result.content.result_summary() == "No relevant memories found."
    assert len(memory.reinforce_calls) == 1  # still folds the (empty) recall


def test_execute_failure_surfaces_error():
    memory = FakeMemory(raise_on_query=True)
    result = RememberAction().execute(
        _input(),
        _context(memory),
        create_default_agent_state(),
        llm=None,
        progress_callback=lambda _: None,
    )
    assert result.type == "failure"
    assert "boom" in result.error
    assert memory.reinforce_calls == []  # never reached reinforce
