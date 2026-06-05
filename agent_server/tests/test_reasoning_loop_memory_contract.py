"""Regression tests for the reasoning loop's memory orchestration."""

from typing import Iterator

from agent.chain_of_action.action_plan import ActionSequence
from agent.chain_of_action.callbacks import NoOpCallback
from agent.chain_of_action.reasoning_loop import ActionBasedReasoningLoop
from agent.chain_of_action.trigger import UserInputTrigger
from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.memory.memory import RetrievedMemories
from agent.state import create_default_agent_state


class FakeMemory:
    def __init__(self) -> None:
        self.prune_calls: list[tuple[int, object]] = []
        self.query_calls = 0
        self.reinforce_calls = 0
        self.store_calls: list[TriggerHistoryEntry] = []
        self.formatted_contexts = [
            "working context before planning",
            "working context after processing",
        ]

    def query(self, memory_queries, llm, model) -> RetrievedMemories:
        self.query_calls += 1
        raise AssertionError("automatic retrieval should not run")

    def reinforce(self, retrieved, budget, llm, model) -> None:
        self.reinforce_calls += 1

    def prune(self, budget, llm, model) -> None:
        self.prune_calls.append((budget, model))

    def get_formatted_context(self) -> str:
        return self.formatted_contexts.pop(0)

    def store(self, trigger_history_entry, state, llm, model) -> None:
        self.store_calls.append(trigger_history_entry)


class FakeLLM:
    def __init__(self) -> None:
        self.generate_calls: list[dict] = []

    def generate(self, model, prompt, caller, images=None):
        self.generate_calls.append(
            {
                "model": model,
                "prompt": prompt,
                "caller": caller,
                "images": images,
            }
        )
        return "situational analysis"


class NoActionPlanner:
    def __init__(self) -> None:
        self.completed_action_counts: list[int] = []

    def plan_actions(
        self,
        trigger,
        completed_actions,
        state,
        trigger_history,
        llm,
        model,
        situational_analysis,
    ) -> ActionSequence:
        self.completed_action_counts.append(len(completed_actions))
        return ActionSequence(
            completed_actions_review="nothing yet",
            sequence_plan="no actions needed",
            dependency_analysis="no dependencies",
            wait_decision="stop",
            actions=[],
        )


class FakeTriggerHistory:
    def __init__(self) -> None:
        self.entries: list[TriggerHistoryEntry] = []

    def add_entry(self, entry: TriggerHistoryEntry) -> None:
        self.entries.append(entry)

    def update_entry(self, entry: TriggerHistoryEntry) -> None:
        pass

    def get_first_entry(self) -> TriggerHistoryEntry | None:
        return None

    def get_last_entry(self) -> TriggerHistoryEntry | None:
        return self.entries[-1] if self.entries else None

    def get_entry_by_id(self, entry_id: str) -> TriggerHistoryEntry:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        raise KeyError(entry_id)

    def get_entry_count(self) -> int:
        return len(self.entries)

    def iter_entries(self, reverse: bool, start: int) -> Iterator[TriggerHistoryEntry]:
        entries = list(reversed(self.entries)) if reverse else self.entries
        return iter(entries[start:])

    def get_entry_index(self, entry_id: str) -> int:
        for index, entry in enumerate(self.entries):
            if entry.entry_id == entry_id:
                return index
        raise KeyError(entry_id)

    def get_last_entry_by_trigger_type(
        self, trigger_type: str
    ) -> TriggerHistoryEntry | None:
        return None

    def __len__(self) -> int:
        return len(self.entries)

    def close(self) -> None:
        pass


def test_process_trigger_prunes_existing_context_without_automatic_retrieval():
    memory = FakeMemory()
    llm = FakeLLM()
    planner = NoActionPlanner()
    trigger_history = FakeTriggerHistory()
    loop = ActionBasedReasoningLoop(enable_image_generation=False)
    loop.planner = planner

    trigger_entry, returned_context = loop.process_trigger(
        trigger=UserInputTrigger(content="hello", user_name="User"),
        state=create_default_agent_state(),
        llm=llm,
        callback=NoOpCallback(),
        trigger_history=trigger_history,
        token_budget=1234,
        memory=memory,
        previous_memory_context="old automatic context should be ignored",
        individual_trigger_compression=False,
    )

    assert len(memory.prune_calls) == 1
    assert memory.prune_calls[0][0] == 1234
    assert memory.query_calls == 0
    assert memory.reinforce_calls == 0
    assert len(memory.store_calls) == 1
    assert memory.store_calls[0] is trigger_entry
    assert trigger_history.entries == [trigger_entry]
    assert trigger_entry.situational_context == "situational analysis"
    assert trigger_entry.actions_taken == []
    assert returned_context == "working context after processing"
    assert planner.completed_action_counts == [0]
    assert len(llm.generate_calls) == 1
    assert llm.generate_calls[0]["caller"] == "situational_analysis"
    assert "working context before planning" in llm.generate_calls[0]["prompt"]
    assert (
        "old automatic context should be ignored" not in llm.generate_calls[0]["prompt"]
    )
