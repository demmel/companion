from dataclasses import dataclass, field
from typing import Protocol

from agent.api_types.triggers import Trigger
from agent.llm.router import LLM
from agent.llm.models import SupportedModel
from agent.memory.dag.actions import MemoryAction
from agent.memory.dag.models import MemoryElement

# Query value types are pure data; they live in a lightweight module so lower layers can use
# them without an import cycle. Re-exported here so existing `from agent.memory.memory import
# MemoryQuery/MemoryQueries/QueryType` call sites keep working.
from agent.memory.queries import MemoryQueries, MemoryQuery, QueryType

from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.state import State

__all__ = [
    "QueryType",
    "MemoryQuery",
    "MemoryQueries",
    "RetrievedMemories",
    "IMemory",
]


@dataclass
class RetrievedMemories:
    """Result of a deliberate recall (`IMemory.query`).

    `elements` are the recalled memories themselves (for rendering / the action output).
    `actions` are the context actions retrieval produced; `reinforce` dispatches them to fold
    the recall into the persistent working context.
    """

    elements: list[MemoryElement] = field(default_factory=list)
    actions: list[MemoryAction] = field(default_factory=list)


class IMemory(Protocol):
    def query(
        self,
        memory_queries: MemoryQueries,
        llm: LLM,
        model: SupportedModel,
    ) -> RetrievedMemories:
        """Pure read: retrieve memories relevant to the queries. Does NOT mutate context."""
        ...

    def reinforce(
        self,
        retrieved: RetrievedMemories,
        budget: int,
        llm: LLM,
        model: SupportedModel,
    ) -> None:
        """Fold a recall's results into the persistent working context, then prune to budget."""
        ...

    def prune(
        self,
        budget: int,
        llm: LLM,
        model: SupportedModel,
    ) -> None:
        """Per-turn maintenance: token decay + prune to budget (no retrieval)."""
        ...

    def get_formatted_context(self) -> str:
        """Pure read of the current accumulated working context."""
        ...

    def store(
        self,
        trigger_history_entry: TriggerHistoryEntry,
        state: State,
        llm: LLM,
        model: SupportedModel,
    ) -> None: ...
