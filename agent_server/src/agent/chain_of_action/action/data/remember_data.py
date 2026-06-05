"""REMEMBER action data types (input/output/record).

This module is part of the data layer. It may import the pure memory *query* value types
(`agent.memory.queries`) — those are plain pydantic containers with no heavy imports — but must
NOT import `agent.memory.memory` (the implementation/interface), which would re-enter the action
data package and form an import cycle.
"""

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field

from agent.memory.queries import MemoryQuery

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData


class RememberInput(BaseModel):
    """Input for REMEMBER action"""

    reason: str = Field(
        description="Why I'm recalling right now - what I'm trying to ground myself in before responding"
    )
    queries: List[MemoryQuery] = Field(
        description="Explicit memory search queries to run. Write several diverse queries covering "
        "the different angles of what I need to recall. Each query has its own text, type "
        "(factual/emotional/causal/temporal/relationship/decision/pattern), importance weight, and "
        "reasoning, so I can target and weight what matters most."
    )


class RetrievedMemoryItem(BaseModel):
    """A single memory recalled by the REMEMBER action (structured for the UI)."""

    memory_id: str
    content: str
    timestamp: datetime
    confidence: str


class RememberOutput(ActionOutput):
    """Output for REMEMBER action"""

    memories: List[RetrievedMemoryItem]

    def result_summary(self) -> str:
        if not self.memories:
            return "No relevant memories found."
        lines = [
            f"- [{item.timestamp:%Y-%m-%d %H:%M}] ({item.confidence}) {item.content}"
            for item in self.memories
        ]
        return "Recalled memories:\n" + "\n".join(lines)


class RememberActionData(BaseActionData[RememberInput, RememberOutput]):
    type: Literal[ActionType.REMEMBER] = ActionType.REMEMBER
