"""
Execution context for action sequences.
"""

from dataclasses import dataclass, field
from typing import List

from agent.chain_of_action.action.base_action_data import BaseActionData
from agent.llm import SupportedModel
from agent.memory.memory import IMemory

from .trigger import BaseTrigger
from .action_plan import ActionPlan


@dataclass
class ExecutionContext:
    """Context information for action execution.

    Purely transient, per-trigger object. It holds live service references (e.g. the
    memory manager) and is never serialized, so it is a plain dataclass rather than a
    pydantic model.
    """

    trigger: BaseTrigger
    situation_analysis: str
    session_id: str
    agent_capabilities_knowledge_prompt: str

    # Memory access for deliberate recall (REMEMBER action) and budget management
    memory: IMemory
    memory_token_budget: int
    memory_retrieval_model: SupportedModel

    # Models for action execution
    think_action_model: SupportedModel
    speak_action_model: SupportedModel
    visual_action_model: SupportedModel
    fetch_url_action_model: SupportedModel
    evaluate_priorities_action_model: SupportedModel

    completed_actions: List[BaseActionData] = field(default_factory=list)
    planned_actions: List[ActionPlan] = field(default_factory=list)
    current_action_index: int = 0

    def add_completed_action(self, result: BaseActionData):
        """Add a completed action to the context"""
        self.completed_actions.append(result)

    def get_thoughts_summary(self) -> str:
        """Get summary of all THINK action results"""
        from agent.chain_of_action.action.actions.think_action import ThinkActionData

        thoughts = [
            r.result.content.result_summary()
            for r in self.completed_actions
            if isinstance(r, ThinkActionData) and r.result.type == "success"
        ]
        return "\n".join(thoughts) if thoughts else "No thoughts yet"
