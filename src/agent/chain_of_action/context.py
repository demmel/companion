"""
Execution context for action sequences.
"""

from typing import List
from pydantic import BaseModel, Field

from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.action.base_action_data import BaseActionData


from .trigger import BaseTrigger


class ExecutionContext(BaseModel):
    """Context information for action execution"""

    trigger: BaseTrigger
    situation_analysis: str
    completed_actions: List[BaseActionData] = Field(default_factory=list)
    session_id: str

    def add_completed_action(self, result: BaseActionData):
        """Add a completed action to the context"""
        self.completed_actions.append(result)

    def get_thoughts_summary(self) -> str:
        """Get summary of all THINK action results"""
        thoughts = [
            r.result.content.result_summary()
            for r in self.completed_actions
            if r.type == ActionType.THINK and r.result.type == "success"
        ]
        return "\n".join(thoughts) if thoughts else "No thoughts yet"
