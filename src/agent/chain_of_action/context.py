"""
Execution context for action sequences.
"""

from typing import List
from pydantic import BaseModel, Field

from agent.chain_of_action.action_types import ActionType

from .trigger import BaseTrigger
from .action_result import ActionResult
from .trigger_history import TriggerHistoryEntry


class ExecutionContext(BaseModel):
    """Context information for action execution"""

    trigger: BaseTrigger
    situation_analysis: str
    completed_actions: List[ActionResult] = Field(default_factory=list)
    session_id: str
    relevant_memories: List[TriggerHistoryEntry] = Field(default_factory=list)

    def add_completed_action(self, result: ActionResult):
        """Add a completed action to the context"""
        self.completed_actions.append(result)

    def get_thoughts_summary(self) -> str:
        """Get summary of all THINK action results"""
        thoughts = [
            r.result_summary
            for r in self.completed_actions
            if r.action == ActionType.THINK and r.success
        ]
        return "\n".join(thoughts) if thoughts else "No thoughts yet"

    def get_completed_actions_summary(self) -> str:
        """Get summary of all completed actions for context"""
        if not self.completed_actions:
            return "No prior actions taken yet."

        # Use the same rich diary formatting that trigger history uses
        from agent.chain_of_action.prompts import _format_action_for_diary
        
        summaries = []
        for result in self.completed_actions:
            formatted_action = _format_action_for_diary(result)
            summaries.append(formatted_action)
        return "\n".join(summaries)
