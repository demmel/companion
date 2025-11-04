"""
Execution context for action sequences.
"""

from typing import List
from pydantic import BaseModel, Field

from agent.chain_of_action.action.base_action_data import BaseActionData
from agent.llm import SupportedModel

from .trigger import BaseTrigger
from .action_plan import ActionPlan


class ExecutionContext(BaseModel):
    """Context information for action execution"""

    trigger: BaseTrigger
    situation_analysis: str
    completed_actions: List[BaseActionData] = Field(default_factory=list)
    session_id: str
    agent_capabilities_knowledge_prompt: str
    planned_actions: List[ActionPlan] = Field(default_factory=list)
    current_action_index: int = 0

    # Models for action execution
    think_action_model: SupportedModel
    speak_action_model: SupportedModel
    visual_action_model: SupportedModel
    fetch_url_action_model: SupportedModel
    evaluate_priorities_action_model: SupportedModel

    def add_completed_action(self, result: BaseActionData):
        """Add a completed action to the context"""
        self.completed_actions.append(result)

    def get_thoughts_summary(self) -> str:
        """Get summary of all THINK action results"""
        from agent.chain_of_action.action.action_data import ThinkActionData

        thoughts = [
            r.result.content.result_summary()
            for r in self.completed_actions
            if isinstance(r, ThinkActionData) and r.result.type == "success"
        ]
        return "\n".join(thoughts) if thoughts else "No thoughts yet"
