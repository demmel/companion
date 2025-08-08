"""
Execution context for action sequences.
"""

from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field

from .trigger import TriggerEvent
from .action_types import ActionType

TMetadata = TypeVar("TMetadata", bound=BaseModel | None)


class ActionResult(BaseModel, Generic[TMetadata]):
    """Result of executing an action"""

    action: ActionType
    result_summary: str
    context_given: str
    duration_ms: float
    metadata: TMetadata
    success: bool = True
    error: str = ""


class ExecutionContext(BaseModel):
    """Context information for action execution"""

    trigger: TriggerEvent
    completed_actions: List[ActionResult] = Field(default_factory=list)
    session_id: str

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

        summaries = []
        for result in self.completed_actions:
            status = "✓" if result.success else "✗"
            summaries.append(
                f"{status} {result.action.value.upper()}: {result.result_summary[:100]}..."
            )
        return "\n".join(summaries)
