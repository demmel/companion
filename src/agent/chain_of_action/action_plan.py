"""
Action planning structures.
"""

from typing import List
from pydantic import BaseModel, Field

from .action_types import ActionType


class ActionPlan(BaseModel):
    """Plan for executing a single action"""

    action: ActionType
    context: str = Field(description="Situational details this action should focus on")


class ActionSequence(BaseModel):
    """Sequence of actions to execute"""

    actions: List[ActionPlan] = Field(description="Actions in execution order")
    can_extend: bool = Field(description="Whether more actions can be added on-the-fly")
    reasoning: str = Field(description="Why this sequence was chosen")
