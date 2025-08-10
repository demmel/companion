"""
Action planning structures.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

from .action_types import ActionType


class ActionPlan(BaseModel):
    """Plan for executing a single action"""

    action: ActionType
    input: Dict[str, Any] = Field(
        description="Structured input parameters for this action based on its input schema"
    )


class ActionSequence(BaseModel):
    """Sequence of actions to execute"""

    situation_analysis: str = Field(
        description="My analysis of the current situation and what it calls for"
    )
    actions: List[ActionPlan] = Field(
        description="The actions I want to take, in order"
    )
    reasoning: str = Field(description="My reasoning for why I chose this sequence")
