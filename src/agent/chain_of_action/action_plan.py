"""
Action planning structures.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

from .action_types import ActionType


class ActionPlan(BaseModel):
    """Plan for executing a single action"""

    reasoning: str = Field(description="My reasoning for why I am taking this action")
    action: ActionType
    input: Dict[str, Any] = Field(
        description="Structured input parameters for this action based on its input schema"
    )


class ActionSequence(BaseModel):
    """Sequence of actions to execute"""

    completed_actions_review: str = Field(
        description="Explicit list of what I've already done this turn before the current sequence and what each action accomplished"
    )
    dependency_analysis: str = Field(
        description="Analysis of action dependencies: (1) How should actions in this round be ordered so later actions can use earlier results, and (2) Which actions should wait for the next planning round so I can provide better context from the results of the current round?"
    )
    continuation_justification: str = Field(
        description="Clear explanation of what new value another sequence after this one would add, or why I should wait at the end of this sequence"
    )
    actions: List[ActionPlan] = Field(
        description="The actions I want to take this round, in order"
    )
