"""
Action planning structures.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

from .action.action_types import ActionType


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
    sequence_plan: str = Field(
        description="Brief description of what I plan to accomplish in this sequence (e.g., 'respond to user's greeting and wait for reaction', 'think through the problem then share my analysis', 'update my appearance to match my mood then speak'). Keep it simple and focused."
    )
    dependency_analysis: str = Field(
        description="Analysis of action dependencies: (1) How should actions in this round be ordered so later actions can use earlier results, and (2) Which actions should wait for the next planning round so I can provide better context from the results of the current round?"
    )
    wait_decision: str = Field(
        description="Should this sequence end with a wait action? If I want to see the user's reaction, need external input, or have completed my response, I should end with wait. If I need to use the results of actions in this sequence to plan better follow-up actions, I should NOT end with wait."
    )
    actions: List[ActionPlan] = Field(
        description="The actions I want to take this round, in order"
    )
