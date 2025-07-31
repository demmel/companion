"""
Chain of Action - Action-based reasoning system

This package implements the action-based architecture that replaces
the monolithic reasoning loop with modular, composable actions.
"""

from .action_types import ActionType
from .trigger import TriggerEvent, UserInputTrigger
from .context import ExecutionContext
from .action_plan import ActionPlan, ActionSequence
from .action_registry import ActionRegistry
from .reasoning_loop import ActionBasedReasoningLoop

__all__ = [
    "ActionType",
    "TriggerEvent",
    "UserInputTrigger",
    "ExecutionContext",
    "ActionPlan",
    "ActionSequence",
    "ActionRegistry",
    "ActionBasedReasoningLoop",
]
