"""
Trigger system for initiating action sequences.
"""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TriggerType(str, Enum):
    """Types of triggers that can initiate action sequences"""

    USER_INPUT = "user_input"
    # Future: SELF_REFLECTION, TIMER_BASED, etc.


class TriggerEvent(BaseModel):
    """Base class for all trigger events"""

    trigger_type: TriggerType
    timestamp: datetime = Field(default_factory=datetime.now)


class UserInputTrigger(TriggerEvent):
    """Trigger caused by user input"""

    trigger_type: TriggerType = TriggerType.USER_INPUT
    content: str
    user_name: str = "User"  # Name of the person speaking


def format_trigger_for_prompt(trigger: TriggerEvent) -> str:
    """Format the trigger with proper context about what happened"""
    if isinstance(trigger, UserInputTrigger):
        # For user input, show who spoke to the agent
        user_trigger = trigger  # Type: UserInputTrigger
        return f'{user_trigger.user_name} said to me: "{user_trigger.content}"'
    else:
        # Future trigger types would be formatted differently
        # e.g., "A tool finished executing: {result}"
        # e.g., "A timer went off: {timer_info}"
        # e.g., "I decided to reflect on: {topic}"
        raise NotImplementedError(
            f"Trigger formatting not implemented for type: {trigger.trigger_type}"
        )
