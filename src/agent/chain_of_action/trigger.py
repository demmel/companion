"""
Trigger system for initiating action sequences.
"""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Union, Annotated, Literal


class BaseTriger(BaseModel):
    """Base class for all trigger events"""

    type: str
    timestamp: datetime = Field(default_factory=datetime.now)


class UserInputTrigger(BaseTriger):
    """Trigger caused by user input"""

    type: Literal["user_input"] = "user_input"
    content: str
    user_name: str = "User"  # Name of the person speaking


# Create discriminated union for proper polymorphic serialization
Trigger = UserInputTrigger


def format_trigger_for_prompt(trigger: BaseTriger) -> str:
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
            f"Trigger formatting not implemented for type: {trigger.type}"
        )
