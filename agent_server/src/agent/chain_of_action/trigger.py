"""
Trigger system for initiating action sequences.
"""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Union, Literal, Optional, List, NewType, assert_never

# Domain type for image file paths
ImageFilePath = NewType("ImageFilePath", str)
ImageFilePaths = Optional[List[ImageFilePath]]


class BaseTrigger(BaseModel):
    """Base class for all trigger events"""

    timestamp: datetime = Field(default_factory=datetime.now)

    def get_images(self) -> ImageFilePaths:
        """Get image file paths associated with this trigger, if any"""
        return None


class BirthTrigger(BaseTrigger):
    """Trigger for agent creation event"""

    type: Literal["birth"] = "birth"
    content: str
    user_name: str
    image_paths: ImageFilePaths = None  # Paths to shared images


class UserInputTrigger(BaseTrigger):
    """Trigger caused by user input"""

    type: Literal["user_input"] = "user_input"
    content: str
    user_name: str  # Name of the person speaking
    image_paths: ImageFilePaths = None  # Paths to shared images

    def get_images(self) -> ImageFilePaths:
        """Get image file paths associated with this trigger"""
        return self.image_paths


class WakeupTrigger(BaseTrigger):
    """Trigger for agent's autonomous reflection/continuation"""

    type: Literal["wakeup"] = "wakeup"


# Create discriminated union for proper polymorphic serialization
Trigger = UserInputTrigger | WakeupTrigger | BirthTrigger


def format_trigger_for_prompt(trigger: Trigger) -> str:
    """Format the trigger with proper context about what happened"""
    match trigger:
        case UserInputTrigger():
            user_trigger = trigger  # Type: UserInputTrigger
            base_text = f'{user_trigger.user_name} said to me: "{user_trigger.content}"'

            # Add image information if present
            if user_trigger.image_paths:
                image_count = len(user_trigger.image_paths)
                if image_count == 1:
                    base_text += f" (User shared an image)"
                else:
                    base_text += f" (User shared {image_count} images)"
            return base_text
        case WakeupTrigger():
            return "I continue to exist and have my own experiences during this quiet moment"
        case BirthTrigger():
            return "I was created and brought to life in this moment"
        case _:
            assert_never(trigger)
