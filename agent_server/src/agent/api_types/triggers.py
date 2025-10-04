from typing import Literal, assert_never
from agent.chain_of_action.trigger import Trigger as BackendTrigger
from pydantic import BaseModel


class UserInputTrigger(BaseModel):
    """DTO for user input triggers"""

    type: Literal["user_input"] = "user_input"
    content: str
    user_name: str
    timestamp: str
    image_urls: list[str] | None = None


class WakeupTrigger(BaseModel):
    """DTO for wakeup triggers"""

    type: Literal["wakeup"] = "wakeup"
    timestamp: str


# Discriminated union for all trigger types
Trigger = UserInputTrigger | WakeupTrigger


def convert_trigger_to_dto(trigger: BackendTrigger) -> Trigger:
    """Convert backend TriggerEvent to DTO"""
    from agent.chain_of_action.trigger import (
        UserInputTrigger as BackendUserInputTrigger,
        WakeupTrigger as BackendWakeupTrigger,
        BirthTrigger as BackendBirthTrigger,
    )

    match trigger:
        case BackendUserInputTrigger() | BackendBirthTrigger():
            # Convert file paths to URLs for client
            image_urls = None
            if trigger.image_paths:
                from pathlib import Path

                image_urls = [
                    f"/uploaded_images/{Path(path).name}"
                    for path in trigger.image_paths
                ]
            return UserInputTrigger(
                content=trigger.content,
                user_name=trigger.user_name,
                timestamp=trigger.timestamp.isoformat(),
                image_urls=image_urls,
            )
        case BackendWakeupTrigger():
            return WakeupTrigger(
                timestamp=trigger.timestamp.isoformat(),
            )
        case _:
            assert_never(trigger)
