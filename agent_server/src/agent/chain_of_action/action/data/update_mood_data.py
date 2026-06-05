"""UPDATE_MOOD action data types (input/output/record)."""

from typing import Literal

from pydantic import BaseModel, Field

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData


class UpdateMoodInput(BaseModel):
    """Input for UPDATE_MOOD action"""

    reason: str = Field(description="Why I'm feeling this way")
    new_mood: str = Field(
        description="My new mood described as an absolute state (not comparative)"
    )
    intensity: str = Field(description="Intensity of the new mood")


class UpdateMoodOutput(ActionOutput):
    """Output for UPDATE_MOOD action"""

    old_mood: str
    old_intensity: str
    new_mood: str
    new_intensity: str
    reason: str

    def result_summary(self):
        return f"Updated mood from '{self.old_mood} ({self.old_intensity})' to '{self.new_mood} ({self.new_intensity})' because {self.reason}"


class UpdateMoodActionData(BaseActionData[UpdateMoodInput, UpdateMoodOutput]):
    type: Literal[ActionType.UPDATE_MOOD] = ActionType.UPDATE_MOOD
