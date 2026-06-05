"""Visual state update action data types (inputs/outputs/records)."""

from typing import Literal

from pydantic import BaseModel, Field

from agent.types import ImageGenerationToolContent

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData


class UpdateAppearanceInput(BaseModel):
    """Input for UPDATE_APPEARANCE action"""

    reason: str = Field(description="Why I'm changing my appearance")
    change_description: str = Field(
        description="What specific aspects of my appearance should change and how. These should be specific and detailed."
    )


class UpdateEnvironmentInput(BaseModel):
    """Input for UPDATE_ENVIRONMENT action"""

    reason: str = Field(description="Why I'm changing my environment")
    change_description: str = Field(
        description="What specific aspects of my environment should change and how. These should be specific and detailed."
    )


class UpdateAppearanceOutput(ActionOutput):
    """Output for UPDATE_APPEARANCE action"""

    image_description: str
    old_appearance: str
    new_appearance: str
    reason: str
    image_result: ImageGenerationToolContent

    def result_summary(self) -> str:
        return f"Appearance updated: {self.new_appearance} (reason: {self.reason})"


class UpdateEnvironmentOutput(ActionOutput):
    """Output for UPDATE_ENVIRONMENT action"""

    image_description: str
    old_environment: str
    new_environment: str
    reason: str
    image_result: ImageGenerationToolContent

    def result_summary(self) -> str:
        return f"Environment updated: {self.new_environment} (reason: {self.reason})"


class UpdateAppearanceActionData(
    BaseActionData[UpdateAppearanceInput, UpdateAppearanceOutput]
):
    type: Literal[ActionType.UPDATE_APPEARANCE] = ActionType.UPDATE_APPEARANCE


class UpdateEnvironmentActionData(
    BaseActionData[UpdateEnvironmentInput, UpdateEnvironmentOutput]
):
    type: Literal[ActionType.UPDATE_ENVIRONMENT] = ActionType.UPDATE_ENVIRONMENT
