"""
UPDATE_MOOD action implementation.
"""

import logging
from typing import Type

from pydantic import BaseModel, Field

from agent.chain_of_action.context import ExecutionContext

from ..action_types import ActionType
from ..base_action import BaseAction
from ..base_action_data import (
    ActionOutput,
    ActionResult,
    ActionSuccessResult,
)

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


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


class UpdateMoodAction(BaseAction[UpdateMoodInput, UpdateMoodOutput]):
    """Update the agent's mood based on the current situation"""

    action_type = ActionType.UPDATE_MOOD

    @classmethod
    def get_action_description(cls) -> str:
        return "Update my current mood to reflect how I'm feeling"

    @classmethod
    def get_input_type(cls) -> Type[UpdateMoodInput]:
        return UpdateMoodInput

    def execute(
        self,
        action_input: UpdateMoodInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        progress_callback,
    ) -> ActionResult[UpdateMoodOutput]:
        logger.debug("=== UPDATE_MOOD ACTION ===")
        logger.debug(f"NEW MOOD: {action_input.new_mood}")
        logger.debug(f"INTENSITY: {action_input.intensity}")
        logger.debug(f"REASON: {action_input.reason}")

        old_mood = state.current_mood
        old_intensity = state.mood_intensity

        # Set the new mood to the absolute state provided
        state.current_mood = action_input.new_mood
        state.mood_intensity = action_input.intensity

        return ActionSuccessResult(
            content=UpdateMoodOutput(
                old_mood=old_mood,
                old_intensity=old_intensity,
                new_mood=state.current_mood,
                new_intensity=state.mood_intensity,
                reason=action_input.reason,
            )
        )
