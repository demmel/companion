"""
UPDATE_MOOD action implementation.
"""

import time
import logging
from typing import Type

from pydantic import BaseModel, Field

from agent.chain_of_action.trigger_history import TriggerHistory

from ..action_types import ActionType
from ..base_action import BaseAction
from ..context import ActionResult, ExecutionContext
from ..action_plan import ActionPlan

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class UpdateMoodInput(BaseModel):
    """Input for UPDATE_MOOD action"""

    new_mood: str = Field(
        description="My new mood described as an absolute state (not comparative)"
    )
    intensity: str = Field(description="Intensity of the new mood")
    reason: str = Field(description="Why I'm feeling this way")


class UpdateMoodAction(BaseAction[UpdateMoodInput, None]):
    """Update the agent's mood based on the current situation"""

    action_type = ActionType.UPDATE_MOOD

    @classmethod
    def get_action_description(cls) -> str:
        return "Update my current mood to reflect how I'm feeling"

    @classmethod
    def get_context_description(cls) -> str:
        return "Rich description of my new mood - how I'm feeling and why"

    @classmethod
    def get_input_type(cls) -> Type[UpdateMoodInput]:
        return UpdateMoodInput

    def execute(
        self,
        action_input: UpdateMoodInput,
        context: ExecutionContext,
        state: State,
        trigger_history: TriggerHistory,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult:
        start_time = time.time()

        logger.debug("=== UPDATE_MOOD ACTION ===")
        logger.debug(f"NEW MOOD: {action_input.new_mood}")
        logger.debug(f"INTENSITY: {action_input.intensity}")
        logger.debug(f"REASON: {action_input.reason}")

        old_mood = state.current_mood
        old_intensity = state.mood_intensity

        # Set the new mood to the absolute state provided
        state.current_mood = action_input.new_mood
        state.mood_intensity = action_input.intensity

        duration_ms = (time.time() - start_time) * 1000

        result_summary = f"Mood updated from '{old_mood} ({old_intensity})' to '{state.current_mood} ({state.mood_intensity})' because {action_input.reason}"

        return ActionResult(
            action=ActionType.UPDATE_MOOD,
            result_summary=result_summary,
            context_given=f"new_mood: {action_input.new_mood}, reason: {action_input.reason}",
            duration_ms=duration_ms,
            success=True,
            metadata=None,  # No additional metadata needed
        )
