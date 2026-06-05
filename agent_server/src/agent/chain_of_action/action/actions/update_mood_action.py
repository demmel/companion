"""
UPDATE_MOOD action implementation.
"""

import logging

from agent.chain_of_action.context import ExecutionContext

from ..action_types import ActionType
from ..base_action import BaseAction, register_action
from ..base_action_data import (
    ActionResult,
    ActionSuccessResult,
)
from ..data.update_mood_data import (
    UpdateMoodInput,
    UpdateMoodOutput,
    UpdateMoodActionData,
)

from agent.state import State
from agent.llm import LLM

logger = logging.getLogger(__name__)


@register_action(ActionType.UPDATE_MOOD)
class UpdateMoodAction(BaseAction[UpdateMoodInput, UpdateMoodOutput]):
    """Update the agent's mood based on the current situation"""

    @classmethod
    def get_action_description(cls) -> str:
        return "Update my current mood to reflect how I'm feeling"

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

        output = UpdateMoodOutput(
            old_mood=state.current_mood,
            old_intensity=state.mood_intensity,
            new_mood=action_input.new_mood,
            new_intensity=action_input.intensity,
            reason=action_input.reason,
        )

        self.apply_state_change(state, action_input, output)
        return ActionSuccessResult(content=output)

    def apply_state_change(
        self,
        state: State,
        action_input: UpdateMoodInput,
        output: UpdateMoodOutput,
    ) -> None:
        """Apply mood changes from the output."""
        state.current_mood = output.new_mood
        state.mood_intensity = output.new_intensity
