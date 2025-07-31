"""
UPDATE_MOOD action implementation.
"""

import time
import logging

from ..action_types import ActionType
from ..base_action import BaseAction
from ..context import ActionResult, ExecutionContext
from ..action_plan import ActionPlan

from agent.state import State
from agent.conversation_history import ConversationHistory
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class UpdateMoodAction(BaseAction):
    """Update the agent's mood based on the current situation"""

    action_type = ActionType.UPDATE_MOOD

    @classmethod
    def get_action_description(cls) -> str:
        return "Update my current mood to reflect how I'm feeling"

    @classmethod
    def get_context_description(cls) -> str:
        return "Rich description of my new mood - how I'm feeling and why"

    def execute(
        self,
        action_plan: ActionPlan,
        context: ExecutionContext,
        state: State,
        conversation_history: ConversationHistory,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult:
        start_time = time.time()

        logger.debug("=== UPDATE_MOOD ACTION ===")
        logger.debug(f"NEW MOOD: {action_plan.context}")

        old_mood = state.current_mood

        # Set the new mood to whatever the planner specified
        state.current_mood = action_plan.context

        duration_ms = (time.time() - start_time) * 1000

        result_summary = f"Mood updated from '{old_mood}' to '{state.current_mood}'"

        return ActionResult(
            action=ActionType.UPDATE_MOOD,
            result_summary=result_summary,
            context_given=action_plan.context,
            duration_ms=duration_ms,
            success=True,
        )
