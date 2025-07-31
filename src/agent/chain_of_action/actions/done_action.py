"""
DONE action implementation.
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


class DoneAction(BaseAction):
    """Signal completion of the action sequence"""

    action_type = ActionType.DONE

    @classmethod
    def get_action_description(cls) -> str:
        return "Signal that I am complete and ready to wait for the next trigger"

    @classmethod
    def get_context_description(cls) -> str:
        return "Reason why I feel complete - what I've accomplished or why no further actions are needed"

    def execute(
        self,
        action_plan: ActionPlan,
        context: ExecutionContext,
        state: "State",
        conversation_history: "ConversationHistory",
        llm: "LLM",
        model: "SupportedModel",
        progress_callback,
    ) -> ActionResult:
        start_time = time.time()

        logger.debug("=== DONE ACTION ===")
        logger.debug(f"REASON: {action_plan.context}")
        logger.debug("Agent has signaled completion")

        # DONE action doesn't need LLM call - it just signals completion
        duration_ms = (time.time() - start_time) * 1000

        return ActionResult(
            action=ActionType.DONE,
            result_summary=f"Completed sequence. Reason: {action_plan.context}",
            context_given=action_plan.context,
            duration_ms=duration_ms,
            success=True,
        )
