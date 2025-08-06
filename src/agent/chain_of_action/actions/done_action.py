"""
WAIT action implementation.
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


class WaitAction(BaseAction):
    """Wait for something else to happen"""

    action_type = ActionType.WAIT

    @classmethod
    def get_action_description(cls) -> str:
        return "Wait for something else to happen"

    @classmethod
    def get_context_description(cls) -> str:
        return "Brief reflection on what I've accomplished and why I'm ready for whatever comes next"

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

        logger.debug("=== DONE ACTION ===")
        logger.debug(f"REASON: {action_plan.context}")
        logger.debug("Agent has signaled completion")

        # DONE action doesn't need LLM call - it just signals completion
        duration_ms = (time.time() - start_time) * 1000

        return ActionResult(
            action=ActionType.WAIT,
            result_summary=f"Waiting for something else to happen. Reason: {action_plan.context}",
            context_given=action_plan.context,
            duration_ms=duration_ms,
            success=True,
        )
