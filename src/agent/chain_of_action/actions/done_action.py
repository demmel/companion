"""
WAIT action implementation.
"""

import time
import logging
from typing import Type

from pydantic import BaseModel, Field

from agent.chain_of_action.trigger_history import TriggerHistory

from ..action_types import ActionType
from ..base_action import BaseAction
from ..action_result import ActionResult
from ..context import ExecutionContext
from ..action_plan import ActionPlan

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class WaitInput(BaseModel):
    """Input for WAIT action"""

    reason: str = Field(
        description="Brief reflection on what I've accomplished and why I'm ready for whatever comes next"
    )


class WaitAction(BaseAction[WaitInput, None]):
    """Wait for something else to happen"""

    action_type = ActionType.WAIT

    @classmethod
    def get_action_description(cls) -> str:
        return "Wait for something else to happen"

    @classmethod
    def get_context_description(cls) -> str:
        return "Brief reflection on what I've accomplished and why I'm ready for whatever comes next"

    @classmethod
    def get_input_type(cls) -> Type[WaitInput]:
        return WaitInput

    def execute(
        self,
        action_input: WaitInput,
        context: ExecutionContext,
        state: State,
        trigger_history: TriggerHistory,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult:
        start_time = time.time()

        logger.debug("=== DONE ACTION ===")
        logger.debug(f"REASON: {action_input.reason}")
        logger.debug("Agent has signaled completion")

        # DONE action doesn't need LLM call - it just signals completion
        duration_ms = (time.time() - start_time) * 1000

        return ActionResult(
            action=ActionType.WAIT,
            result_summary=f"Waiting for something else to happen. Reason: {action_input.reason}",
            context_given=action_input.reason,
            duration_ms=duration_ms,
            success=True,
            metadata=None,  # No additional metadata needed
        )
