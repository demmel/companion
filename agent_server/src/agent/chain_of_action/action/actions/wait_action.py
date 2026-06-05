"""
WAIT action implementation.
"""

import logging

from ..action_types import ActionType
from ..base_action import BaseAction, register_action
from ..base_action_data import (
    ActionResult,
    ActionSuccessResult,
)
from ..data.wait_data import WaitInput, WaitOutput, WaitActionData
from agent.chain_of_action.context import ExecutionContext

from agent.state import State
from agent.llm import LLM

logger = logging.getLogger(__name__)


@register_action(ActionType.WAIT)
class WaitAction(BaseAction[WaitInput, WaitOutput]):
    """Wait for something else to happen"""

    @classmethod
    def get_action_description(cls) -> str:
        return "Wait for something else to happen"

    def execute(
        self,
        action_input: WaitInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        progress_callback,
    ) -> ActionResult[WaitOutput]:
        logger.debug("=== DONE ACTION ===")
        logger.debug(f"REASON: {action_input.reason}")
        logger.debug("Agent has signaled completion")

        return ActionSuccessResult(content=WaitOutput(reason=action_input.reason))
