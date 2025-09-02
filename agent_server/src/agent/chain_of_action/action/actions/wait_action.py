"""
WAIT action implementation.
"""

import logging
from typing import Type

from pydantic import BaseModel, Field

from ..action_types import ActionType
from ..base_action import BaseAction
from ..base_action_data import ActionOutput, ActionResult, ActionSuccessResult
from agent.chain_of_action.context import ExecutionContext

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class WaitInput(BaseModel):
    """Input for WAIT action"""

    reason: str = Field(
        description="Brief reflection on what I've accomplished and why I'm ready for whatever comes next"
    )


class WaitOutput(ActionOutput):
    """Output for WAIT action"""

    reason: str

    def result_summary(self) -> str:
        return f"Waiting for something else to happen. Reason: {self.reason}"


class WaitAction(BaseAction[WaitInput, WaitOutput]):
    """Wait for something else to happen"""

    action_type = ActionType.WAIT

    @classmethod
    def get_action_description(cls) -> str:
        return "Wait for something else to happen"

    @classmethod
    def get_input_type(cls) -> Type[WaitInput]:
        return WaitInput

    def execute(
        self,
        action_input: WaitInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult[WaitOutput]:
        logger.debug("=== DONE ACTION ===")
        logger.debug(f"REASON: {action_input.reason}")
        logger.debug("Agent has signaled completion")

        return ActionSuccessResult(content=WaitOutput(reason=action_input.reason))
