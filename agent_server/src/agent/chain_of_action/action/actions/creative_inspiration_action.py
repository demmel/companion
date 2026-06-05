"""
GET_CREATIVE_INSPIRATION action implementation.
"""

import logging

from agent.chain_of_action.context import ExecutionContext

from ..action_types import ActionType
from ..base_action import BaseAction, register_action
from ..base_action_data import (
    ActionFailureResult,
    ActionResult,
    ActionSuccessResult,
)
from ..data.creative_inspiration_data import (
    CreativeInspirationInput,
    CreativeInspirationOutput,
    CreativeInspirationActionData,
)

from agent.state import State
from agent.llm import LLM

logger = logging.getLogger(__name__)


@register_action(ActionType.GET_CREATIVE_INSPIRATION)
class CreativeInspirationAction(
    BaseAction[CreativeInspirationInput, CreativeInspirationOutput]
):
    """Get random words for creative inspiration"""

    @classmethod
    def get_action_description(cls) -> str:
        return "Get random words to spark creative ideas and new directions"

    def execute(
        self,
        action_input: CreativeInspirationInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        progress_callback,
    ) -> ActionResult[CreativeInspirationOutput]:
        try:
            # Import the function from prompts module
            from agent.chain_of_action.prompts import generate_random_inspiration_words

            # Generate random words
            words = generate_random_inspiration_words(
                count=action_input.count, seed=action_input.seed
            )

            return ActionSuccessResult(content=CreativeInspirationOutput(words=words))
        except Exception as e:
            return ActionFailureResult(
                error=f"Unexpected error during GET_CREATIVE_INSPIRATION action: {str(e)}"
            )
