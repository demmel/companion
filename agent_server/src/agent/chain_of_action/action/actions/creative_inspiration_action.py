"""
GET_CREATIVE_INSPIRATION action implementation.
"""

import logging
from typing import Type, Optional, List

from pydantic import BaseModel, Field

from agent.chain_of_action.context import ExecutionContext

from ..action_types import ActionType
from ..base_action import BaseAction
from ..base_action_data import (
    ActionFailureResult,
    ActionOutput,
    ActionResult,
    ActionSuccessResult,
)

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class CreativeInspirationInput(BaseModel):
    """Input for GET_CREATIVE_INSPIRATION action"""

    count: int = Field(
        default=10,
        description="Number of random words to generate for inspiration",
    )
    seed: Optional[int] = Field(
        default=None, description="Optional seed for reproducible randomness"
    )


class CreativeInspirationOutput(ActionOutput):
    """Output for GET_CREATIVE_INSPIRATION action"""

    words: List[str]

    def result_summary(self) -> str:
        return f"Creative inspiration words: {', '.join(self.words)}"


class CreativeInspirationAction(
    BaseAction[CreativeInspirationInput, CreativeInspirationOutput]
):
    """Get random words for creative inspiration"""

    action_type = ActionType.GET_CREATIVE_INSPIRATION

    @classmethod
    def get_action_description(cls) -> str:
        return "Get random words to spark creative ideas and new directions"

    @classmethod
    def get_input_type(cls) -> Type[CreativeInspirationInput]:
        return CreativeInspirationInput

    def execute(
        self,
        action_input: CreativeInspirationInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
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
