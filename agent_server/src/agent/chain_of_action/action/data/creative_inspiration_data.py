"""GET_CREATIVE_INSPIRATION action data types (input/output/record)."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData


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


class CreativeInspirationActionData(
    BaseActionData[CreativeInspirationInput, CreativeInspirationOutput]
):
    type: Literal[ActionType.GET_CREATIVE_INSPIRATION] = (
        ActionType.GET_CREATIVE_INSPIRATION
    )
