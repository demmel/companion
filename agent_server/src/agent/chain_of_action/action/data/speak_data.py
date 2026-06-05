"""SPEAK action data types (input/output/record)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData


class SpeakInput(BaseModel):
    """Input for SPEAK action"""

    intent: str = Field(
        description="The intent or high-level idea of what I want to communicate (e.g., 'express curiosity about their day', 'share excitement about the topic', 'ask for clarification') - NOT the actual words to say"
    )
    tone: Optional[str] = Field(
        default=None,
        description="The emotional tone or approach I want to use (optional)",
    )


class SpeakOutput(ActionOutput):
    """Output for SPEAK action"""

    response: str

    def result_summary(self) -> str:
        return self.response


class SpeakActionData(BaseActionData[SpeakInput, SpeakOutput]):
    type: Literal[ActionType.SPEAK] = ActionType.SPEAK
