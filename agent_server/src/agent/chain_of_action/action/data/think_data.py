"""THINK action data types (input/output/record)."""

from typing import Literal

from pydantic import BaseModel, Field

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData


class ThinkInput(BaseModel):
    """Input for THINK action"""

    focus: str = Field(
        description="Specific topic or question to think through (e.g., 'How to best support them during their stressful week', 'Whether to share my creative idea or wait', 'What this change means for my priorities') - NOT generic like 'emotional elements'"
    )


class ThinkOutput(ActionOutput):
    """Output for THINK action"""

    thoughts: str

    def result_summary(self) -> str:
        return self.thoughts


class ThinkActionData(BaseActionData[ThinkInput, ThinkOutput]):
    type: Literal[ActionType.THINK] = ActionType.THINK
