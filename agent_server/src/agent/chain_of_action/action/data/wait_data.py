"""WAIT action data types (input/output/record)."""

from typing import Literal

from pydantic import BaseModel, Field

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData


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


class WaitActionData(BaseActionData[WaitInput, WaitOutput]):
    type: Literal[ActionType.WAIT] = ActionType.WAIT
