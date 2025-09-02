"""
Action result definitions.
"""

from abc import abstractmethod
from typing import Generic, TypeVar
from typing_extensions import Literal
from pydantic import BaseModel

from agent.chain_of_action.action.action_types import ActionType


class ActionOutput(BaseModel):

    @abstractmethod
    def result_summary(self) -> str:
        raise NotImplementedError("Subclasses must implement result_summary")


TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=ActionOutput)


class ActionSuccessResult(BaseModel, Generic[TOutput]):
    """Result of a successful action execution"""

    type: Literal["success"] = "success"
    content: TOutput


class ActionFailureResult(BaseModel):
    """Result of a failed action execution"""

    type: Literal["failure"] = "failure"
    error: str


type ActionResult[T: ActionOutput] = ActionSuccessResult[T] | ActionFailureResult


class BaseActionData(BaseModel, Generic[TInput, TOutput]):
    """Result of executing an action"""

    type: ActionType
    reasoning: str
    input: TInput
    result: ActionResult[TOutput]
    duration_ms: float
