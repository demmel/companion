"""
Action result definitions.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Generic, TypeVar
from typing_extensions import Literal
from pydantic import BaseModel, ConfigDict

from .action_types import ActionType


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


# Concrete per-action data classes (the `FooActionData` defined in each action module)
# register themselves here as they are defined, so the discriminated `ActionData` union
# can be assembled from the registry instead of being hand-maintained.
_ACTION_DATA_TYPES: list[type["BaseActionData"]] = []


class BaseActionData(BaseModel, Generic[TInput, TOutput]):
    """Record of an executed action.

    Concrete actions declare a tiny ``FooActionData(BaseActionData[FooInput, FooOutput])``
    co-located in their module, pinning ``type`` to a ``Literal`` discriminator. Those
    classes auto-collect into ``_ACTION_DATA_TYPES`` so the union and the
    type->class lookup are derived rather than hand-written.
    """

    model_config = ConfigDict(extra="ignore")

    type: ActionType
    reasoning: str
    input: TInput
    result: ActionResult[TOutput]
    duration_ms: float
    start_timestamp: datetime

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Only collect concrete per-action classes (those that pin `type` to a specific
        # Literal default). Pydantic's internal generic parameterizations don't redefine
        # `type` in their own namespace, so they're skipped.
        if "type" in cls.__dict__:
            _ACTION_DATA_TYPES.append(cls)
