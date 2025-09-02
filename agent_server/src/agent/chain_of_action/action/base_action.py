"""
Base action classes.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Generic, TypeVar, Type

from pydantic import BaseModel

from .action_types import ActionType
from .base_action_data import ActionOutput, ActionResult
from ..context import ExecutionContext

from agent.state import State
from agent.llm import LLM, SupportedModel

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=ActionOutput)


class BaseAction(ABC, Generic[TInput, TOutput]):
    """Base class for all actions"""

    action_type: ActionType

    @classmethod
    @abstractmethod
    def get_action_description(cls) -> str:
        """What this action does"""
        pass

    @classmethod
    @abstractmethod
    def get_input_type(cls) -> Type[TInput]:
        """Get the Pydantic model class for this action's input"""
        pass

    @abstractmethod
    def execute(
        self,
        action_input: TInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
        progress_callback: Callable[[Any], None],
    ) -> ActionResult[TOutput]:
        """Execute the action and return result"""
        pass
