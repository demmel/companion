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

    @classmethod
    def can_perform(cls, state: State) -> bool:
        """Whether this action is currently available"""
        return True  # Default: always available

    @abstractmethod
    def execute(
        self,
        action_input: TInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        progress_callback: Callable[[Any], None],
    ) -> ActionResult[TOutput]:
        """Execute the action and return result"""
        pass

    def apply_state_change(
        self,
        state: State,
        action_input: TInput,
        output: TOutput,
    ) -> None:
        """
        Apply state changes based on action input and output.

        Mutates state in place. This separates state mutations from side effects,
        enabling deterministic replay of conversations without re-executing LLM
        calls or other side effects.

        The default implementation does nothing. Actions that modify state should
        override this method.

        This method should be called by execute() after creating the output,
        ensuring the same logic is used during execution and replay.

        Args:
            state: The state to mutate
            action_input: The input that was passed to execute()
            output: The successful output (caller should check for failure first)
        """
        pass
