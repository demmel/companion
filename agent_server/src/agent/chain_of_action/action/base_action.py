"""
Base action classes.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel

from .action_types import ActionType
from .base_action_data import ActionOutput, ActionResult
from ..context import ExecutionContext

from agent.state import State
from agent.llm import LLM, SupportedModel

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=ActionOutput)
TAction = TypeVar("TAction", bound="BaseAction")


# Registry of action type -> action class, populated by @register_action at import time.
# This replaces the old hand-maintained ActionRegistry._register_default_actions().
_ACTION_REGISTRY: dict[ActionType, type["BaseAction"]] = {}


def register_action(
    action_type: ActionType,
) -> Callable[[type[TAction]], type[TAction]]:
    """Register an action class under its ActionType.

    The ActionType lives in exactly one place: this decorator call, co-located with
    the action class. The decorator returns the class unchanged so static analysis
    treats the action exactly as before.
    """

    def deco(cls: type[TAction]) -> type[TAction]:
        _ACTION_REGISTRY[action_type] = cls
        return cls

    return deco


class BaseAction(ABC, Generic[TInput, TOutput]):
    """Base class for all actions"""

    @classmethod
    @abstractmethod
    def get_action_description(cls) -> str:
        """What this action does"""
        pass

    @classmethod
    def get_input_type(cls) -> type[TInput]:
        """Get the Pydantic model class for this action's input.

        Derived from the generic parameter (`BaseAction[Input, Output]`), so concrete
        actions never have to declare it.
        """
        for base in cls.__orig_bases__:
            origin = get_origin(base)
            if origin is not None and isinstance(origin, type) and issubclass(origin, BaseAction):
                return get_args(base)[0]
        raise TypeError(
            f"{cls.__name__} must parameterize BaseAction[Input, Output] to derive its input type"
        )

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
