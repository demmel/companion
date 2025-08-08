"""
Base action classes.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Generic, TypeVar

from pydantic import BaseModel

from agent.chain_of_action.trigger_history import TriggerHistory

from .action_types import ActionType
from .context import ActionResult, ExecutionContext
from .action_plan import ActionPlan

from agent.state import State
from agent.llm import LLM, SupportedModel

TMetadata = TypeVar("TMetadata", bound=BaseModel | None)


class BaseAction(ABC, Generic[TMetadata]):
    """Base class for all actions"""

    action_type: ActionType

    @classmethod
    @abstractmethod
    def get_action_description(cls) -> str:
        """What this action does"""
        pass

    @classmethod
    @abstractmethod
    def get_context_description(cls) -> str:
        """What context this action needs when planned"""
        pass

    @abstractmethod
    def execute(
        self,
        action_plan: ActionPlan,
        context: ExecutionContext,
        state: State,
        trigger_history: TriggerHistory,
        llm: LLM,
        model: SupportedModel,
        progress_callback: Callable[[Any], None],
    ) -> ActionResult[TMetadata]:
        """Execute the action and return result"""
        pass

    def build_agent_state_description(self, state: State) -> str:
        """Build fresh state description when needed"""
        from agent.state import build_agent_state_description

        return build_agent_state_description(state)
