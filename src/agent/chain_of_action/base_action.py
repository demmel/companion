"""
Base action classes.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any

from .action_types import ActionType
from .context import ActionResult, ExecutionContext
from .action_plan import ActionPlan

from agent.state import State
from agent.conversation_history import ConversationHistory
from agent.llm import LLM, SupportedModel


class BaseAction(ABC):
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
        state: "State",
        conversation_history: "ConversationHistory",
        llm: "LLM",
        model: "SupportedModel",
        progress_callback: Callable[[Any], None],
    ) -> ActionResult:
        """Execute the action and return result"""
        pass

    def serialize_conversation_history(
        self, conversation_history: "ConversationHistory"
    ) -> str:
        """Serialize conversation history when needed for prompts"""
        # This will need to be implemented based on ConversationHistory interface
        return str(conversation_history)

    def build_agent_state_description(self, state: "State") -> str:
        """Build fresh state description when needed"""
        from agent.state import build_agent_state_description

        return build_agent_state_description(state)
