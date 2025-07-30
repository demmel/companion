"""
Action registry for discovering and managing available actions.
"""

from typing import Dict, List, Type, TYPE_CHECKING
import logging

from .action_types import ActionType
from .base_action import BaseAction
from .actions import ThinkAction

if TYPE_CHECKING:
    from agent.state import State
    from agent.conversation_history import ConversationHistory
    from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class ActionRegistry:
    """Registry for all available actions"""
    
    def __init__(self):
        self._actions: Dict[ActionType, Type[BaseAction]] = {}
        self._register_default_actions()
    
    def _register_default_actions(self):
        """Register the core actions"""
        self.register(ThinkAction)
        # Future actions will be registered here as they're implemented
        # self.register(SpeakAction)
        # self.register(UpdateMoodAction)
        # etc.
    
    def register(self, action_class: Type[BaseAction]):
        """Register an action class"""
        self._actions[action_class.action_type] = action_class
        logger.debug(f"Registered action: {action_class.action_type}")
    
    def get_action(self, action_type: ActionType) -> Type[BaseAction]:
        """Get action class by type"""
        if action_type not in self._actions:
            raise ValueError(f"Action type not registered: {action_type}")
        return self._actions[action_type]
    
    def get_available_actions(self) -> List[ActionType]:
        """Get list of all available action types"""
        return list(self._actions.keys())
    
    def get_action_descriptions(self) -> Dict[ActionType, str]:
        """Get descriptions of all available actions for planning prompts"""
        descriptions = {}
        for action_type, action_class in self._actions.items():
            descriptions[action_type] = action_class.get_action_description()
        return descriptions
    
    def get_context_descriptions(self) -> Dict[ActionType, str]:
        """Get context descriptions for all actions for planning prompts"""
        descriptions = {}
        for action_type, action_class in self._actions.items():
            descriptions[action_type] = action_class.get_context_description()
        return descriptions
    
    def create_action(self, action_type: ActionType) -> BaseAction:
        """Create an action instance for the given action type"""
        action_class = self.get_action(action_type)
        return action_class()