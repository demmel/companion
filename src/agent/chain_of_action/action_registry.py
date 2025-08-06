"""
Action registry for discovering and managing available actions.
"""

from typing import Dict, List, Type
import logging

from .action_types import ActionType
from .base_action import BaseAction
from .actions import (
    ThinkAction,
    WaitAction,
    SpeakAction,
    UpdateMoodAction,
    UpdateAppearanceAction,
)


logger = logging.getLogger(__name__)


class ActionRegistry:
    """Registry for all available actions"""

    def __init__(self):
        self._actions: Dict[ActionType, Type[BaseAction]] = {}
        self._register_default_actions()

    def _register_default_actions(self):
        """Register the core actions"""
        self.register(ThinkAction)
        self.register(WaitAction)
        self.register(SpeakAction)
        self.register(UpdateMoodAction)
        self.register(UpdateAppearanceAction)
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

    def get_available_actions_for_prompt(self) -> str:
        """Get formatted string of available actions for prompts"""
        # Format available actions for the prompt
        action_descriptions = self.get_action_descriptions()
        context_descriptions = self.get_context_descriptions()

        actions_info = []
        for action_type in self.get_available_actions():
            action_desc = action_descriptions[action_type]
            context_desc = context_descriptions[action_type]
            actions_info.append(f"- {action_type.value}: {action_desc}")
            actions_info.append(f"  Context needed: {context_desc}")

        actions_list = "\n".join(actions_info)

        return actions_list
