"""
Action registry for discovering and managing available actions.
"""

from typing import Dict, List, Type
import logging

from agent.chain_of_action.action.actions.fetch_url_action import FetchUrlAction
from agent.chain_of_action.action.actions.priority_actions import (
    AddPriorityAction,
    RemovePriorityAction,
)
from agent.chain_of_action.action.actions.search_web_action import SearchWebAction
from agent.chain_of_action.action.actions.speak_action import SpeakAction
from agent.chain_of_action.action.actions.think_action import ThinkAction
from agent.chain_of_action.action.actions.update_appearance_action import (
    UpdateAppearanceAction,
)
from agent.chain_of_action.action.actions.update_environment_action import (
    UpdateEnvironmentAction,
)
from agent.chain_of_action.action.actions.update_mood_action import UpdateMoodAction
from agent.chain_of_action.action.actions.wait_action import WaitAction

from .action.action_types import ActionType
from .action.base_action import BaseAction


logger = logging.getLogger(__name__)


class ActionRegistry:
    """Registry for all available actions"""

    def __init__(self, enable_image_generation: bool = True):
        self._actions: Dict[ActionType, Type[BaseAction]] = {}
        self.enable_image_generation = enable_image_generation
        self._register_default_actions()

    def _register_default_actions(self):
        """Register the core actions"""
        self.register(ThinkAction)
        self.register(WaitAction)
        self.register(SpeakAction)
        self.register(UpdateMoodAction)
        self.register(UpdateAppearanceAction)
        self.register(UpdateEnvironmentAction)
        self.register(FetchUrlAction)
        self.register(SearchWebAction)
        self.register(AddPriorityAction)
        self.register(RemovePriorityAction)
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

    def create_action(self, action_type: ActionType) -> BaseAction:
        """Create an action instance for the given action type"""
        action_class = self.get_action(action_type)

        # Pass enable_image_generation flag to UpdateAppearanceAction and UpdateEnvironmentAction
        if action_class == UpdateAppearanceAction:
            return UpdateAppearanceAction(
                enable_image_generation=self.enable_image_generation
            )
        elif action_class == UpdateEnvironmentAction:
            return UpdateEnvironmentAction(
                enable_image_generation=self.enable_image_generation
            )
        else:
            return action_class()

    def get_available_actions_for_prompt(self) -> str:
        """Get formatted string of available actions with input schemas for prompts"""
        action_descriptions = self.get_action_descriptions()

        actions_info = []
        for action_type in self.get_available_actions():
            action_desc = action_descriptions[action_type]
            action_class = self.get_action(action_type)

            # Get the input schema
            input_type = action_class.get_input_type()
            schema = input_type.model_json_schema()

            actions_info.append(f"- {action_type.value}: {action_desc}")

            # Add input parameters from schema
            if "properties" in schema:
                actions_info.append("  Input parameters:")
                for field_name, field_info in schema["properties"].items():
                    description = field_info.get("description", "No description")
                    field_type = field_info.get("type", "unknown")
                    required = field_name in schema.get("required", [])
                    req_str = " (required)" if required else " (optional)"
                    actions_info.append(
                        f"    - {field_name} ({field_type}){req_str}: {description}"
                    )

        actions_list = "\n".join(actions_info)
        return actions_list

    def get_system_knowledge_for_context(self) -> str:
        """Get system knowledge description for situational awareness (not execution)"""
        action_descriptions = self.get_action_descriptions()

        capabilities = []
        for action_type in self.get_available_actions():
            description = action_descriptions[action_type]
            capabilities.append(f"- {action_type.value}: {description}")

        capabilities_text = "\n".join(capabilities)

        return f"""What I can actually do in this system:
{capabilities_text}

I exist digitally, so I can't physically prepare food, move objects, or do anything in the physical world. This is just for my own understanding - I'm not planning actions right now."""
