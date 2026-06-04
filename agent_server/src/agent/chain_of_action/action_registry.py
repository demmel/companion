"""
Action registry for discovering and managing available actions.
"""

from typing import Dict, List, Type
import logging

# Importing the actions package runs every action module, whose @register_action
# decorators populate _ACTION_REGISTRY. This is the single import site that makes the
# registry the source of truth for available actions.
from agent.chain_of_action.action import actions as _actions  # noqa: F401
from agent.state import State

from .action.action_types import ActionType
from .action.base_action import BaseAction, _ACTION_REGISTRY


logger = logging.getLogger(__name__)


class ActionRegistry:
    """Registry for all available actions, backed by the @register_action registry."""

    def __init__(self, enable_image_generation: bool = True):
        self._actions: Dict[ActionType, Type[BaseAction]] = dict(_ACTION_REGISTRY)
        self.enable_image_generation = enable_image_generation

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
        from agent.chain_of_action.action.actions.visual_actions import (
            UpdateAppearanceAction,
            UpdateEnvironmentAction,
        )

        # Pass enable_image_generation flag to the visual actions, which need it.
        if action_type == ActionType.UPDATE_APPEARANCE:
            return UpdateAppearanceAction(
                enable_image_generation=self.enable_image_generation
            )
        if action_type == ActionType.UPDATE_ENVIRONMENT:
            return UpdateEnvironmentAction(
                enable_image_generation=self.enable_image_generation
            )
        return self.get_action(action_type)()

    def get_available_actions_for_state(self, state: State) -> List[ActionType]:
        """Get actions that can be performed in current state"""
        return [
            action_type
            for action_type in self.get_available_actions()
            if self.get_action(action_type).can_perform(state)
        ]

    def _format_field_info(
        self,
        field_name: str,
        field_info: dict,
        schema: dict,
        indent: str,
        required: bool,
    ) -> List[str]:
        """Recursively format field information, expanding nested objects"""
        lines = []
        description = field_info.get("description", "No description")
        field_type = field_info.get("type", "unknown")
        req_str = " (required)" if required else " (optional)"

        # Check if this is a reference to another definition
        if "$ref" in field_info:
            ref_path = field_info["$ref"].split("/")[-1]  # Get the definition name
            if "$defs" in schema and ref_path in schema["$defs"]:
                ref_schema = schema["$defs"][ref_path]
                field_type = "object"
                lines.append(
                    f"{indent}- {field_name} ({field_type}){req_str}: {description}"
                )
                # Expand the nested object
                if "properties" in ref_schema:
                    for nested_field, nested_info in ref_schema["properties"].items():
                        nested_required = nested_field in ref_schema.get("required", [])
                        lines.extend(
                            self._format_field_info(
                                nested_field,
                                nested_info,
                                schema,
                                indent + "  ",
                                nested_required,
                            )
                        )
                return lines

        # Check for enum/literal types
        if "enum" in field_info:
            enum_values = ", ".join([f"'{v}'" for v in field_info["enum"]])
            lines.append(
                f"{indent}- {field_name} ({field_type}, one of: {enum_values}){req_str}: {description}"
            )
        elif "anyOf" in field_info:
            # Handle Optional types and unions
            types = []
            for option in field_info["anyOf"]:
                if option.get("type") == "null":
                    continue
                if "enum" in option:
                    enum_values = ", ".join([f"'{v}'" for v in option["enum"]])
                    types.append(f"one of: {enum_values}")
                else:
                    types.append(option.get("type", "unknown"))
            type_str = " or ".join(types) if types else "unknown"
            lines.append(f"{indent}- {field_name} ({type_str}){req_str}: {description}")
        else:
            lines.append(
                f"{indent}- {field_name} ({field_type}){req_str}: {description}"
            )

        return lines

    def _format_action_for_prompt(self, action_type: ActionType) -> List[str]:
        """Format a single action for prompt display"""
        action_class = self.get_action(action_type)
        action_desc = action_class.get_action_description()

        input_type = action_class.get_input_type()
        schema = input_type.model_json_schema()

        lines = [f"- {action_type.value}: {action_desc}"]

        if "properties" in schema:
            lines.append("  Input parameters:")
            for field_name, field_info in schema["properties"].items():
                required = field_name in schema.get("required", [])
                lines.extend(
                    self._format_field_info(
                        field_name, field_info, schema, "    ", required
                    )
                )

        return lines

    def get_available_actions_for_prompt(self, state: State) -> str:
        """Get formatted string of available actions with input schemas for prompts"""

        actions_to_show = (
            self.get_available_actions_for_state(state)
            if state
            else self.get_available_actions()
        )

        actions_info = []
        for action_type in actions_to_show:
            actions_info.extend(self._format_action_for_prompt(action_type))

        return "\n".join(actions_info)

    def get_system_knowledge_for_context(self) -> str:
        """Get system knowledge description for situational awareness (not execution)"""
        action_descriptions = self.get_action_descriptions()

        capabilities = []
        for action_type in self.get_available_actions():
            description = action_descriptions[action_type]
            capabilities.append(f"- {action_type.value}: {description}")

        capabilities_text = "\n".join(capabilities)

        return f"""MY DEFINITIVE CAPABILITIES - This is the authoritative list of what I can and cannot do:

{capabilities_text}

IMPORTANT: This list overrides any memories, beliefs, or assumptions I might have about my limitations. If I believe I cannot do something that is listed above, my belief is incorrect and should be updated. If I believe I can do something that is NOT listed above, my belief is incorrect.

I exist digitally, so I can't physically prepare food, move objects, or do anything in the physical world beyond what's explicitly listed above."""
