"""
Roleplay tools as proper tool classes
"""

from typing import Type, Dict, Any, Callable
from pydantic import Field

from agent.tools import (
    BaseTool,
    ToolInput,
)
from agent.types import (
    TextToolContent,
    ToolCallError,
    ToolCallSuccess,
    ToolResult,
)


# ============================================================================
# INPUT SCHEMAS
# ============================================================================


class AssumeCharacterInput(ToolInput):
    character_name: str = Field(description="Name of the character to roleplay")
    personality: str = Field(description="Key personality traits and characteristics")
    background: str = Field(default="", description="Character's background story")
    quirks: str = Field(default="", description="Unique quirks or mannerisms")


class SetMoodInput(ToolInput):
    mood: str = Field(description="Current emotional state or mood")
    intensity: str = Field(
        default="moderate", description="Intensity level (low, moderate, high)"
    )
    flavor_text: str = Field(
        default="", description="Optional flavor text describing the mood change"
    )


class RememberDetailInput(ToolInput):
    detail: str = Field(description="Important detail to remember")
    category: str = Field(
        default="general", description="Category for organizing the memory"
    )


class InternalThoughtInput(ToolInput):
    thought: str = Field(description="Character's internal thought or motivation")


class RelationshipStatusInput(ToolInput):
    relationship: str = Field(description="Type of relationship with the user")
    feelings: str = Field(default="", description="Current feelings or dynamic")


class SceneSettingInput(ToolInput):
    location: str = Field(description="Where the scene takes place")
    atmosphere: str = Field(default="", description="Mood or atmosphere of the scene")
    time: str = Field(default="", description="Time of day or period")


class CharacterActionInput(ToolInput):
    action: str = Field(description="Physical action or behavior to perform")
    reason: str = Field(default="", description="Motivation behind the action")


class EmotionalReactionInput(ToolInput):
    emotion: str = Field(description="Type of emotional reaction")
    trigger: str = Field(default="", description="What triggered this emotion")
    intensity: str = Field(
        default="moderate", description="Intensity level (low, moderate, high)"
    )


class SwitchCharacterInput(ToolInput):
    character_name: str = Field(description="Name of existing character to switch to")


class CorrectDetailInput(ToolInput):
    old_detail: str = Field(description="The incorrect detail that needs to be changed")
    new_detail: str = Field(description="The correct detail to replace it with")
    category: str = Field(
        default="general", description="Category of the detail being corrected"
    )


# ============================================================================
# TOOL CLASSES
# ============================================================================


class AssumeCharacterTool(BaseTool):
    """Tool for creating or switching to a character"""

    @property
    def name(self) -> str:
        return "assume_character"

    @property
    def description(self) -> str:
        return "Assume a character persona for roleplay"

    @property
    def input_schema(self) -> Type[ToolInput]:
        return AssumeCharacterInput

    def run(
        self,
        agent,
        input_data: AssumeCharacterInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        from ..character_state import create_character_state

        # Check if character already exists
        characters = agent.get_state("characters") or {}
        char_id = None

        for cid, char in characters.items():
            if char["name"].lower() == input_data.character_name.lower():
                char_id = cid
                break

        if char_id:
            # Switch to existing character
            agent.set_state("current_character_id", char_id)
            char = characters[char_id]
            result = f"SWITCHED TO EXISTING CHARACTER: {char['name']}"
        else:
            # Create new character
            char = create_character_state(
                input_data.character_name,
                input_data.personality,
                input_data.background,
                input_data.quirks,
            )
            characters[char["id"]] = char
            agent.set_state("characters", characters)
            agent.set_state("current_character_id", char["id"])
            result = f"NEW CHARACTER CREATED: {input_data.character_name}"

        content = f"""{result}
PERSONALITY: {input_data.personality}
BACKGROUND: {input_data.background}
QUIRKS: {input_data.quirks}

You are now roleplaying as {input_data.character_name}. Respond in character from now on."""

        content = TextToolContent(
            text=content,
        )

        return ToolCallSuccess(content=content)


class SetMoodTool(BaseTool):
    """Tool for setting character mood"""

    @property
    def name(self) -> str:
        return "set_mood"

    @property
    def description(self) -> str:
        return "Set the character's current mood/emotional state"

    @property
    def input_schema(self) -> Type[ToolInput]:
        return SetMoodInput

    def run(
        self,
        agent,
        input_data: SetMoodInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        from ..character_state import update_character_mood

        current_char_id = agent.get_state("current_character_id")
        characters = agent.get_state("characters") or {}
        current_char = characters.get(current_char_id) if current_char_id else None

        if not current_char:
            return ToolCallError(error="No character is currently active")

        update_character_mood(current_char, input_data.mood, input_data.intensity)

        content = f"CHARACTER MOOD SET: {input_data.mood} (intensity: {input_data.intensity}). Adjust your responses to reflect this emotional state."

        content = TextToolContent(
            text=content,
        )

        return ToolCallSuccess(content=content)


class RememberDetailTool(BaseTool):
    """Tool for remembering conversation details"""

    @property
    def name(self) -> str:
        return "remember_detail"

    @property
    def description(self) -> str:
        return "Remember an important detail about the conversation"

    @property
    def input_schema(self) -> Type[ToolInput]:
        return RememberDetailInput

    def run(
        self,
        agent,
        input_data: RememberDetailInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        from ..character_state import add_character_memory

        current_char_id = agent.get_state("current_character_id")
        characters = agent.get_state("characters") or {}
        current_char = characters.get(current_char_id) if current_char_id else None

        if not current_char:
            return ToolCallError(error="No character is currently active")

        add_character_memory(current_char, input_data.detail, input_data.category)

        content = f"MEMORY STORED ({input_data.category}): {input_data.detail}. This detail will influence future interactions."
        content = TextToolContent(
            text=content,
        )
        return ToolCallSuccess(content=content)


class InternalThoughtTool(BaseTool):
    """Tool for character internal thoughts"""

    @property
    def name(self) -> str:
        return "internal_thought"

    @property
    def description(self) -> str:
        return "Share character's internal thoughts"

    @property
    def input_schema(self) -> Type[ToolInput]:
        return InternalThoughtInput

    def run(
        self,
        agent,
        input_data: InternalThoughtInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        from ..character_state import add_internal_thought

        current_char_id = agent.get_state("current_character_id")
        characters = agent.get_state("characters") or {}
        current_char = characters.get(current_char_id) if current_char_id else None

        if not current_char:
            return ToolCallError(error="No character is currently active")

        add_internal_thought(current_char, input_data.thought)

        content = f"INTERNAL THOUGHT: {input_data.thought}. This provides context for the character's behavior but isn't spoken aloud."
        content = TextToolContent(
            text=content,
        )
        return ToolCallSuccess(content=content)


class SceneSettingTool(BaseTool):
    """Tool for setting the scene"""

    @property
    def name(self) -> str:
        return "scene_setting"

    @property
    def description(self) -> str:
        return "Set or change the scene/environment"

    @property
    def input_schema(self) -> Type[ToolInput]:
        return SceneSettingInput

    def run(
        self,
        agent,
        input_data: SceneSettingInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        # Scene is global, not per-character
        agent.set_state(
            "global_scene",
            {
                "location": input_data.location,
                "atmosphere": input_data.atmosphere,
                "time": input_data.time,
            },
        )

        content = f"SCENE SET: {input_data.location}"
        if input_data.atmosphere:
            content += f"\nATMOSPHERE: {input_data.atmosphere}"
        if input_data.time:
            content += f"\nTIME: {input_data.time}"
        content = TextToolContent(
            text=content,
        )
        return ToolCallSuccess(content=content)


class CharacterActionTool(BaseTool):
    """Tool for character actions"""

    @property
    def name(self) -> str:
        return "character_action"

    @property
    def description(self) -> str:
        return "Describe a physical action the character performs"

    @property
    def input_schema(self) -> Type[ToolInput]:
        return CharacterActionInput

    def run(
        self,
        agent,
        input_data: CharacterActionInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        from ..character_state import add_character_action

        current_char_id = agent.get_state("current_character_id")
        characters = agent.get_state("characters") or {}
        current_char = characters.get(current_char_id) if current_char_id else None

        if not current_char:
            return ToolCallError(error="No character is currently active")

        add_character_action(current_char, input_data.action, input_data.reason)

        content = f"CHARACTER ACTION: {input_data.action}"
        if input_data.reason:
            content += f"\nREASON: {input_data.reason}"

        content = TextToolContent(
            text=content,
        )

        return ToolCallSuccess(content=content)


class SwitchCharacterTool(BaseTool):
    """Tool for switching between characters"""

    @property
    def name(self) -> str:
        return "switch_character"

    @property
    def description(self) -> str:
        return "Switch to a different existing character"

    @property
    def input_schema(self) -> Type[ToolInput]:
        return SwitchCharacterInput

    def run(
        self,
        agent,
        input_data: SwitchCharacterInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        characters = agent.get_state("characters") or {}
        current_char_id = agent.get_state("current_character_id")

        # Find character by name
        target_char_id = None
        for char_id, char in characters.items():
            if char["name"].lower() == input_data.character_name.lower():
                target_char_id = char_id
                break

        if not target_char_id:
            available_chars = [char["name"] for char in characters.values()]
            error_msg = f"Character '{input_data.character_name}' not found. Available characters: {', '.join(available_chars)}"
            return ToolCallError(error=error_msg)

        # Switch to the character
        old_char = characters.get(current_char_id)
        old_char_name = old_char["name"] if old_char else "None"

        agent.set_state("current_character_id", target_char_id)
        new_char = characters[target_char_id]

        content = f"SWITCHED from {old_char_name} to {new_char['name']}. You are now roleplaying as {new_char['name']} ({new_char['personality']})."
        content = TextToolContent(
            text=content,
        )
        return ToolCallSuccess(content=content)


class CorrectDetailTool(BaseTool):
    """Tool for correcting/changing established details"""

    @property
    def name(self) -> str:
        return "correct_detail"

    @property
    def description(self) -> str:
        return "Correct or change an established detail in the roleplay"

    @property
    def input_schema(self) -> Type[ToolInput]:
        return CorrectDetailInput

    def run(
        self,
        agent,
        input_data: CorrectDetailInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        current_char_id = agent.get_state("current_character_id")
        characters = agent.get_state("characters") or {}
        current_char = characters.get(current_char_id) if current_char_id else None

        if not current_char:
            return ToolCallError(error="No active character to correct details for")

        # Check if this is a character name change
        old_name = current_char.get("name", "")
        if (
            old_name.lower() in input_data.old_detail.lower()
            and "name" in input_data.old_detail.lower()
        ):
            # Extract new name from new_detail
            if "name" in input_data.new_detail.lower():
                # Try to extract the actual name
                import re

                name_match = re.search(
                    r"name is (\w+)|called (\w+)|(\w+) is|^(\w+)$",
                    input_data.new_detail,
                    re.IGNORECASE,
                )
                if name_match:
                    new_name = next(group for group in name_match.groups() if group)
                    current_char["name"] = new_name
                    result_msg = f"CHARACTER NAME CHANGED: '{old_name}' → '{new_name}'"
                else:
                    # Fallback: use the new_detail as the name if it's simple
                    words = input_data.new_detail.strip().split()
                    if len(words) == 1 and words[0].isalpha():
                        current_char["name"] = words[0]
                        result_msg = (
                            f"CHARACTER NAME CHANGED: '{old_name}' → '{words[0]}'"
                        )
                    else:
                        result_msg = f"CHARACTER NAME UPDATE: {input_data.new_detail}"
            else:
                result_msg = f"CHARACTER NAME UPDATE: {input_data.new_detail}"
        else:
            result_msg = f"DETAIL CORRECTED: '{input_data.old_detail}' → '{input_data.new_detail}'"

        # Remove old memories containing the incorrect detail
        if "memories" in current_char:
            current_char["memories"] = [
                memory
                for memory in current_char["memories"]
                if input_data.old_detail.lower() not in memory["detail"].lower()
            ]

        # Add the corrected detail as a new memory
        from ..character_state import add_character_memory

        add_character_memory(current_char, input_data.new_detail, input_data.category)

        # Update character state
        characters[current_char_id] = current_char
        agent.set_state("characters", characters)

        # Also update global memories if needed
        global_memories = agent.get_state("global_memories") or []
        global_memories = [
            memory
            for memory in global_memories
            if input_data.old_detail.lower() not in memory.lower()
        ]
        global_memories.append(input_data.new_detail)
        agent.set_state("global_memories", global_memories)

        content = f"{result_msg}. The character now knows and remembers the correct information."
        content = TextToolContent(
            text=content,
        )
        return ToolCallSuccess(content=content)


# ============================================================================
# TOOL REGISTRY
# ============================================================================

ROLEPLAY_TOOLS = [
    AssumeCharacterTool(),
    SetMoodTool(),
    RememberDetailTool(),
    InternalThoughtTool(),
    SceneSettingTool(),
    CharacterActionTool(),
    SwitchCharacterTool(),
    CorrectDetailTool(),
]
