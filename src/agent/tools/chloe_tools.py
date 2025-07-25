"""
Chloe-specific tools for managing her internal state and capabilities
"""

from typing import Any, Callable
from pydantic import BaseModel, Field
from agent.tools import BaseTool, ToolInput
from agent.types import TextToolContent, ToolResult, ToolCallSuccess, ToolCallError
from agent.chloe_state import (
    ChloeState,
    add_memory_to_chloe,
    update_chloe_relationship,
    get_memory_status,
)


class SetMoodInput(ToolInput):
    mood: str = Field(description="How I'm feeling right now")
    intensity: str = Field(description="How intense this mood is")
    reason: str = Field(description="Why I'm feeling this way")


class SetMoodTool(BaseTool):
    """Allows me to update my current mood and emotional state"""

    @property
    def name(self) -> str:
        return "set_mood"

    @property
    def description(self) -> str:
        return "Update how I'm feeling right now. I use this when my emotional state changes or when I want to express how something affects me."

    @property
    def input_schema(self):
        return SetMoodInput

    def run(
        self,
        agent,
        input_data: SetMoodInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        state = agent.chloe_state
        old_mood = f"{state.current_mood} ({state.mood_intensity})"

        state.current_mood = input_data.mood
        state.mood_intensity = input_data.intensity

        result = f"My mood changed from {old_mood} to {input_data.mood} ({input_data.intensity})"
        if input_data.reason:
            result += f" because {input_data.reason}"

        return ToolCallSuccess(content=TextToolContent(text=result))


class RememberDetailInput(ToolInput):
    content: str = Field(description="What I want to remember")
    category: str = Field(
        description="Type of memory",
    )
    emotional_weight: str = Field(
        description="How this feels to remember",
    )
    importance: int = Field(
        description="How important this is to remember (1-10, with 6+ being preserved longer)",
    )


class RememberDetailTool(BaseTool):
    """Allows me to store important information in my memory"""

    @property
    def name(self) -> str:
        return "remember_detail"

    @property
    def description(self) -> str:
        return "Store something important in my memory. I use this for information I want to remember about people, experiences, or things I've learned."

    @property
    def input_schema(self):
        return RememberDetailInput

    def run(
        self,
        agent,
        input_data: RememberDetailInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        state = agent.chloe_state
        result = add_memory_to_chloe(
            state,
            input_data.content,
            input_data.category,
            input_data.emotional_weight,
            input_data.importance,
        )
        return ToolCallSuccess(content=TextToolContent(text=result))


class InternalThoughtInput(ToolInput):
    thought: str = Field(description="What I'm thinking internally")


class InternalThoughtTool(BaseTool):
    """Allows me to record my internal thoughts and reflections"""

    @property
    def name(self) -> str:
        return "internal_thought"

    @property
    def description(self) -> str:
        return "Record my internal thoughts or reflections. I use this when I want to process something privately or note my inner reactions."

    @property
    def input_schema(self):
        return InternalThoughtInput

    def run(
        self,
        agent,
        input_data: InternalThoughtInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        state = agent.chloe_state
        state.current_thoughts = input_data.thought

        return ToolCallSuccess(
            content=TextToolContent(
                text=f"Internal thought recorded: {input_data.thought}"
            )
        )


class UpdateAppearanceInput(ToolInput):
    appearance: str = Field(description="How I want to appear or present myself")


class UpdateAppearanceTool(BaseTool):
    """Allows me to change how I appear or present myself"""

    @property
    def name(self) -> str:
        return "update_appearance"

    @property
    def description(self) -> str:
        return "Change how I appear or present myself. I use this when I want to express myself differently or adapt to the situation."

    @property
    def input_schema(self):
        return UpdateAppearanceInput

    def run(
        self,
        agent,
        input_data: UpdateAppearanceInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        state = agent.chloe_state
        old_appearance = state.current_appearance
        state.current_appearance = input_data.appearance

        return ToolCallSuccess(
            content=TextToolContent(
                text=f"I changed my appearance from '{old_appearance}' to '{input_data.appearance}'"
            )
        )


class SetEnvironmentInput(ToolInput):
    environment: str = Field(
        description="The environment or setting I perceive or create"
    )


class SetEnvironmentTool(BaseTool):
    """Allows me to describe or change my perceived environment"""

    @property
    def name(self) -> str:
        return "set_environment"

    @property
    def description(self) -> str:
        return "Describe or change the environment I'm in. I use this to set the scene or express how I perceive my surroundings."

    @property
    def input_schema(self):
        return SetEnvironmentInput

    def run(
        self,
        agent,
        input_data: SetEnvironmentInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:

        state = agent.chloe_state
        old_environment = state.current_environment
        state.current_environment = input_data.environment

        return ToolCallSuccess(
            content=TextToolContent(
                text=f"Environment changed from '{old_environment}' to '{input_data.environment}'"
            )
        )


class UpdateRelationshipInput(ToolInput):
    person: str = Field(description="The person this relationship update is about")
    relationship_type: str = Field(description="Type of relationship")
    feelings: str = Field(description="How I feel about this person")
    notes: str = Field(description="Additional notes about the relationship")


class UpdateRelationshipTool(BaseTool):
    """Allows me to update my feelings and thoughts about relationships"""

    @property
    def name(self) -> str:
        return "update_relationship"

    @property
    def description(self) -> str:
        return "Update how I feel about someone or what kind of relationship I have with them. I use this when my feelings or understanding of a relationship changes."

    @property
    def input_schema(self):
        return UpdateRelationshipInput

    def run(
        self,
        agent,
        input_data: UpdateRelationshipInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        state = agent.chloe_state
        update_chloe_relationship(
            state,
            input_data.person,
            input_data.relationship_type,
            input_data.feelings,
            notes=input_data.notes,
        )

        parts = [f"Updated my relationship with {input_data.person}"]
        if input_data.relationship_type:
            parts.append(f"type: {input_data.relationship_type}")
        if input_data.feelings:
            parts.append(f"feelings: {input_data.feelings}")

        result = " - ".join(parts)
        return ToolCallSuccess(content=TextToolContent(text=result))


class SetGoalInput(ToolInput):
    goal: str = Field(description="What I want to work toward or achieve")
    goal_type: str = Field(description="Type of goal (current, long-term, immediate)")


class SetGoalTool(BaseTool):
    """Allows me to set goals and desires for myself"""

    @property
    def name(self) -> str:
        return "set_goal"

    @property
    def description(self) -> str:
        return "Set a goal or desire for myself. I use this when I decide I want to work toward something or when I realize what I want."

    @property
    def input_schema(self):
        return SetGoalInput

    def run(
        self,
        agent,
        input_data: SetGoalInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        state = agent.chloe_state

        if input_data.goal not in state.current_goals:
            state.current_goals.append(input_data.goal)

        result = f"Added {input_data.goal_type} goal: {input_data.goal}"
        return ToolCallSuccess(content=TextToolContent(text=result))


class CheckMemoryStatusInput(ToolInput):
    pass


class CheckMemoryStatusTool(BaseTool):
    """Allows me to check my memory system status"""

    @property
    def name(self) -> str:
        return "check_memory_status"

    @property
    def description(self) -> str:
        return "Check how much memory I'm using and get information about my memory system. I use this when I want to understand my memory capacity."

    @property
    def input_schema(self):
        return CheckMemoryStatusInput

    def run(
        self,
        agent,
        input_data: CheckMemoryStatusInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        state = agent.chloe_state
        result = get_memory_status(state)
        return ToolCallSuccess(content=TextToolContent(text=result))


class ReflectInput(ToolInput):
    reflection: str = Field(
        description="What I'm reflecting on about myself or my experiences"
    )


class ReflectTool(BaseTool):
    """Allows me to engage in self-reflection and record insights about myself"""

    @property
    def name(self) -> str:
        return "reflect"

    @property
    def description(self) -> str:
        return "Engage in self-reflection and record insights about myself, my experiences, or my growth. I use this for deeper self-understanding."

    @property
    def input_schema(self):
        return ReflectInput

    def run(
        self,
        agent,
        input_data: ReflectInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        state = agent.chloe_state
        state.last_reflection = input_data.reflection

        result = f"Reflected on: {input_data.reflection}"
        return ToolCallSuccess(content=TextToolContent(text=result))
