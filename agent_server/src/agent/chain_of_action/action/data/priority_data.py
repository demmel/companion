"""Priority management action data types (inputs/outputs/records)."""

from typing import List, Literal, Optional, Union, assert_never

from pydantic import BaseModel, Field, model_validator

from agent.state import Priority

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData


class RelativePosition(BaseModel):
    type: Literal["before", "after", "highest", "lowest"]
    relative_to_id: Optional[str] = Field(
        default=None,
        description="ID of priority to position relative to (required for 'before' or 'after')",
    )

    @model_validator(mode="after")
    def validate_relative_to_id(self) -> "RelativePosition":
        if self.type in ["before", "after"] and not self.relative_to_id:
            raise ValueError(f"relative_to_id is required when type is '{self.type}'")
        return self

    def calculate_insert_index(self, priorities: List[Priority]) -> int:
        """Calculate insert index from this position specification"""
        match self.type:
            case "highest":
                return 0
            case "lowest":
                return len(priorities)
            case "before":
                return next(
                    i for i, p in enumerate(priorities) if p.id == self.relative_to_id
                )
            case "after":
                return (
                    next(
                        i
                        for i, p in enumerate(priorities)
                        if p.id == self.relative_to_id
                    )
                    + 1
                )
            case _:
                assert_never(self.type)


class AddPriorityInput(BaseModel):
    """Input for ADD_PRIORITY action"""

    reason: str = Field(
        description="Why this is important to me and worth prioritizing"
    )
    priority_content: str = Field(
        description="What I want to prioritize - a clear description of something I choose to focus on"
    )
    position: RelativePosition = Field(
        default_factory=lambda: RelativePosition(type="lowest"),
        description="Where to place this priority in my ordered list",
    )


class AddPrioritySuccessOutput(BaseModel):
    """Output for successful ADD_PRIORITY action"""

    reason: str
    type: Literal["success"] = "success"
    priority_id: str


class AddPriorityDuplicateOutput(BaseModel):
    """Output for duplicate ADD_PRIORITY action"""

    reason: str
    type: Literal["duplicate"] = "duplicate"
    existing_priority_id: str | None = None
    existing_priority_content: str | None = None


class AddPriorityOutput(ActionOutput):
    """Output for ADD_PRIORITY action"""

    content: str
    result: Union[AddPrioritySuccessOutput, AddPriorityDuplicateOutput]

    def result_summary(self) -> str:
        result = self.result
        match result.type:
            case "success":
                return f"Added new priority: '{self.content}' (id: {result.priority_id}) because {result.reason}"
            case "duplicate":
                if result.existing_priority_content:
                    return f"Priority '{self.content}' is similar to existing priority '{result.existing_priority_content}' (id: {result.existing_priority_id}). {result.reason}"
                else:
                    return f"Priority '{self.content}' appears to be a duplicate. {result.reason}"
            case _:
                return "Unknown result type"


class RemovePriorityInput(BaseModel):
    """Input for REMOVE_PRIORITY action"""

    reason: str = Field(
        description="Why I'm removing this priority (completed, no longer relevant, etc.)"
    )
    priority_id: str = Field(
        description="The ID of the priority I want to remove (e.g., 'p1', 'p2')"
    )


class RemovePriorityOutput(ActionOutput):
    """Output for REMOVE_PRIORITY action"""

    priority: Priority
    reason: str

    def result_summary(self) -> str:
        return f"Removed priority '{self.priority.content}' (id: {self.priority.id}) because {self.reason}"


class AddPriorityActionData(BaseActionData[AddPriorityInput, AddPriorityOutput]):
    type: Literal[ActionType.ADD_PRIORITY] = ActionType.ADD_PRIORITY


class RemovePriorityActionData(
    BaseActionData[RemovePriorityInput, RemovePriorityOutput]
):
    type: Literal[ActionType.REMOVE_PRIORITY] = ActionType.REMOVE_PRIORITY
