"""EVALUATE_PRIORITIES action data types (operations/inputs/outputs/record)."""

from typing import List, Literal, Union

from pydantic import BaseModel, Field, model_validator

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData
from .priority_data import RelativePosition


class AddPriorityOp(BaseModel):
    type: Literal["add"] = "add"
    reasoning: str = Field(description="Why this priority needs to be added")
    content: str
    position: RelativePosition


class RemovePriorityOp(BaseModel):
    type: Literal["remove"] = "remove"
    reasoning: str = Field(description="Why this priority should be removed")
    priority_id: str


class MergePrioritiesOp(BaseModel):
    type: Literal["merge"] = "merge"
    reasoning: str = Field(
        description="Why these priorities should be merged and what the combined focus should be",
    )
    priority_ids: List[str]  # First one's position kept


class RefinePriorityOp(BaseModel):
    type: Literal["refine"] = "refine"
    reasoning: str = Field(
        description="Why this priority needs refinement and what improvement is needed",
    )
    priority_id: str
    refinement_guidance: str  # How to refine, not the refined content


class ReorderPriorityOp(BaseModel):
    type: Literal["reorder"] = "reorder"
    reasoning: str = Field(description="Why this priority's position should change")
    priority_id: str
    new_position: RelativePosition


PriorityOperation = Union[
    AddPriorityOp,
    RemovePriorityOp,
    MergePrioritiesOp,
    RefinePriorityOp,
    ReorderPriorityOp,
]


class EvaluatePrioritiesInput(BaseModel):
    focus: str = Field(
        description="Your reasoning for why you're evaluating priorities right now and what you want to achieve, given your current situation"
    )


class OperationResult(BaseModel):
    """Result of applying a single operation"""

    operation_type: Literal["add", "remove", "merge", "refine", "reorder"]
    success: bool = True  # Default for backwards compat with old data
    summary: str

    # Computed values for replay (not in plan)
    created_id: str | None = None  # For add, merge - the assigned priority ID
    new_index: int | None = None  # For add, merge, reorder - absolute position
    new_content: str | None = None  # For merge, refine - LLM-generated content

    @model_validator(mode="after")
    def parse_summary_for_backwards_compat(self) -> "OperationResult":
        """Parse summary string to extract computed values for old data."""
        import re

        # Skip if values already populated (new data format)
        if self.operation_type == "refine" and self.new_content is None:
            # Format: "- Refined [p1]: 'old' → 'new content' (reasoning: ...)"
            # Content may contain quotes, so match until ' (reasoning:
            match = re.search(r"→ '(.+)' \(reasoning:", self.summary)
            if match:
                self.new_content = match.group(1)

        elif self.operation_type == "add" and self.created_id is None:
            # Format: "- Added [p5]: 'content' (reasoning: ...)"
            match = re.search(r"Added \[(p\d+)\]:", self.summary)
            if match:
                self.created_id = match.group(1)

        elif self.operation_type == "merge" and self.created_id is None:
            # Format: "- Merged ... into [p5]: 'merged content' (reasoning: ...)"
            # Content may contain quotes, so match until ' (reasoning:
            match = re.search(r"into \[(p\d+)\]: '(.+)' \(reasoning:", self.summary)
            if match:
                self.created_id = match.group(1)
                self.new_content = match.group(2)

        return self


class EvaluatePrioritiesOutput(ActionOutput):
    operations: List[PriorityOperation]
    operation_results: List[OperationResult]
    execution_summary: str

    def result_summary(self) -> str:
        return self.execution_summary


class EvaluatePrioritiesActionData(
    BaseActionData[EvaluatePrioritiesInput, EvaluatePrioritiesOutput]
):
    type: Literal[ActionType.EVALUATE_PRIORITIES] = ActionType.EVALUATE_PRIORITIES
