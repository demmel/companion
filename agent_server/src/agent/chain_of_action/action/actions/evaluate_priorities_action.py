"""
Priority evaluation action - holistic reevaluation of priorities.
"""

import logging
from typing import List, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from agent.state import Priority, State
from agent.llm import LLM, SupportedModel
from agent.chain_of_action.context import ExecutionContext

from ..action_types import ActionType
from ..base_action import BaseAction, register_action
from ..base_action_data import (
    ActionFailureResult,
    ActionOutput,
    ActionResult,
    ActionSuccessResult,
    BaseActionData,
)
from .priority_actions import RelativePosition

logger = logging.getLogger(__name__)


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


# Input/Output models
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


# Action implementation
@register_action(ActionType.EVALUATE_PRIORITIES)
class EvaluatePrioritiesAction(
    BaseAction[EvaluatePrioritiesInput, EvaluatePrioritiesOutput]
):
    """Holistically reevaluate priorities"""

    @classmethod
    def get_action_description(cls) -> str:
        return "Holistically reevaluate my priorities - refine, merge, reorder, add, or remove to align with current situation"

    def execute(
        self,
        action_input: EvaluatePrioritiesInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        progress_callback,
    ) -> ActionResult[EvaluatePrioritiesOutput]:
        from agent.structured_llm import direct_structured_llm_call
        from agent.chain_of_action.prompts import format_section
        from agent.state import build_agent_state_description

        logger.debug("=== EVALUATE_PRIORITIES ACTION ===")
        logger.debug(f"FOCUS: {action_input.focus}")

        # Build prompt with current priorities and situational context
        state_desc = build_agent_state_description(state)

        prompt = f"""I am {state.name}, {state.role}. I need to holistically reevaluate my priorities.

{state_desc}

{format_section("MY SITUATIONAL CONTEXT", context.situation_analysis)}

{format_section("WHY I'M EVALUATING", action_input.focus)}

I should review my current priorities and decide what operations to perform:
- **Add**: Add a new priority if something important is missing (but only if under my limit of {state.max_priorities})
- **Remove**: Remove priorities that are no longer relevant or completed
- **Merge**: Combine similar or related priorities into one clearer priority
- **Refine**: Improve the wording or specificity of a priority
- **Reorder**: Change the precedence of priorities to better reflect what matters most right now

I will generate a list of operations that will improve my priority list to better align with my current situation and goals.

CRITICAL - VERIFY BEFORE EVERY OPERATION:
- The ONLY priorities that currently exist are those listed in "My Current Priorities" above with their [id: ...] tags.
- I MUST NOT attempt to remove, merge, refine, or reorder any priority ID that is not explicitly shown in that list.
- If I remember or believe a priority exists (like p309, p307, p331, etc.) but it's NOT in the current list above, it does NOT exist anymore.
- I should ONLY operate on priority IDs that I can see in the current state description above.
- Example: If "My Current Priorities" shows [id: p1], [id: p5], [id: p10], then ONLY p1, p5, and p10 exist. Any other ID (p2, p3, p4, p6, p7, p8, p9, p11, etc.) does NOT exist and cannot be operated on.

IMPORTANT GUIDELINES:
- I should be thoughtful and intentional - not every evaluation needs to result in changes. If my priorities are already well-aligned, I may return an empty list of operations.
- Operations are applied sequentially in the order I specify. Each operation sees the state created by previous operations.
- I must NOT reference a priority ID in a later operation if I removed or merged it in an earlier operation.
- For example: If I remove priority [p1], I cannot later refine, reorder, or merge [p1].
- When using reorder operations after remove/merge operations, ensure the relative_to_id still exists.
- Plan the sequence carefully: typically do removes first, then merges, then refines, then reorders, then adds.
- Keep refinement guidance clear and specific - the LLM will generate ONLY the refined text, not explanations."""

        try:
            # Get operations from LLM
            class EvaluationPlan(BaseModel):
                operations: List[PriorityOperation]

            plan = direct_structured_llm_call(
                prompt=prompt,
                response_model=EvaluationPlan,
                model=context.evaluate_priorities_action_model,
                llm=llm,
                caller="evaluate_priorities",
            )

            # Compute operations on working copies (don't modify real state yet)
            working_priorities = list(state.current_priorities)
            working_next_id = state.next_priority_id
            operation_results: List[OperationResult] = []
            operations = plan.operations

            for op in operations:
                result = _compute_operation(
                    op,
                    working_priorities,
                    working_next_id,
                    llm,
                    context.evaluate_priorities_action_model,
                    state.max_priorities,
                )
                if not result.success:
                    return ActionFailureResult(error=result.summary)
                operation_results.append(result)
                # Update working_next_id if operation created a new ID
                if result.created_id:
                    working_next_id = int(result.created_id[1:]) + 1

            # Validate final state
            if len(working_priorities) > state.max_priorities:
                return ActionFailureResult(
                    error=f"Operations would exceed max priorities ({state.max_priorities})"
                )

            # Build summary
            summary_parts = [r.summary for r in operation_results]
            execution_summary = (
                "Priority evaluation:\n" + "\n".join(summary_parts)
                if summary_parts
                else "No priority changes made"
            )

            output = EvaluatePrioritiesOutput(
                operations=operations,
                operation_results=operation_results,
                execution_summary=execution_summary,
            )

            # Apply state changes now that all operations succeeded
            self.apply_state_change(state, action_input, output)

            return ActionSuccessResult(content=output)

        except Exception as e:
            logger.error(f"Failed to evaluate priorities: {e}")
            import traceback

            traceback.print_exc()
            return ActionFailureResult(error=f"Failed to evaluate priorities: {str(e)}")

    def apply_state_change(
        self,
        state: State,
        action_input: EvaluatePrioritiesInput,
        output: EvaluatePrioritiesOutput,
    ) -> None:
        """Apply priority changes from the computed operation results."""
        for op, op_result in zip(output.operations, output.operation_results):
            if not op_result.success:
                continue

            match op.type:
                case "add":
                    assert isinstance(op, AddPriorityOp)
                    assert op_result.created_id is not None
                    # Compute index from position if not stored (backwards compat)
                    new_index = op_result.new_index
                    if new_index is None:
                        new_index = op.position.calculate_insert_index(
                            state.current_priorities
                        )
                    new_priority = Priority(id=op_result.created_id, content=op.content)
                    state.current_priorities.insert(new_index, new_priority)
                    state.next_priority_id = int(op_result.created_id[1:]) + 1

                case "remove":
                    assert isinstance(op, RemovePriorityOp)
                    state.current_priorities = [
                        p for p in state.current_priorities if p.id != op.priority_id
                    ]

                case "merge":
                    assert isinstance(op, MergePrioritiesOp)
                    assert op_result.created_id is not None
                    assert op_result.new_content is not None
                    # Find position of first priority before removing
                    first_pos = op_result.new_index
                    if first_pos is None:
                        first_pos = next(
                            (
                                i
                                for i, p in enumerate(state.current_priorities)
                                if p.id == op.priority_ids[0]
                            ),
                            0,
                        )
                    # Remove all merged priorities
                    state.current_priorities = [
                        p
                        for p in state.current_priorities
                        if p.id not in op.priority_ids
                    ]
                    # Insert merged priority at computed position
                    merged = Priority(
                        id=op_result.created_id, content=op_result.new_content
                    )
                    state.current_priorities.insert(first_pos, merged)
                    state.next_priority_id = int(op_result.created_id[1:]) + 1

                case "refine":
                    assert isinstance(op, RefinePriorityOp)
                    assert op_result.new_content is not None
                    # Find and update the priority content
                    for p in state.current_priorities:
                        if p.id == op.priority_id:
                            p.content = op_result.new_content
                            break

                case "reorder":
                    assert isinstance(op, ReorderPriorityOp)
                    # Find and remove priority
                    priority = next(
                        (p for p in state.current_priorities if p.id == op.priority_id),
                        None,
                    )
                    if priority:
                        state.current_priorities.remove(priority)
                        # Compute index from position if not stored (backwards compat)
                        new_index = op_result.new_index
                        if new_index is None:
                            new_index = op.new_position.calculate_insert_index(
                                state.current_priorities
                            )
                        state.current_priorities.insert(new_index, priority)


def _compute_operation(
    op: PriorityOperation,
    working_priorities: List[Priority],
    working_next_id: int,
    llm: LLM,
    model: SupportedModel,
    max_priorities: int,
) -> OperationResult:
    """
    Compute operation result, updating working state.

    Args:
        op: The operation to compute
        working_priorities: Working copy of priorities (will be modified)
        working_next_id: Current next priority ID
        llm: LLM instance for merge/refine
        model: Model to use
        max_priorities: Maximum allowed priorities

    Returns:
        OperationResult with success status and computed values
    """
    match op.type:
        case "add":
            return _compute_add_operation(
                op, working_priorities, working_next_id, max_priorities
            )
        case "remove":
            return _compute_remove_operation(op, working_priorities)
        case "merge":
            return _compute_merge_operation(
                op, working_priorities, working_next_id, llm, model
            )
        case "refine":
            return _compute_refine_operation(op, working_priorities, llm, model)
        case "reorder":
            return _compute_reorder_operation(op, working_priorities)
        case _:
            return OperationResult(
                operation_type=op.type,
                success=False,
                summary=f"Unknown operation type: {op.type}",
            )


def _compute_add_operation(
    op: "AddPriorityOp",
    working_priorities: List[Priority],
    working_next_id: int,
    max_priorities: int,
) -> OperationResult:
    """Compute add operation result and update working priorities."""
    # Validate position reference exists
    if op.position.relative_to_id:
        if not any(p.id == op.position.relative_to_id for p in working_priorities):
            return OperationResult(
                operation_type="add",
                success=False,
                summary=f"Priority with ID '{op.position.relative_to_id}' not found",
            )

    # Check max limit
    if len(working_priorities) >= max_priorities:
        return OperationResult(
            operation_type="add",
            success=False,
            summary=f"Cannot add priority: at maximum of {max_priorities}",
        )

    # Calculate position and insert into working copy
    insert_index = op.position.calculate_insert_index(working_priorities)
    new_id = f"p{working_next_id}"
    new_priority = Priority(id=new_id, content=op.content)
    working_priorities.insert(insert_index, new_priority)

    return OperationResult(
        operation_type="add",
        success=True,
        summary=f"- Added [{new_id}]: '{op.content}' (reasoning: {op.reasoning})",
        created_id=new_id,
        new_index=insert_index,
    )


def _compute_remove_operation(
    op: "RemovePriorityOp",
    working_priorities: List[Priority],
) -> OperationResult:
    """Compute remove operation result and update working priorities."""
    # Find the priority to get its content
    priority = next((p for p in working_priorities if p.id == op.priority_id), None)
    if not priority:
        return OperationResult(
            operation_type="remove",
            success=False,
            summary=f"Cannot remove: priority {op.priority_id} not found (may have been removed by previous operation)",
        )

    # Remove from working copy
    working_priorities[:] = [p for p in working_priorities if p.id != op.priority_id]

    return OperationResult(
        operation_type="remove",
        success=True,
        summary=f"- Removed [{op.priority_id}]: '{priority.content}' (reasoning: {op.reasoning})",
    )


def _compute_merge_operation(
    op: "MergePrioritiesOp",
    working_priorities: List[Priority],
    working_next_id: int,
    llm: LLM,
    model: SupportedModel,
) -> OperationResult:
    """Compute merge operation result and update working priorities."""
    # Get priorities to merge
    priorities_to_merge = [p for p in working_priorities if p.id in op.priority_ids]
    if len(priorities_to_merge) != len(op.priority_ids):
        missing = set(op.priority_ids) - {p.id for p in priorities_to_merge}
        return OperationResult(
            operation_type="merge",
            success=False,
            summary=f"Cannot merge: priorities {missing} not found (may have been removed by previous operation)",
        )

    priorities_text = "\n".join([f"[{p.id}] {p.content}" for p in priorities_to_merge])

    # Separate LLM call for merge
    merge_prompt = f"""Merge these priorities into one:
{priorities_text}

Reasoning: {op.reasoning}

Output ONLY the merged priority text itself - no explanations, no meta-commentary, just the single merged priority statement:"""

    merged_content = llm.generate(
        model, merge_prompt, caller="merge_priorities"
    ).strip()

    # Check if LLM refused to generate content
    refusal_phrases = ["i'm unable", "i can't", "i cannot", "i apologize", "i'm sorry"]
    if any(phrase in merged_content.lower()[:100] for phrase in refusal_phrases):
        return OperationResult(
            operation_type="merge",
            success=False,
            summary=f"Cannot merge {op.priority_ids}: LLM refused to generate content (possible content policy issue)",
        )

    # Find position of first priority (we know it exists due to validation above)
    first_pos = next(
        i for i, p in enumerate(working_priorities) if p.id == op.priority_ids[0]
    )

    # Remove all merged priorities from working copy
    working_priorities[:] = [
        p for p in working_priorities if p.id not in op.priority_ids
    ]

    # Insert merged priority at first position
    new_id = f"p{working_next_id}"
    merged_priority = Priority(id=new_id, content=merged_content)
    working_priorities.insert(first_pos, merged_priority)

    # Build summary showing original priorities and result
    originals = ", ".join([f"[{p.id}] '{p.content}'" for p in priorities_to_merge])
    return OperationResult(
        operation_type="merge",
        success=True,
        summary=f"- Merged {originals} into [{new_id}]: '{merged_content}' (reasoning: {op.reasoning})",
        created_id=new_id,
        new_index=first_pos,
        new_content=merged_content,
    )


def _compute_refine_operation(
    op: "RefinePriorityOp",
    working_priorities: List[Priority],
    llm: LLM,
    model: SupportedModel,
) -> OperationResult:
    """Compute refine operation result and update working priorities."""
    # Find priority
    priority = next(
        (p for p in working_priorities if p.id == op.priority_id),
        None,
    )
    if not priority:
        return OperationResult(
            operation_type="refine",
            success=False,
            summary=f"Priority with ID '{op.priority_id}' not found",
        )

    # Separate LLM call for refinement
    refine_prompt = f"""Current priority: {priority.content}

Reasoning for refinement: {op.reasoning}

Refinement guidance: {op.refinement_guidance}

Output ONLY the refined priority text itself - no explanations, no meta-commentary, no preamble, just the single refined priority statement:"""

    refined_content = llm.generate(
        model, refine_prompt, caller="refine_priority"
    ).strip()

    # Check if LLM refused to generate content
    refusal_phrases = ["i'm unable", "i can't", "i cannot", "i apologize", "i'm sorry"]
    if any(phrase in refined_content.lower()[:100] for phrase in refusal_phrases):
        return OperationResult(
            operation_type="refine",
            success=False,
            summary=f"Cannot refine [{op.priority_id}]: LLM refused to generate content (possible content policy issue)",
        )

    # Save original content for summary, then update working copy
    original_content = priority.content
    priority.content = refined_content

    return OperationResult(
        operation_type="refine",
        success=True,
        summary=f"- Refined [{op.priority_id}]: '{original_content}' → '{refined_content}' (reasoning: {op.reasoning})",
        new_content=refined_content,
    )


def _compute_reorder_operation(
    op: "ReorderPriorityOp",
    working_priorities: List[Priority],
) -> OperationResult:
    """Compute reorder operation result and update working priorities."""
    # Find and remove priority
    priority = next(
        (p for p in working_priorities if p.id == op.priority_id),
        None,
    )
    if not priority:
        return OperationResult(
            operation_type="reorder",
            success=False,
            summary=f"Priority with ID '{op.priority_id}' not found",
        )

    working_priorities.remove(priority)

    # Validate position reference exists
    if op.new_position.relative_to_id:
        if not any(p.id == op.new_position.relative_to_id for p in working_priorities):
            # Restore priority to avoid corrupted state
            working_priorities.append(priority)
            return OperationResult(
                operation_type="reorder",
                success=False,
                summary=f"Priority with ID '{op.new_position.relative_to_id}' not found",
            )

    # Calculate new position and insert
    new_index = op.new_position.calculate_insert_index(working_priorities)
    working_priorities.insert(new_index, priority)

    position_desc = op.new_position.type
    if op.new_position.relative_to_id:
        position_desc += f" [{op.new_position.relative_to_id}]"

    return OperationResult(
        operation_type="reorder",
        success=True,
        summary=f"- Moved [{op.priority_id}] to {position_desc} (reasoning: {op.reasoning})",
        new_index=new_index,
    )


class EvaluatePrioritiesActionData(
    BaseActionData[EvaluatePrioritiesInput, EvaluatePrioritiesOutput]
):
    type: Literal[ActionType.EVALUATE_PRIORITIES] = ActionType.EVALUATE_PRIORITIES
