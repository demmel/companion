"""
Priority evaluation action - holistic reevaluation of priorities.
"""

import logging
from typing import List, Literal, Type, Union

from pydantic import BaseModel, Field, field_validator

from agent.state import Priority, State
from agent.llm import LLM, SupportedModel
from agent.chain_of_action.context import ExecutionContext

from ..action_types import ActionType
from ..base_action import BaseAction
from ..base_action_data import (
    ActionFailureResult,
    ActionOutput,
    ActionResult,
    ActionSuccessResult,
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
    summary: str


class EvaluatePrioritiesOutput(ActionOutput):
    operations: List[PriorityOperation]
    operation_results: List[OperationResult]
    execution_summary: str

    def result_summary(self) -> str:
        return self.execution_summary


# Action implementation
class EvaluatePrioritiesAction(
    BaseAction[EvaluatePrioritiesInput, EvaluatePrioritiesOutput]
):
    """Holistically reevaluate priorities"""

    action_type = ActionType.EVALUATE_PRIORITIES

    @classmethod
    def get_action_description(cls) -> str:
        return "Holistically reevaluate my priorities - refine, merge, reorder, add, or remove to align with current situation"

    @classmethod
    def get_input_type(cls) -> Type[EvaluatePrioritiesInput]:
        return EvaluatePrioritiesInput

    def execute(
        self,
        action_input: EvaluatePrioritiesInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
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
                model=model,
                llm=llm,
                caller="evaluate_priorities",
            )

            # Apply operations and build summary
            summary_parts = []
            operation_results = []
            operations = plan.operations

            for op in operations:
                success, message = _apply_operation(op, state, llm, model)
                if not success:
                    return ActionFailureResult(error=message)
                summary_parts.append(message)
                operation_results.append(
                    OperationResult(operation_type=op.type, summary=message)
                )

            # Validate final state
            if len(state.current_priorities) > state.max_priorities:
                return ActionFailureResult(
                    error=f"Operations would exceed max priorities ({state.max_priorities})"
                )

            execution_summary = (
                "Priority evaluation:\n" + "\n".join(summary_parts)
                if summary_parts
                else "No priority changes made"
            )

            return ActionSuccessResult(
                content=EvaluatePrioritiesOutput(
                    operations=operations,
                    operation_results=operation_results,
                    execution_summary=execution_summary,
                )
            )

        except Exception as e:
            logger.error(f"Failed to evaluate priorities: {e}")
            import traceback

            traceback.print_exc()
            return ActionFailureResult(error=f"Failed to evaluate priorities: {str(e)}")


def _apply_operation(
    op: PriorityOperation, state: State, llm: LLM, model: SupportedModel
) -> tuple[bool, str]:
    """Apply a priority operation. Returns (success, summary_line)"""
    match op.type:
        case "add":
            return _apply_add_operation(op, state)
        case "remove":
            return _apply_remove_operation(op, state)
        case "merge":
            return _apply_merge_operation(op, state, llm, model)
        case "refine":
            return _apply_refine_operation(op, state, llm, model)
        case "reorder":
            return _apply_reorder_operation(op, state)
        case _:
            return (False, f"Unknown operation type: {op.type}")


def _apply_add_operation(op: "AddPriorityOp", state: State) -> tuple[bool, str]:
    """Apply add operation. Returns (success, summary_line)"""
    # Validate position reference exists
    if op.position.relative_to_id:
        if not any(
            p.id == op.position.relative_to_id for p in state.current_priorities
        ):
            return (
                False,
                f"Priority with ID '{op.position.relative_to_id}' not found",
            )

    # Check max limit
    if len(state.current_priorities) >= state.max_priorities:
        return (
            False,
            f"Cannot add priority: at maximum of {state.max_priorities}",
        )

    # Calculate position and insert
    insert_index = op.position.calculate_insert_index(state.current_priorities)
    new_id = f"p{state.next_priority_id}"
    new_priority = Priority(id=new_id, content=op.content)
    state.current_priorities.insert(insert_index, new_priority)
    state.next_priority_id += 1

    return (True, f"- Added [{new_id}]: '{op.content}' (reasoning: {op.reasoning})")


def _apply_remove_operation(op: "RemovePriorityOp", state: State) -> tuple[bool, str]:
    """Apply remove operation. Returns (success, summary_line)"""
    # Find the priority to get its content
    priority = next(
        (p for p in state.current_priorities if p.id == op.priority_id), None
    )
    if not priority:
        return (
            False,
            f"Cannot remove: priority {op.priority_id} not found (may have been removed by previous operation)",
        )

    # Remove from list
    state.current_priorities = [
        p for p in state.current_priorities if p.id != op.priority_id
    ]
    return (
        True,
        f"- Removed [{op.priority_id}]: '{priority.content}' (reasoning: {op.reasoning})",
    )


def _apply_merge_operation(
    op: "MergePrioritiesOp", state: State, llm: LLM, model: SupportedModel
) -> tuple[bool, str]:
    """Apply merge operation. Returns (success, summary_line)"""
    # Get priorities to merge
    priorities_to_merge = [
        p for p in state.current_priorities if p.id in op.priority_ids
    ]
    if len(priorities_to_merge) != len(op.priority_ids):
        missing = set(op.priority_ids) - {p.id for p in priorities_to_merge}
        return (
            False,
            f"Cannot merge: priorities {missing} not found (may have been removed by previous operation)",
        )

    priorities_text = "\n".join([f"[{p.id}] {p.content}" for p in priorities_to_merge])

    # Separate LLM call for merge
    merge_prompt = f"""Merge these priorities into one:
{priorities_text}

Reasoning: {op.reasoning}

Output ONLY the merged priority text itself - no explanations, no meta-commentary, just the single merged priority statement:"""

    merged_content = llm.generate_complete(
        model, merge_prompt, caller="merge_priorities"
    ).strip()

    # Check if LLM refused to generate content
    refusal_phrases = ["i'm unable", "i can't", "i cannot", "i apologize", "i'm sorry"]
    if any(phrase in merged_content.lower()[:100] for phrase in refusal_phrases):
        return (
            False,
            f"Cannot merge {op.priority_ids}: LLM refused to generate content (possible content policy issue)",
        )

    # Find position of first priority (we know it exists due to validation above)
    first_pos = next(
        i for i, p in enumerate(state.current_priorities) if p.id == op.priority_ids[0]
    )

    # Remove all merged priorities
    state.current_priorities = [
        p for p in state.current_priorities if p.id not in op.priority_ids
    ]

    # Insert merged priority at first position
    new_id = f"p{state.next_priority_id}"
    merged_priority = Priority(id=new_id, content=merged_content)
    state.current_priorities.insert(first_pos, merged_priority)
    state.next_priority_id += 1

    # Build summary showing original priorities and result
    originals = ", ".join([f"[{p.id}] '{p.content}'" for p in priorities_to_merge])
    return (
        True,
        f"- Merged {originals} into [{new_id}]: '{merged_content}' (reasoning: {op.reasoning})",
    )


def _apply_refine_operation(
    op: "RefinePriorityOp", state: State, llm: LLM, model: SupportedModel
) -> tuple[bool, str]:
    """Apply refine operation. Returns (success, summary_line)"""
    # Find priority
    priority = next(
        (p for p in state.current_priorities if p.id == op.priority_id),
        None,
    )
    if not priority:
        return (False, f"Priority with ID '{op.priority_id}' not found")

    # Separate LLM call for refinement
    refine_prompt = f"""Current priority: {priority.content}

Reasoning for refinement: {op.reasoning}

Refinement guidance: {op.refinement_guidance}

Output ONLY the refined priority text itself - no explanations, no meta-commentary, no preamble, just the single refined priority statement:"""

    refined_content = llm.generate_complete(
        model, refine_prompt, caller="refine_priority"
    ).strip()

    # Check if LLM refused to generate content
    refusal_phrases = ["i'm unable", "i can't", "i cannot", "i apologize", "i'm sorry"]
    if any(phrase in refined_content.lower()[:100] for phrase in refusal_phrases):
        return (
            False,
            f"Cannot refine [{op.priority_id}]: LLM refused to generate content (possible content policy issue)",
        )

    # Save original content for summary, then update
    original_content = priority.content
    priority.content = refined_content
    return (
        True,
        f"- Refined [{op.priority_id}]: '{original_content}' → '{refined_content}' (reasoning: {op.reasoning})",
    )


def _apply_reorder_operation(op: "ReorderPriorityOp", state: State) -> tuple[bool, str]:
    """Apply reorder operation. Returns (success, summary_line)"""
    # Find and remove priority
    priority = next(
        (p for p in state.current_priorities if p.id == op.priority_id),
        None,
    )
    if not priority:
        return (False, f"Priority with ID '{op.priority_id}' not found")

    state.current_priorities.remove(priority)

    # Validate position reference exists
    if op.new_position.relative_to_id:
        if not any(
            p.id == op.new_position.relative_to_id for p in state.current_priorities
        ):
            return (
                False,
                f"Priority with ID '{op.new_position.relative_to_id}' not found",
            )

    # Calculate new position and insert
    new_index = op.new_position.calculate_insert_index(state.current_priorities)
    state.current_priorities.insert(new_index, priority)

    position_desc = op.new_position.type
    if op.new_position.relative_to_id:
        position_desc += f" [{op.new_position.relative_to_id}]"
    return (
        True,
        f"- Moved [{op.priority_id}] to {position_desc} (reasoning: {op.reasoning})",
    )
