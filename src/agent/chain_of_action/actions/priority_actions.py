"""
Priority management actions implementation.
"""

import time
import logging
from typing import Type

from pydantic import BaseModel, Field
from typing import Optional

from agent.chain_of_action.trigger_history import TriggerHistory
from agent.state import Priority

from ..action_types import ActionType
from ..base_action import BaseAction
from ..action_result import ActionResult
from ..context import ExecutionContext

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class DuplicatePriorityCheck(BaseModel):
    """Result of checking if a priority is a duplicate"""

    reasoning: str = Field(
        description="Explanation of why this is or isn't a duplicate"
    )
    is_duplicate: bool = Field(
        description="True if the new priority is a duplicate or very similar to an existing one"
    )
    existing_priority_id: Optional[str] = Field(
        default=None,
        description="The ID of the existing priority that this duplicates (if is_duplicate is True)",
    )


class AddPriorityInput(BaseModel):
    """Input for ADD_PRIORITY action"""

    reason: str = Field(
        description="Why this is important to me and worth prioritizing"
    )
    priority_content: str = Field(
        description="What I want to prioritize - a clear description of something I choose to focus on"
    )


class RemovePriorityInput(BaseModel):
    """Input for REMOVE_PRIORITY action"""

    reason: str = Field(
        description="Why I'm removing this priority (completed, no longer relevant, etc.)"
    )
    priority_id: str = Field(
        description="The ID of the priority I want to remove (e.g., 'p1', 'p2')"
    )


class AddPriorityAction(BaseAction[AddPriorityInput, None]):
    """Add a new priority that the agent wants to focus on"""

    action_type = ActionType.ADD_PRIORITY

    def _check_for_duplicate_priority(
        self,
        new_priority: str,
        existing_priorities: list,
        llm: LLM,
        model: SupportedModel,
    ) -> DuplicatePriorityCheck:
        """Check if the new priority is a duplicate or very similar to existing ones"""
        from agent.structured_llm import direct_structured_llm_call

        existing_exact_match = next(
            (p for p in existing_priorities if p.content == new_priority), None
        )
        if existing_exact_match:
            return DuplicatePriorityCheck(
                is_duplicate=True,
                existing_priority_id=existing_exact_match.id,
                reasoning="Exact match found in existing priorities",
            )

        existing_list = "\n".join(
            [f"- {p.content} (id: {p.id})" for p in existing_priorities]
        )

        prompt = f"""I'm considering adding a new priority: "{new_priority}"

My current priorities are:
{existing_list}

I need to determine if this new priority is truly redundant with any existing priority. Priorities are only duplicates if they are essentially identical in meaning and scope - NOT if one is more specific than another or if they focus on different aspects of the same general area.

STRICT CRITERIA FOR DUPLICATES (all must be true):
1. Nearly identical wording or meaning
2. Same level of specificity 
3. Would drive the exact same actions in practice
4. No meaningful distinction in focus or approach

Examples of TRUE duplicates:
- "get to know this person" and "understand who I'm talking with" (identical meaning)
- "help with coding" and "assist with programming" (identical activity, same words)
- "maintain my appearance" and "keep looking good" (identical meaning)

Examples of NOT duplicates (should be allowed):
- "help with coding" and "help with debugging" (different specific activities)
- "maintain appearance" and "provide detailed appearance descriptions" (different focuses: maintaining vs communicating)
- "explore new interests" and "explore fashion" (general vs specific)
- "be attractive" and "get appearance right the first time" (different approaches: ongoing vs precision)
- "express creativity" and "design loungewear" (general vs specific creative outlet)

Key principle: Specific implementations of broader goals are NOT duplicates. Different approaches to similar areas are NOT duplicates. Only nearly identical priorities should be rejected.

Is the new priority truly redundant (not just related) to any existing priority?"""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=DuplicatePriorityCheck,
                model=model,
                llm=llm,
                caller="check_duplicate_priority",
            )
            return result
        except Exception as e:
            logger.warning(f"Failed to check for duplicate priorities: {e}")
            # Fall back to no duplicate detected
            return DuplicatePriorityCheck(
                is_duplicate=False,
                existing_priority_id=None,
                reasoning="Could not perform duplicate check due to error",
            )

    @classmethod
    def get_action_description(cls) -> str:
        return "Add a new priority - something I consciously choose to focus on"

    @classmethod
    def get_input_type(cls) -> Type[AddPriorityInput]:
        return AddPriorityInput

    def execute(
        self,
        action_input: AddPriorityInput,
        context: ExecutionContext,
        state: State,
        trigger_history: TriggerHistory,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult:
        start_time = time.time()

        logger.debug("=== ADD_PRIORITY ACTION ===")
        logger.debug(f"PRIORITY: {action_input.priority_content}")
        logger.debug(f"REASON: {action_input.reason}")

        # Check for duplicate or similar priorities using LLM
        if state.current_priorities:
            duplicate_check = self._check_for_duplicate_priority(
                action_input.priority_content, state.current_priorities, llm, model
            )

            if duplicate_check.is_duplicate:
                duration_ms = (time.time() - start_time) * 1000
                # Find the existing priority to get its content
                existing_priority = next(
                    (
                        p
                        for p in state.current_priorities
                        if p.id == duplicate_check.existing_priority_id
                    ),
                    None,
                )
                if existing_priority:
                    return ActionResult(
                        action=ActionType.ADD_PRIORITY,
                        result_summary=f"Priority '{action_input.priority_content}' is similar to existing priority '{existing_priority.content}' (id: {existing_priority.id}). {duplicate_check.reasoning}",
                        context_given=f"priority: {action_input.priority_content}, reason: {action_input.reason}",
                        duration_ms=duration_ms,
                        success=True,
                        metadata=None,
                    )
                else:
                    # Fallback if ID not found (shouldn't happen but be safe)
                    return ActionResult(
                        action=ActionType.ADD_PRIORITY,
                        result_summary=f"Priority '{action_input.priority_content}' appears to be a duplicate. {duplicate_check.reasoning}",
                        context_given=f"priority: {action_input.priority_content}, reason: {action_input.reason}",
                        duration_ms=duration_ms,
                        success=True,
                        metadata=None,
                    )

        # Generate new sequential ID
        new_id = f"p{state.next_priority_id}"
        state.next_priority_id += 1

        # Add the new priority
        new_priority = Priority(id=new_id, content=action_input.priority_content)
        state.current_priorities.append(new_priority)

        duration_ms = (time.time() - start_time) * 1000

        result_summary = f"Added new priority: '{action_input.priority_content}' because {action_input.reason}"

        return ActionResult(
            action=ActionType.ADD_PRIORITY,
            result_summary=result_summary,
            context_given=f"priority: {action_input.priority_content}, reason: {action_input.reason}",
            duration_ms=duration_ms,
            success=True,
            metadata=None,
        )


class RemovePriorityAction(BaseAction[RemovePriorityInput, None]):
    """Remove a priority that is no longer relevant"""

    action_type = ActionType.REMOVE_PRIORITY

    @classmethod
    def get_action_description(cls) -> str:
        return "Remove a priority that is no longer relevant or has been completed"

    @classmethod
    def get_input_type(cls) -> Type[RemovePriorityInput]:
        return RemovePriorityInput

    def execute(
        self,
        action_input: RemovePriorityInput,
        context: ExecutionContext,
        state: State,
        trigger_history: TriggerHistory,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult:
        start_time = time.time()

        logger.debug("=== REMOVE_PRIORITY ACTION ===")
        logger.debug(f"REMOVING ID: {action_input.priority_id}")
        logger.debug(f"REASON: {action_input.reason}")

        # Find and remove the priority by ID
        priority_found = None
        for i, priority in enumerate(state.current_priorities):
            if priority.id == action_input.priority_id:
                priority_found = state.current_priorities.pop(i)
                break

        duration_ms = (time.time() - start_time) * 1000

        if priority_found:
            result_summary = f"Removed priority '{priority_found.content}' (id: {priority_found.id}) because {action_input.reason}"
            success = True
        else:
            result_summary = (
                f"Priority with ID '{action_input.priority_id}' not found to remove"
            )
            success = False

        return ActionResult(
            action=ActionType.REMOVE_PRIORITY,
            result_summary=result_summary,
            context_given=f"priority_id: {action_input.priority_id}, reason: {action_input.reason}",
            duration_ms=duration_ms,
            success=success,
            metadata=None,
        )
