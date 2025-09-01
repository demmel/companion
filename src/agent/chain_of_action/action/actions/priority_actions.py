"""
Priority management actions implementation.
"""

import time
import logging
from typing import Literal, Type, Union

from pydantic import BaseModel, Field
from typing import Optional

from agent.state import Priority

from ..action_types import ActionType
from ..base_action import BaseAction
from ..base_action_data import (
    ActionFailureResult,
    ActionOutput,
    ActionResult,
    ActionSuccessResult,
)
from agent.chain_of_action.context import ExecutionContext

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


class AddPrioritySuccessOutput(BaseModel):
    """Output for successful ADD_PRIORITY action"""

    type: Literal["success"] = "success"
    priority_id: str
    reason: str


class AddPriorityDuplicateOutput(BaseModel):
    """Output for duplicate ADD_PRIORITY action"""

    type: Literal["duplicate"] = "duplicate"
    existing_priority_id: str | None = None
    existing_priority_content: str | None = None
    reason: str


class AddPriorityOutput(ActionOutput):
    """Output for ADD_PRIORITY action"""

    content: str
    result: Union[AddPrioritySuccessOutput, AddPriorityDuplicateOutput]

    def result_summary(self) -> str:
        result = self.result
        match result.type:
            case "success":
                return f"Added new priority: '{self.content}' because {result.reason}"
            case "duplicate":
                if result.existing_priority_content:
                    return f"Priority '{self.content}' is similar to existing priority '{result.existing_priority_content}' (id: {result.existing_priority_id}). {result.reason}"
                else:
                    return f"Priority '{self.content}' appears to be a duplicate. {result.reason}"
            case _:
                return "Unknown result type"


class AddPriorityAction(BaseAction[AddPriorityInput, AddPriorityOutput]):
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
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult[AddPriorityOutput]:
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
                # Find the existing priority to get its content
                existing_priority = next(
                    (
                        p
                        for p in state.current_priorities
                        if p.id == duplicate_check.existing_priority_id
                    ),
                    None,
                )
                return ActionSuccessResult(
                    content=AddPriorityOutput(
                        content=action_input.priority_content,
                        result=AddPriorityDuplicateOutput(
                            existing_priority_id=(
                                duplicate_check.existing_priority_id
                                if duplicate_check.existing_priority_id
                                else None
                            ),
                            existing_priority_content=(
                                existing_priority.content if existing_priority else None
                            ),
                            reason=action_input.reason,
                        ),
                    )
                )

        # Generate new sequential ID
        new_id = f"p{state.next_priority_id}"
        state.next_priority_id += 1

        # Add the new priority
        new_priority = Priority(id=new_id, content=action_input.priority_content)
        state.current_priorities.append(new_priority)

        return ActionSuccessResult(
            content=AddPriorityOutput(
                content=action_input.priority_content,
                result=AddPrioritySuccessOutput(
                    priority_id=new_id, reason=action_input.reason
                ),
            )
        )


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


class RemovePriorityAction(BaseAction[RemovePriorityInput, RemovePriorityOutput]):
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
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult[RemovePriorityOutput]:

        logger.debug("=== REMOVE_PRIORITY ACTION ===")
        logger.debug(f"REMOVING ID: {action_input.priority_id}")
        logger.debug(f"REASON: {action_input.reason}")

        # Find and remove the priority by ID
        priority_found = None
        for i, priority in enumerate(state.current_priorities):
            if priority.id == action_input.priority_id:
                priority_found = state.current_priorities.pop(i)
                break

        if priority_found:
            return ActionSuccessResult(
                content=RemovePriorityOutput(
                    priority=priority_found, reason=action_input.reason
                )
            )
        else:
            return ActionFailureResult(
                error=f"Priority with ID '{action_input.priority_id}' not found"
            )
