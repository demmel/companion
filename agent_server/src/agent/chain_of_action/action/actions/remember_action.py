"""
REMEMBER action implementation.

Deliberate recall: the agent issues explicit memory queries to pull relevant memories from
long-term memory into its working context. This replaces the previous automatic per-trigger
retrieval — recall is now a deliberate action the agent chooses to take.
"""

import logging

from agent.chain_of_action.context import ExecutionContext
from agent.llm import LLM
from agent.memory.queries import MemoryQueries
from agent.state import State

from ..action_types import ActionType
from ..base_action import BaseAction, register_action
from ..base_action_data import (
    ActionFailureResult,
    ActionResult,
    ActionSuccessResult,
)
from ..data.remember_data import (
    RememberInput,
    RememberOutput,
    RememberActionData,
    RetrievedMemoryItem,
)

logger = logging.getLogger(__name__)


@register_action(ActionType.REMEMBER)
class RememberAction(BaseAction[RememberInput, RememberOutput]):
    """Deliberately recall relevant memories from long-term memory"""

    @classmethod
    def get_action_description(cls) -> str:
        return (
            "Deliberately recall relevant memories from my long-term memory by issuing explicit "
            "search queries. My memories are NOT retrieved automatically — I only have whatever is "
            "already in my working context. Use this whenever I need to ground myself in past "
            "interactions, facts, decisions, or relationship history before thinking or responding. "
            "Retrieved memories become part of my context for the rest of this turn."
        )

    def execute(
        self,
        action_input: RememberInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        progress_callback,
    ) -> ActionResult[RememberOutput]:
        try:
            memory_queries = MemoryQueries(
                queries=action_input.queries,
                max_tokens=context.memory_token_budget,
            )

            # Pure read: retrieve relevant memories without mutating context...
            retrieved = context.memory.query(
                memory_queries,
                llm=llm,
                model=context.memory_retrieval_model,
            )
            # ...then deliberately fold them into the working context.
            context.memory.reinforce(
                retrieved,
                budget=context.memory_token_budget,
                llm=llm,
                model=context.memory_retrieval_model,
            )

            logger.info(
                f"Remember action ran {len(action_input.queries)} queries: {action_input.reason}"
            )

            memories = [
                RetrievedMemoryItem(
                    memory_id=element.id,
                    content=element.content,
                    timestamp=element.timestamp,
                    confidence=element.confidence_level.value,
                )
                for element in retrieved.elements
            ]
            return ActionSuccessResult(content=RememberOutput(memories=memories))

        except Exception as e:
            error_msg = f"Failed to recall memories: {str(e)}"
            logger.error(error_msg)
            return ActionFailureResult(error=error_msg)
