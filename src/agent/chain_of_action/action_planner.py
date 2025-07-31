"""
Action planner for generating action sequences from triggers.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from agent.chain_of_action.context import ActionResult

from .action_types import ActionType
from .action_plan import ActionPlan, ActionSequence
from .action_registry import ActionRegistry
from ..structured_llm import direct_structured_llm_call

from agent.state import State
from agent.conversation_history import ConversationHistory
from agent.llm import LLM, SupportedModel
from .trigger import TriggerEvent

logger = logging.getLogger(__name__)


class PlannedAction(BaseModel):
    """A single planned action with context"""

    action: ActionType = Field(description="The action type to execute")
    context: str = Field(description="Specific context or focus for this action")


class ActionPlanningResult(BaseModel):
    """Result of action planning with sequence of actions"""

    reasoning: str = Field(
        description="My reasoning for why I chose this sequence of actions"
    )
    actions: List[PlannedAction] = Field(
        description="Planned sequence of actions to execute"
    )


class ActionPlanner:
    """Plans action sequences based on triggers and current state"""

    def __init__(self, registry: ActionRegistry):
        self.registry = registry

    def plan_actions(
        self,
        trigger: TriggerEvent,
        completed_actions: List[ActionResult],
        state: State,
        conversation_history: ConversationHistory,
        llm: LLM,
        model: SupportedModel,
    ) -> ActionSequence:
        """Plan a sequence of actions to respond to the trigger"""

        # Use the extracted prompt function
        from .prompts import build_action_planning_prompt

        planning_prompt = build_action_planning_prompt(
            state=state,
            trigger=trigger,
            completed_actions=completed_actions,
            conversation_history=conversation_history,
            registry=self.registry,
        )

        try:
            logger.debug("=== ACTION PLANNING PROMPT ===")
            logger.debug(planning_prompt)

            # Use structured LLM to plan the actions
            result = direct_structured_llm_call(
                prompt=planning_prompt,
                response_model=ActionSequence,
                model=model,
                llm=llm,
                temperature=0.3,
            )

            logger.debug("=== ACTION PLANNING ===")
            logger.debug(f"TRIGGER: {trigger}")
            logger.debug(f"REASONING: {result.reasoning}")
            logger.debug(f"PLANNED: {len(result.actions)} actions")
            for i, action in enumerate(result.actions):
                logger.debug(f"  {i+1}. {action.action.value}: {action.context}")
            logger.debug("=" * 40)

            return result

        except Exception as e:
            logger.error(f"Action planning failed: {e}")
            # Fallback to just thinking
            return ActionSequence(
                actions=[],
                can_extend=False,
                reasoning="I couldn't plan any actions due to an error, so I won't do anything for now.",
            )
