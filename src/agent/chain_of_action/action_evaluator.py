"""
Action evaluator for detecting and preventing repetitive action patterns.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from agent.chain_of_action.action_registry import ActionRegistry
from agent.chain_of_action.prompts import build_completed_action_list
from agent.chain_of_action.trigger import format_trigger_for_prompt

from .action_result import ActionResult
from .action_plan import ActionPlan
from ..structured_llm import direct_structured_llm_call

from agent.llm import LLM, SupportedModel
from .trigger import BaseTriger

logger = logging.getLogger(__name__)


class SequenceEvaluation(BaseModel):
    """Evaluation and correction of action sequences focusing on repetitive patterns"""

    has_repetition: bool = Field(
        description="Does the planned sequence show genuine repetitive patterns based on completed actions?"
    )
    pattern_detected: str = Field(
        description="What repetitive pattern was detected in the action history"
    )
    corrected_actions: List[ActionPlan] = Field(
        description="Corrected sequence that avoids repetition (only if truly repetitive)"
    )
    reasoning: str = Field(
        description="Conservative analysis of why sequence was or wasn't changed"
    )


def format_action_plan_for_prompt(action: ActionPlan) -> str:
    """
    Format action plan for prompt, ensuring it includes both action and context.
    """
    inputs = "\n".join(f"  - {key}: {value}" for key, value in action.input.items())
    return f"- {action.action.value}:\n{inputs}"


class ActionEvaluator:
    """Analyzes action sequences for repetitive patterns"""

    def __init__(self, registry: ActionRegistry):
        self.registry = registry

    def analyze_and_correct_sequence(
        self,
        planned_actions: List[ActionPlan],
        all_completed_actions: List[ActionResult],
        trigger: BaseTriger,
        llm: LLM,
        model: SupportedModel,
    ) -> SequenceEvaluation:
        """Analyze planned sequence against ALL completed actions and correct if repetitive"""

        trigger_description = format_trigger_for_prompt(trigger)
        actions_list = self.registry.get_available_actions_for_prompt()
        completed_actions_text = build_completed_action_list(all_completed_actions)
        planned_actions_text = "\n".join(
            format_action_plan_for_prompt(action) for action in planned_actions
        )

        prompt = f"""Analyze and correct this agent's planned action sequence to avoid repetitive patterns.

================================================================================
                                WHAT JUST HAPPENED
================================================================================
{trigger_description}

================================================================================
                                AVAILABLE ACTIONS
================================================================================
{actions_list}

================================================================================
                            ACTIONS I'VE ALREADY TAKEN
================================================================================
{completed_actions_text}

================================================================================
                                PLANNED ACTIONS
================================================================================
{planned_actions_text}

EVALUATION TASK:
1. Look for GENUINE repetitive patterns in the completed action history
2. Only flag as repetitive if there are clear cycles or redundant work
3. Be conservative - don't optimize unless there's actual repetition

REPETITION DETECTION GUIDELINES:
- Multiple THINK actions with similar content = repetitive
- Agent cycling between same action types without progress = repetitive
- Agent has already adequately addressed the trigger = repetitive
- First-time actions are NOT repetitive (e.g., first THINK is normal)

CORRECTION APPROACH:
- Only suggest DONE if genuine repetition is detected
- Preserve natural conversation flow when no repetition exists
- Focus on preventing loops, not optimizing sequences

IMPORTANT: All actions in corrected_actions MUST include both "action" and "context" fields.
For DONE actions, provide a context explaining why completion is appropriate."""

        try:
            analysis = direct_structured_llm_call(
                prompt=prompt,
                response_model=SequenceEvaluation,
                model=model,
                llm=llm,
                temperature=0.3,  # Lower temperature for consistent analysis
                caller="action_evaluator",
            )

            logger.debug(f"=== SEQUENCE ANALYSIS ===")
            logger.debug(f"HAS_REPETITION: {analysis.has_repetition}")
            logger.debug(f"PATTERN: {analysis.pattern_detected}")
            logger.debug(f"ORIGINAL: {len(planned_actions)} actions")
            logger.debug(f"CORRECTED: {len(analysis.corrected_actions)} actions")
            logger.debug(f"REASONING: {analysis.reasoning}")
            logger.debug("=" * 40)

            return analysis

        except Exception as e:
            logger.error(f"Sequence analysis failed: {e}")
            # Default to keeping original sequence if analysis fails
            return SequenceEvaluation(
                has_repetition=False,
                pattern_detected="Analysis failed",
                corrected_actions=planned_actions,
                reasoning=f"Could not analyze due to error: {e}",
            )
