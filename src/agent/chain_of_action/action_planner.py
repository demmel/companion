"""
Action planner for generating action sequences from triggers.
"""

import logging
from typing import List

from agent.chain_of_action.context import ActionResult
from agent.chain_of_action.trigger_history import TriggerHistory

from .action_plan import ActionSequence
from .action_registry import ActionRegistry
from ..structured_llm import direct_structured_llm_call

from agent.state import State
from agent.llm import LLM, SupportedModel
from .trigger import BaseTriger

logger = logging.getLogger(__name__)


class ActionPlanner:
    """Plans action sequences based on triggers and current state"""

    def __init__(self, registry: ActionRegistry):
        self.registry = registry

    def plan_actions(
        self,
        trigger: BaseTriger,
        completed_actions: List[ActionResult],
        state: State,
        trigger_history: TriggerHistory,
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
            trigger_history=trigger_history,
            registry=self.registry,
        )

        max_retries = 2
        current_prompt = planning_prompt

        for retry_attempt in range(max_retries + 1):
            logger.debug(
                f"=== ACTION PLANNING {'RETRY ' + str(retry_attempt) if retry_attempt > 0 else ''} ==="
            )
            logger.debug(current_prompt)

            # Use structured LLM to plan the actions
            result = direct_structured_llm_call(
                prompt=current_prompt,
                response_model=ActionSequence,
                model=model,
                llm=llm,
                temperature=0.3,
                caller="action_planner",
            )

            # Validate the structured inputs
            validation_errors = []
            for i, action_plan in enumerate(result.actions):
                try:
                    action_class = self.registry.get_action(action_plan.action)
                    input_type = action_class.get_input_type()
                    validated_input = input_type(**action_plan.input)
                    logger.debug(f"  {i+1}. {action_plan.action.value}: validated")

                except Exception as validation_error:
                    error_msg = f"Action {i+1} ({action_plan.action.value}): {str(validation_error)}"
                    validation_errors.append(error_msg)
                    logger.error(f"Validation error: {error_msg}")

            # If validation successful, return result
            if not validation_errors:
                logger.debug(f"PLANNED: {len(result.actions)} actions successfully")
                return result

            # If we have retries left, update prompt and continue
            if retry_attempt < max_retries:
                error_details = "\n".join([f"- {error}" for error in validation_errors])

                # Show the agent the JSON they produced
                attempted_json = result.model_dump_json(indent=2)

                current_prompt = f"""{planning_prompt}

**IMPORTANT: Your previous action planning had validation errors.**

**Your previous JSON response was:**
```json
{attempted_json}
```

**Validation errors that occurred:**
{error_details}

Please fix these validation errors. Make sure each action's input parameters match the required schema exactly. Check field names, types, and required vs optional fields. Provide a corrected action plan."""
            else:
                # Final attempt failed
                error_details = "\n".join([f"- {error}" for error in validation_errors])
                raise ValueError(
                    f"Action planning failed validation after {max_retries} retries:\n{error_details}"
                )

        # If we reach here, something went wrong
        raise RuntimeError(
            "Action planning did not complete successfully after retries"
        )
