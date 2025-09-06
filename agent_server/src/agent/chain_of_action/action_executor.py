"""
Action executor for running action sequences.
"""

import logging
import time

from typing import List

from agent.chain_of_action.action.action_data import ActionData, create_action_data
from agent.chain_of_action.action.base_action_data import (
    ActionFailureResult,
    BaseActionData,
)
from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry

from .action_plan import ActionSequence
from .action_registry import ActionRegistry
from .context import ExecutionContext
from .callbacks import ActionCallback, NoOpCallback

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Executes action sequences and manages execution context"""

    def __init__(self, registry: ActionRegistry):
        self.registry = registry

    def execute_sequence(
        self,
        sequence: ActionSequence,
        context: ExecutionContext,
        state: State,
        trigger_history: TriggerHistory,
        llm: LLM,
        model: SupportedModel,
        sequence_number: int,
        callback: ActionCallback,
        trigger_entry: TriggerHistoryEntry,
    ) -> List[BaseActionData]:
        """Execute a complete action sequence"""

        if callback is None:
            callback = NoOpCallback()

        logger.debug(f"=== EXECUTING ACTION SEQUENCE ===")
        logger.debug(f"ACTIONS: {len(sequence.actions)}")
        logger.debug(f"PRIOR COMPLETED: {len(context.completed_actions)}")

        # Update context with planned actions for this sequence
        context.planned_actions = sequence.actions
        context.current_action_index = 0

        # Notify sequence started
        callback.on_sequence_started(sequence_number, len(sequence.actions), "")

        results: List[BaseActionData] = []

        for i, action_plan in enumerate(sequence.actions):
            logger.debug(
                f"Sending action start event for action {i+1}/{len(sequence.actions)}: {action_plan.action.value}"
            )
            input_summary = ", ".join(
                [f"{k}: {v}" for k, v in action_plan.input.items()]
            )
            callback.on_action_started(
                action_plan.action,
                input_summary,
                sequence_number,
                i + 1,
                trigger_entry.entry_id,
            )

        # Execute each action in sequence
        for i, action_plan in enumerate(sequence.actions):
            logger.debug(
                f"Executing action {i+1}/{len(sequence.actions)}: {action_plan.action.value}"
            )

            # Update current action index
            context.current_action_index = i

            action_input = None
            try:
                # Create action instance
                action = self.registry.create_action(action_plan.action)

                # Execute the action with progress callback
                def action_progress_callback(data):
                    # Forward progress events via main callback
                    callback.on_action_progress(
                        action_plan.action,
                        data,
                        sequence_number,
                        i + 1,
                        trigger_entry.entry_id,
                    )

                # Create the validated action input
                action_class = self.registry.get_action(action_plan.action)
                input_type = action_class.get_input_type()
                action_input = input_type(**action_plan.input)

                start_time = time.time()
                result = action.execute(
                    action_input=action_input,
                    context=context,
                    state=state,
                    llm=llm,
                    model=model,
                    progress_callback=action_progress_callback,
                )
                duration_ms = (time.time() - start_time) * 1000

                action_data = create_action_data(
                    type=action_plan.action,
                    reasoning=action_plan.reasoning,
                    input=action_input,
                    result=result,
                    duration_ms=duration_ms,
                )
                # Add result to list
                results.append(action_data)

                # Update context with completed action
                context.completed_actions.append(action_data)

                # Notify action finished
                callback.on_action_finished(
                    action_plan.action,
                    action_data,
                    sequence_number,
                    i + 1,
                    trigger_entry.entry_id,
                )

                logger.debug(
                    f"Action completed: {action_data.type} ({action_data.result.type})"
                )
                if action_data.result.type == "success":
                    logger.debug(
                        f"Result: {action_data.result.content.result_summary()[500:]}..."
                    )

            except Exception as e:
                # Catastrophic failure - action couldn't be created or executed
                logger.error(
                    f"Catastrophic failure executing {action_plan.action.value}: {e}",
                    exc_info=True,
                )

                error_data = create_action_data(
                    type=action_plan.action,
                    reasoning=action_plan.reasoning,
                    input=action_input,
                    result=ActionFailureResult(error=str(e)),
                    duration_ms=0.0,
                )

                results.append(error_data)
                context.completed_actions.append(error_data)

                # Notify action finished with error
                callback.on_action_finished(
                    action_plan.action,
                    error_data,
                    sequence_number,
                    i + 1,
                    trigger_entry.entry_id,
                )

                # Stop execution - this is a serious system failure
                logger.error(f"Stopping sequence execution due to catastrophic failure")
                break

        # Notify sequence finished
        successful = sum(1 for r in results if r.result.type == "success")
        callback.on_sequence_finished(sequence_number, len(results), successful)

        logger.debug(f"=== SEQUENCE EXECUTION COMPLETE ===")
        logger.debug(f"Total actions: {len(results)}")
        logger.debug(f"Successful: {successful}/{len(results)}")

        return results
