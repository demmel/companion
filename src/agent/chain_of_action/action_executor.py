"""
Action executor for running action sequences.
"""

import logging
from typing import List

from agent.chain_of_action.trigger_history import TriggerHistory

from .action_plan import ActionSequence
from .action_registry import ActionRegistry
from .context import ExecutionContext, ActionResult
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
    ) -> List[ActionResult]:
        """Execute a complete action sequence"""

        if callback is None:
            callback = NoOpCallback()

        logger.debug(f"=== EXECUTING ACTION SEQUENCE ===")
        logger.debug(f"ACTIONS: {len(sequence.actions)}")
        logger.debug(f"PRIOR COMPLETED: {len(context.completed_actions)}")

        # Notify sequence started
        callback.on_sequence_started(
            sequence_number, len(sequence.actions), sequence.reasoning
        )

        results = []

        # Execute each action in sequence
        for i, action_plan in enumerate(sequence.actions):
            logger.debug(
                f"Executing action {i+1}/{len(sequence.actions)}: {action_plan.action.value}"
            )

            # Notify action started
            callback.on_action_started(
                action_plan.action, action_plan.context, sequence_number, i + 1
            )

            try:
                # Create action instance
                action = self.registry.create_action(action_plan.action)

                # Execute the action with progress callback
                def action_progress_callback(data):
                    # Forward progress events via main callback
                    callback.on_action_progress(
                        action_plan.action, data, sequence_number, i + 1
                    )

                result = action.execute(
                    action_plan=action_plan,
                    context=context,
                    state=state,
                    trigger_history=trigger_history,
                    llm=llm,
                    model=model,
                    progress_callback=action_progress_callback,
                )

                # Add result to list
                results.append(result)

                # Update context with completed action
                context.completed_actions.append(result)

                # Notify action finished
                callback.on_action_finished(
                    action_plan.action, result, sequence_number, i + 1
                )

                logger.debug(
                    f"Action completed: {result.action.value} ({'success' if result.success else 'failed'})"
                )
                if result.result_summary:
                    logger.debug(f"Result: {result.result_summary[:100]}...")

            except Exception as e:
                # Catastrophic failure - action couldn't be created or executed
                logger.error(
                    f"Catastrophic failure executing {action_plan.action.value}: {e}",
                    exc_info=True,
                )

                # Create error result
                error_result = ActionResult(
                    action=action_plan.action,
                    result_summary="",
                    context_given=action_plan.context,
                    duration_ms=0.0,
                    success=False,
                    error=f"Execution exception: {str(e)}",
                )

                results.append(error_result)
                context.completed_actions.append(error_result)

                # Notify action finished with error
                callback.on_action_finished(
                    action_plan.action, error_result, sequence_number, i + 1
                )

                # Stop execution - this is a serious system failure
                logger.error(f"Stopping sequence execution due to catastrophic failure")
                break

        # Notify sequence finished
        successful = sum(1 for r in results if r.success)
        callback.on_sequence_finished(sequence_number, len(results), successful)

        logger.debug(f"=== SEQUENCE EXECUTION COMPLETE ===")
        logger.debug(f"Total actions: {len(results)}")
        logger.debug(f"Successful: {successful}/{len(results)}")

        return results
