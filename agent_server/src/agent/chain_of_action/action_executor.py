"""
Action executor for running action sequences.
"""

import logging
import time

from typing import List

from agent.chain_of_action.action.action_data import ActionData, create_action_data
from agent.chain_of_action.action.action_types import ActionType
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
                action_plan.reasoning,
            )

        # Create execution units (handles batching optimization)
        from .execution_unit import create_execution_units

        execution_units = create_execution_units(sequence.actions)
        logger.debug(
            f"Created {len(execution_units)} execution units from {len(sequence.actions)} actions"
        )

        # Execute each unit
        for unit in execution_units:
            logger.debug(f"Executing unit with {unit.get_action_count()} actions")

            unit_results = unit.execute(
                context,
                state,
                llm,
                model,
                sequence_number,
                callback,
                trigger_entry,
                self.registry,
            )

            results.extend(unit_results)

            # Check for catastrophic failures
            if any(result.result.type == "failure" for result in unit_results):
                logger.debug("Unit completed with failures")
                # Continue with other units unless it's a catastrophic system failure
            else:
                logger.debug("Unit completed successfully")

        # Notify sequence finished
        successful = sum(1 for r in results if r.result.type == "success")
        callback.on_sequence_finished(sequence_number, len(results), successful)

        logger.debug(f"=== SEQUENCE EXECUTION COMPLETE ===")
        logger.debug(f"Total actions: {len(results)}")
        logger.debug(f"Successful: {successful}/{len(results)}")

        return results
