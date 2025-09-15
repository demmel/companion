"""
Execution units for action batching and optimization.
"""

from abc import ABC, abstractmethod
from typing import List
import time
import logging

from agent.chain_of_action.action.action_data import ActionData, create_action_data
from agent.chain_of_action.action.base_action_data import ActionFailureResult
from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.context import ExecutionContext
from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class ExecutionUnit(ABC):
    """Base class for units of action execution"""

    @abstractmethod
    def execute(
        self,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
        sequence_number: int,
        callback,
        trigger_entry,
        action_registry,
    ) -> List[ActionData]:
        """Execute this unit and return action data results"""
        pass

    @abstractmethod
    def get_action_count(self) -> int:
        """Number of planned actions this unit represents"""
        pass

    @abstractmethod
    def get_action_plans(self) -> List:
        """Get the original action plans this unit represents"""
        pass


class SingleActionUnit(ExecutionUnit):
    """Execution unit for a single action"""

    def __init__(self, action_plan, action_index: int):
        self.action_plan = action_plan
        self.action_index = action_index

    def get_action_count(self) -> int:
        return 1

    def get_action_plans(self) -> List:
        return [self.action_plan]

    def execute(
        self,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
        sequence_number: int,
        callback,
        trigger_entry,
        action_registry,
    ) -> List[ActionData]:
        """Execute single action using existing logic"""

        context.current_action_index = self.action_index
        action_input = None

        try:
            # Create action instance
            action = action_registry.create_action(self.action_plan.action)

            # Execute the action with progress callback
            def action_progress_callback(data):
                callback.on_action_progress(
                    self.action_plan.action,
                    data,
                    sequence_number,
                    self.action_index + 1,
                    trigger_entry.entry_id,
                )

            # Create the validated action input
            action_class = action_registry.get_action(self.action_plan.action)
            input_type = action_class.get_input_type()
            action_input = input_type(**self.action_plan.input)

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
                type=self.action_plan.action,
                reasoning=self.action_plan.reasoning,
                input=action_input,
                result=result,
                duration_ms=duration_ms,
            )

            # Update context with completed action
            context.completed_actions.append(action_data)

            # Notify action finished
            callback.on_action_finished(
                self.action_plan.action,
                action_data,
                sequence_number,
                self.action_index + 1,
                trigger_entry.entry_id,
            )

            logger.debug(f"Action completed: {action_data.type} ({action_data.result.type})")
            if action_data.result.type == "success":
                logger.debug(f"Result: {action_data.result.content.result_summary()[:500]}...")

            return [action_data]

        except Exception as e:
            logger.error(f"Catastrophic failure executing {self.action_plan.action.value}: {e}", exc_info=True)

            error_data = create_action_data(
                type=self.action_plan.action,
                reasoning=self.action_plan.reasoning,
                input=action_input,
                result=ActionFailureResult(error=str(e)),
                duration_ms=0.0,
            )

            context.completed_actions.append(error_data)

            callback.on_action_finished(
                self.action_plan.action,
                error_data,
                sequence_number,
                self.action_index + 1,
                trigger_entry.entry_id,
            )

            return [error_data]


class VisualBatchUnit(ExecutionUnit):
    """Execution unit for batched visual actions"""

    def __init__(self, action1_plan, action2_plan, start_index: int):
        self.action1_plan = action1_plan
        self.action2_plan = action2_plan
        self.start_index = start_index

    def get_action_count(self) -> int:
        return 2

    def get_action_plans(self) -> List:
        return [self.action1_plan, self.action2_plan]

    def execute(
        self,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
        sequence_number: int,
        callback,
        trigger_entry,
        action_registry,
    ) -> List[ActionData]:
        """Execute visual batch using shared logic"""
        from .action.actions.visual_actions import (
            execute_visual_actions_batch,
            UpdateAppearanceInput,
            UpdateEnvironmentInput,
        )

        logger.debug(f"Batching visual actions: {self.action1_plan.action} + {self.action2_plan.action}")

        # Determine which is which
        appearance_plan = self.action1_plan if self.action1_plan.action == ActionType.UPDATE_APPEARANCE else self.action2_plan
        environment_plan = self.action1_plan if self.action1_plan.action == ActionType.UPDATE_ENVIRONMENT else self.action2_plan

        # Create inputs
        appearance_input = UpdateAppearanceInput(**appearance_plan.input)
        environment_input = UpdateEnvironmentInput(**environment_plan.input)

        # Execute batch
        start_time = time.time()

        def batch_progress_callback(data):
            # Forward progress for the second action (the one generating the image)
            callback.on_action_progress(
                self.action2_plan.action,
                data,
                sequence_number,
                self.start_index + 2,
                trigger_entry.entry_id,
            )

        try:
            appearance_result, environment_result = execute_visual_actions_batch(
                appearance_input, environment_input, context, state, llm, model, batch_progress_callback
            )
            duration_ms = (time.time() - start_time) * 1000

            # Create action data for both actions
            appearance_data = create_action_data(
                type=ActionType.UPDATE_APPEARANCE,
                reasoning=appearance_plan.reasoning,
                input=appearance_input,
                result=appearance_result,
                duration_ms=duration_ms,
            )

            environment_data = create_action_data(
                type=ActionType.UPDATE_ENVIRONMENT,
                reasoning=environment_plan.reasoning,
                input=environment_input,
                result=environment_result,
                duration_ms=duration_ms,
            )

            # Return results in original order
            if self.action1_plan.action == ActionType.UPDATE_APPEARANCE:
                results = [appearance_data, environment_data]
            else:
                results = [environment_data, appearance_data]

            # Update context with completed actions
            context.completed_actions.extend(results)

            # Notify actions finished
            for i, (action_plan, action_data) in enumerate(zip([self.action1_plan, self.action2_plan], results)):
                callback.on_action_finished(
                    action_plan.action,
                    action_data,
                    sequence_number,
                    self.start_index + i + 1,
                    trigger_entry.entry_id,
                )

            logger.debug("Batched visual actions completed successfully")
            return results

        except Exception as e:
            logger.error(f"Failed to execute visual batch: {e}", exc_info=True)

            # Create error data for both actions
            error_data1 = create_action_data(
                type=self.action1_plan.action,
                reasoning=self.action1_plan.reasoning,
                input=None,
                result=ActionFailureResult(error=str(e)),
                duration_ms=0.0,
            )

            error_data2 = create_action_data(
                type=self.action2_plan.action,
                reasoning=self.action2_plan.reasoning,
                input=None,
                result=ActionFailureResult(error=str(e)),
                duration_ms=0.0,
            )

            results = [error_data1, error_data2]
            context.completed_actions.extend(results)

            # Notify actions finished with errors
            for i, (action_plan, error_data) in enumerate(zip([self.action1_plan, self.action2_plan], results)):
                callback.on_action_finished(
                    action_plan.action,
                    error_data,
                    sequence_number,
                    self.start_index + i + 1,
                    trigger_entry.entry_id,
                )

            return results


def create_execution_units(action_plans: List) -> List[ExecutionUnit]:
    """Convert a list of action plans into optimized execution units"""
    units = []
    i = 0

    while i < len(action_plans):
        action_plan = action_plans[i]

        # Check if we should batch this action with the next one
        if (i + 1 < len(action_plans) and
            _should_batch_visual_actions(action_plan.action, action_plans[i + 1].action)):

            # Create visual batch unit
            units.append(VisualBatchUnit(action_plan, action_plans[i + 1], i))
            i += 2  # Skip both actions since we batched them
        else:
            # Create single action unit
            units.append(SingleActionUnit(action_plan, i))
            i += 1

    return units


def _should_batch_visual_actions(action1_type: ActionType, action2_type: ActionType) -> bool:
    """Check if two consecutive actions should be batched (different visual actions)"""
    visual_actions = {ActionType.UPDATE_APPEARANCE, ActionType.UPDATE_ENVIRONMENT}
    return (action1_type in visual_actions and
            action2_type in visual_actions and
            action1_type != action2_type)