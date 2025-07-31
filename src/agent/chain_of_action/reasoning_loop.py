"""
Main action-based reasoning loop.
"""

import logging
from typing import List

from .action_registry import ActionRegistry
from .action_planner import ActionPlanner
from .action_executor import ActionExecutor
from .action_evaluator import ActionEvaluator
from .trigger import UserInputTrigger
from .context import ActionResult
from .callbacks import ActionCallback, NoOpCallback
from agent.state import State
from agent.conversation_history import ConversationHistory
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class ActionBasedReasoningLoop:
    """Main orchestrator for action-based agent reasoning"""

    def __init__(self):
        self.registry = ActionRegistry()
        self.planner = ActionPlanner(self.registry)
        self.executor = ActionExecutor(self.registry)
        self.action_evaluator = ActionEvaluator(self.registry)

    def process_user_input(
        self,
        user_input: str,
        user_name: str,
        state: "State",
        conversation_history: "ConversationHistory",
        llm: "LLM",
        model: "SupportedModel",
        callback: ActionCallback,
    ) -> List[ActionResult]:
        """
        Process user input through the action-based reasoning system.

        Executes sequences of actions, then asks agent if they want to continue
        with more sequences until they decide to stop. Repetition detector prevents loops.

        Returns list of ActionResult objects from all executed sequences.
        """

        if callback is None:
            callback = NoOpCallback()

        logger.debug(f"=== PROCESSING USER INPUT ===")
        logger.debug(f"INPUT: {user_input}")
        logger.debug(f"USER: {user_name}")

        # Create trigger from user input
        trigger = UserInputTrigger(content=user_input, user_name=user_name)

        # Create execution context for the full chain
        from .context import ExecutionContext
        import uuid

        context = ExecutionContext(
            trigger=trigger, completed_actions=[], session_id=str(uuid.uuid4())
        )

        sequence_num = 0

        while True:
            sequence_num += 1
            logger.debug(f"Planning sequence {sequence_num}...")

            # Plan next sequence of actions, showing what's already been done
            sequence = self.planner.plan_actions(
                trigger=trigger,
                completed_actions=context.completed_actions,  # Show planner what's already done
                state=state,
                conversation_history=conversation_history,
                llm=llm,
                model=model,
            )

            if not sequence.actions:
                logger.debug("No more actions planned, stopping chain")
                break

            # Evaluate sequence for repetitive patterns and correct if needed
            logger.debug("Evaluating sequence for repetitive patterns...")
            evaluation = self.action_evaluator.analyze_and_correct_sequence(
                planned_actions=sequence.actions,
                all_completed_actions=context.completed_actions,
                trigger=trigger,
                llm=llm,
                model=model,
            )

            # Notify about evaluation
            callback.on_evaluation(
                evaluation.has_repetition,
                evaluation.pattern_detected,
                len(sequence.actions),
                len(evaluation.corrected_actions),
            )

            # Use corrected sequence if evaluation found genuine repetition
            if evaluation.has_repetition:
                logger.debug(
                    f"Repetitive pattern detected: {evaluation.pattern_detected}"
                )
                logger.debug(
                    f"Using corrected sequence with {len(evaluation.corrected_actions)} actions"
                )
                sequence.actions = evaluation.corrected_actions
            else:
                logger.debug("No repetitive patterns detected, using original sequence")

            # Execute the planned sequence, updating the shared context
            logger.debug(
                f"Executing sequence {sequence_num} with {len(sequence.actions)} actions..."
            )
            sequence_results = self.executor.execute_sequence(
                sequence=sequence,
                context=context,  # Pass shared context
                state=state,
                conversation_history=conversation_history,
                llm=llm,
                model=model,
                sequence_number=sequence_num,
                callback=callback,
            )

            # Results are already added to context.completed_actions by executor

            # Check if agent signaled completion with DONE action
            done_actions = [
                r for r in sequence_results if r.action.value == "done" and r.success
            ]
            if done_actions:
                logger.debug(
                    f"Agent signaled completion: {done_actions[0].result_summary}"
                )
                break

        # Notify processing complete
        callback.on_processing_complete(sequence_num, len(context.completed_actions))

        logger.debug(f"=== PROCESSING COMPLETE ===")
        logger.debug(
            f"Executed {len(context.completed_actions)} total actions across {sequence_num} sequences"
        )

        return context.completed_actions
