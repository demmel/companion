"""
Main action-based reasoning loop.
"""

import logging
import uuid
from typing import List

from .action_registry import ActionRegistry
from .action_planner import ActionPlanner
from .action_executor import ActionExecutor
from .action_evaluator import ActionEvaluator
from .trigger import Trigger
from .action_result import ActionResult
from .callbacks import ActionCallback, NoOpCallback
from agent.state import State
from agent.llm import LLM, SupportedModel
from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry

logger = logging.getLogger(__name__)


class ActionBasedReasoningLoop:
    """Main orchestrator for action-based agent reasoning"""

    def __init__(
        self,
        enable_image_generation: bool = True,
        enable_action_evaluation: bool = True,
    ):
        self.registry = ActionRegistry(enable_image_generation=enable_image_generation)
        self.planner = ActionPlanner(self.registry)
        self.executor = ActionExecutor(self.registry)
        self.enable_action_evaluation = enable_action_evaluation
        if enable_action_evaluation:
            self.action_evaluator = ActionEvaluator(self.registry)
        else:
            self.action_evaluator = None

    def process_trigger(
        self,
        trigger: Trigger,
        state: State,
        llm: LLM,
        model: SupportedModel,
        callback: ActionCallback,
        trigger_history: TriggerHistory,
        individual_trigger_compression: bool = True,
    ) -> List[ActionResult]:
        """
        Process user input through the action-based reasoning system.

        Executes sequences of actions, then asks agent if they want to continue
        with more sequences until they decide to stop. Repetition detector prevents loops.

        Returns list of ActionResult objects from all executed sequences.
        """

        if callback is None:
            callback = NoOpCallback()

        logger.debug(f"=== PROCESSING TRIGGER ===")
        logger.debug(f"TRIGGER: {trigger}")
        logger.debug(f"TRIGGER_HISTORY: {trigger_history}")

        # Create trigger history entry (which generates the entry_id)
        from .trigger_history import TriggerHistoryEntry

        trigger_entry = TriggerHistoryEntry(
            trigger=trigger,
            actions_taken=[],  # Will be populated as actions complete
        )
        entry_id = trigger_entry.entry_id

        # Notify callback about trigger start
        callback.on_trigger_started(entry_id, trigger)

        # Extract memory queries and retrieve relevant memories
        from agent.memory.memory_extraction import (
            extract_memory_queries,
            retrieve_relevant_memories,
        )

        memory_extraction = extract_memory_queries(
            state=state,
            trigger=trigger,
            trigger_history=trigger_history,
            llm=llm,
            model=model,
        )

        relevant_memories = (
            retrieve_relevant_memories(
                memory_query=memory_extraction,
                trigger_history=trigger_history,
                max_results=5,
            )
            if memory_extraction
            else []
        )

        # Perform situational analysis once before action planning loop
        from .prompts import build_situational_analysis_prompt

        situational_analysis_prompt = build_situational_analysis_prompt(
            state=state,
            trigger=trigger,
            trigger_history=trigger_history,
            relevant_memories=relevant_memories,
        )

        situational_analysis = llm.generate_complete(
            model, situational_analysis_prompt, caller="situational_analysis"
        )
        trigger_entry.situational_context = situational_analysis

        logger.info(f"Situational Analysis: {situational_analysis}")

        # Create execution context for the full chain
        from .context import ExecutionContext

        context = ExecutionContext(
            trigger=trigger,
            completed_actions=[],
            session_id=str(uuid.uuid4()),
            relevant_memories=relevant_memories,
            situation_analysis=situational_analysis,
        )

        sequence_num = 0
        max_sequences = 3

        while True:
            sequence_num += 1

            if sequence_num > max_sequences:
                logger.debug("Max sequences reached, stopping chain")
                break

            logger.debug(f"Planning sequence {sequence_num}...")

            # Plan next sequence of actions, showing what's already been done
            sequence = self.planner.plan_actions(
                trigger=trigger,
                completed_actions=context.completed_actions,  # Show planner what's already done
                state=state,
                trigger_history=trigger_history,
                llm=llm,
                model=model,
                relevant_memories=relevant_memories,
                situational_analysis=situational_analysis,
            )

            if not sequence.actions:
                logger.debug("No more actions planned, stopping chain")
                break

            # Evaluate sequence for repetitive patterns and correct if needed (if enabled)
            if self.enable_action_evaluation and self.action_evaluator:
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
                    logger.debug(
                        "No repetitive patterns detected, using original sequence"
                    )
            else:
                logger.debug("Action evaluation disabled, using original sequence")

            # Execute the planned sequence, updating the shared context
            logger.debug(
                f"Executing sequence {sequence_num} with {len(sequence.actions)} actions..."
            )
            sequence_results = self.executor.execute_sequence(
                sequence=sequence,
                context=context,  # Pass shared context
                state=state,
                trigger_history=trigger_history,
                llm=llm,
                model=model,
                sequence_number=sequence_num,
                callback=callback,
                trigger_entry=trigger_entry,
            )

            # Results are already added to context.completed_actions by executor

            # Check if agent signaled completion with WAIT action
            wait_actions = [
                r for r in sequence_results if r.action.value == "wait" and r.success
            ]
            if wait_actions:
                logger.debug(
                    f"Agent signaled completion: {wait_actions[0].result_summary}"
                )
                break

        # Notify processing complete
        callback.on_processing_complete(sequence_num, len(context.completed_actions))

        # Notify trigger completion
        successful_count = sum(
            1 for action in context.completed_actions if action.success
        )

        # Add completed actions to the trigger entry and add to trigger history
        trigger_entry.actions_taken = context.completed_actions
        trigger_history.add_trigger_entry(trigger_entry)

        # Compress this trigger entry into a summary (if enabled)
        if individual_trigger_compression:
            _compress_trigger_entry(trigger_entry, state, llm, model)
            _extract_memory_embedding(trigger_entry)

        # Emit completion event after adding to history
        callback.on_trigger_completed(
            entry_id, len(context.completed_actions), successful_count
        )

        logger.debug(
            f"Added trigger entry {entry_id} with {len(context.completed_actions)} actions to trigger history"
        )

        logger.debug(f"=== PROCESSING COMPLETE ===")
        logger.debug(
            f"Executed {len(context.completed_actions)} total actions across {sequence_num} sequences"
        )

        return context.completed_actions


def _compress_trigger_entry(
    trigger_entry: TriggerHistoryEntry,
    state: State,
    llm: LLM,
    model: SupportedModel,
):
    """Compress a completed trigger entry into a concise stream of consciousness entry"""
    from agent.chain_of_action.prompts import (
        format_single_trigger_entry,
        format_section,
    )
    from agent.state import build_agent_state_description

    # Format the full trigger entry for compression
    full_entry_text = format_single_trigger_entry(trigger_entry)
    state_desc = build_agent_state_description(state)

    # Check trigger type for context-aware compression
    from agent.chain_of_action.trigger import WakeupTrigger, UserInputTrigger

    is_wakeup_trigger = isinstance(trigger_entry.trigger, WakeupTrigger)

    if is_wakeup_trigger:
        # Autonomous time compression - preserve intellectual development
        compression_focus = """I need to compress this autonomous experience into a concise but detailed summary that captures both my intellectual development and emotional experience from my perspective. I should maximize the preservation of important details while eliminating redundancy. This should read naturally as part of my internal monologue while preserving essential autonomous development.

The compressed entry MUST capture:
- **Intellectual Development**: Key ideas explored, insights gained, creative frameworks developed, autonomous reflection topics
- **Self-Development**: How I grew or evolved my thinking, what I learned about myself, independent exploration
- **Concrete Ideas**: Specific concepts, frameworks, or thought processes I developed (not just "I thought about X" but what I actually concluded)
- **Autonomous Context**: That this was independent thinking/development, not reactive to interaction
- **Emotional Experience**: My feelings and internal state during this autonomous development

I will write this preserving the intellectual substance of my autonomous thinking while maintaining my natural emotional voice. I will capture both "what I intellectually developed" and "how I felt while developing it" so I can build on these ideas in future autonomous time."""

    else:
        # Interactive time compression - preserve relational context
        compression_focus = """I need to compress this interaction into a concise but detailed summary that captures both the emotional essence and concrete details of what happened from my perspective. I should maximize the preservation of important details while eliminating redundancy. This should read naturally as part of my internal monologue while preserving essential referential information.

The compressed entry MUST capture:
- **Concrete Details**: What was specifically said, which priority topics were relevant (use actual priority content, not IDs), specific actions taken, and factual context
- **Emotional Experience**: My thoughts and feelings in response to what happened  
- **Relational Dynamics**: Important interpersonal moments and emotional shifts
- **Actionable Context**: Details I need to reference in future conversations ("what I just said/did/thought")

I will write this in my natural emotional voice while weaving in specific words spoken, priority content, mood changes, and actions taken. I will include both the "how I felt" and the "what actually happened" so I can reference these details coherently in future interactions."""

    # Build compression prompt following agent conventions
    prompt = f"""I am {state.name}, {state.role}. I need to compress my recent experience into a concise stream of consciousness entry for my memory that preserves both emotional depth and factual details.

{state_desc}

{format_section("MY RECENT EXPERIENCE TO COMPRESS", full_entry_text)}

**MY COMPRESSION TASK:**

{compression_focus}

CRITICAL: I will write ONLY my compressed stream of consciousness entry - no headers, no explanations, no formatting, no analysis sections. Just my natural internal monologue capturing the experience.

**MY COMPRESSED EXPERIENCE:**"""

    try:
        # Calculate original entry size (without any existing compressed summary)
        original_size = len(
            format_single_trigger_entry(trigger_entry, use_summary=False)
        )

        best_summary = None
        best_size = float("inf")

        # Try compression up to 3 times if needed
        max_retries = 3
        compression_threshold = 0.9  # 90% of original size

        for attempt in range(max_retries):
            compressed_summary = llm.generate_complete(
                model, prompt, caller=f"compress_trigger_entry_attempt_{attempt+1}"
            )

            compressed_summary = compressed_summary.strip()
            compressed_size = len(compressed_summary)
            compression_ratio = (
                compressed_size / original_size if original_size > 0 else 0
            )

            logger.debug(
                f"Compression attempt {attempt+1}: {compressed_size} chars ({compression_ratio:.1%} of original {original_size} chars)"
            )

            # Keep track of the best (smallest) summary
            if compressed_size < best_size:
                best_summary = compressed_summary
                best_size = compressed_size

            # If compression is good enough (under 90% of original), use it
            if compression_ratio < compression_threshold:
                logger.debug(
                    f"Compression successful on attempt {attempt+1}: {compression_ratio:.1%} of original"
                )
                trigger_entry.compressed_summary = compressed_summary
                return

            # If this is the last attempt or we got a good result, break
            if attempt == max_retries - 1:
                break

        # Use the best summary we found, even if it's not under the threshold
        if best_summary is not None:
            final_ratio = best_size / original_size if original_size > 0 else 0
            logger.debug(
                f"Using best compression after {max_retries} attempts: {final_ratio:.1%} of original"
            )
            trigger_entry.compressed_summary = best_summary
        else:
            logger.warning("No valid compression generated after retries")

    except Exception as e:
        logger.error(f"Failed to compress trigger entry: {e}")
        # Continue without compression rather than failing


def _extract_memory_embedding(trigger_entry):
    """Extract embedding vector from a trigger entry for memory similarity"""
    from agent.chain_of_action.prompts import format_single_trigger_entry
    from agent.memory.embedding_service import get_embedding_service

    try:
        # Format the full trigger entry for embedding
        full_entry_text = format_single_trigger_entry(trigger_entry)

        # Get embedding service and generate embedding
        embedding_service = get_embedding_service()
        embedding_vector = embedding_service.encode(full_entry_text)

        # Store embedding in trigger entry
        trigger_entry.embedding_vector = embedding_vector

        logger.info(
            f"Generated embedding vector ({len(embedding_vector)} dimensions) for memory"
        )

    except Exception as e:
        logger.warning(f"Failed to extract memory embedding: {e}")
        # Continue without embedding rather than failing
