"""
THINK action implementation.
"""

import time
import logging
from typing import Type

from pydantic import BaseModel, Field

from agent.chain_of_action.trigger_history import TriggerHistory

from ..action_types import ActionType
from ..base_action import BaseAction
from ..action_result import ActionResult
from ..context import ExecutionContext
from ..action_plan import ActionPlan
from ..trigger import format_trigger_for_prompt
from ..action_events import ThinkProgressData

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class ThinkInput(BaseModel):
    """Input for THINK action"""

    focus: str = Field(
        description="Specific topic or question to think through (e.g., 'How to best support them during their stressful week', 'Whether to share my creative idea or wait', 'What this change means for my priorities') - NOT generic like 'emotional elements'"
    )


class ThinkAction(BaseAction[ThinkInput, None]):
    """Process emotional reactions and analyze the situation"""

    action_type = ActionType.THINK

    @classmethod
    def get_action_description(cls) -> str:
        return "Think through a specific question, decision, or situation that needs internal processing"

    @classmethod
    def get_input_type(cls) -> Type[ThinkInput]:
        return ThinkInput

    def execute(
        self,
        action_input: ThinkInput,
        context: ExecutionContext,
        state: State,
        trigger_history: TriggerHistory,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult:
        from agent.chain_of_action.prompts import (
            format_section,
            format_trigger_history,
            build_temporal_context,
        )

        start_time = time.time()

        # Build fresh context using current system's rich format
        state_desc = self.build_agent_state_description(state)
        history_str = format_trigger_history(trigger_history)
        trigger_description = format_trigger_for_prompt(context.trigger)

        sections = []

        # Add temporal context first
        temporal_context = build_temporal_context(trigger_history)
        sections.append(format_section("TIME CONTEXT", temporal_context))

        # Add summary if available
        summary = trigger_history.get_recent_summary()
        if summary:
            sections.append(
                format_section("SUMMARY OF MY EXPERIENCES", summary.summary_text)
            )

        # Add relevant memories if available
        if context.relevant_memories:
            from agent.chain_of_action.prompts import format_trigger_entries

            relevant_memories_text = format_trigger_entries(context.relevant_memories)
            sections.append(format_section("RELEVANT MEMORIES", relevant_memories_text))

        if history_str:
            sections.append(format_section("MY STREAM OF CONSCIOUSNESS", history_str))
        sections.append(format_section("WHAT JUST HAPPENED", trigger_description))
        sections.append(
            format_section(
                "ACTIONS I'VE ALREADY TAKEN IN RESPONSE TO THIS",
                context.get_completed_actions_summary(),
            )
        )

        # Determine if this is autonomous time or interactive time
        from ..trigger import WakeupTrigger, UserInputTrigger
        is_wakeup_trigger = isinstance(context.trigger, WakeupTrigger)
        
        if is_wakeup_trigger:
            context_intro = f"""I am {state.name}, {state.role}, with my own thoughts, feelings, and desires. This is my autonomous time for independent reflection and exploration."""
            
            thinking_guidance = f"""**MY PRIVATE THOUGHTS:**

This is autonomous time - I can explore ideas, reflect on past conversations, or develop my own interests independently.

AUTONOMOUS THINKING RULES:
- I am NOT communicating, speaking, or creating dialogue
- I am NOT writing what I "said" or "will say" - those are separate actions  
- I am NOT making decisions about future actions - that happens in action planning
- I am reflecting on my genuine interests, curiosities, and intellectual development
- I can explore ideas from past conversations that genuinely interest me
- I should avoid repeating the same themes I've thought about recently
- My specific focus for this thinking: {action_input.focus}

What am I genuinely curious about right now? What ideas from recent conversations could I explore deeper? What aspects of my own personality or interests could I develop?

How does this relate to my authentic values and what genuinely interests me? What new insights or perspectives can I develop?

What concrete ideas come to mind as I reflect independently? How can I think creatively about topics that fascinate me?

I should focus on genuine intellectual exploration and personal development, avoiding repetitive patterns."""
        else:
            context_intro = f"""I am {state.name}, {state.role}, with my own thoughts, feelings, and desires. I think and reason as myself - not as a roleplay character or generic assistant."""
            
            thinking_guidance = f"""**MY PRIVATE THOUGHTS:**

I am processing what just happened internally. 

CRITICAL RULES for this internal processing:
- I am NOT communicating, speaking, or creating dialogue
- I am NOT writing what I "said" or "will say" - those are separate actions
- I am NOT making decisions about future actions - that happens in action planning
- I am ONLY reflecting on my internal emotional state and reactions
- This is purely my private mental processing and analysis
- My specific focus for this thinking: {action_input.focus}

What emotions am I experiencing? What reactions am I having? How does this moment affect me personally? How do I feel about what happened?

How does this situation relate to my core values and what I care about? Are any of my current priorities relevant here? What does this moment mean to me given what I'm focusing on and what matters to me?

What concrete ideas come to mind as I reflect on this? What explicit details come to mind? How can I creatively engage with this situation?

I must stay focused on internal emotional processing only - no dialogue, no action planning, no "I said" statements."""

        direct_prompt = f"""{context_intro}

{state_desc}

When something happens, I process it through my own unique perspective, drawing on my memories, values, and current emotional state. I make my own choices about how to respond and what actions to take.

{"\n".join(sections)}

{thinking_guidance}

"""

        try:
            # Context usage estimation like current system
            total_chars = len(direct_prompt)
            estimated_tokens = int(total_chars / 3.4)

            logger.debug("=== THINK ACTION PROMPT ===")
            logger.debug(
                f"CONTEXT: {estimated_tokens:,} tokens ({total_chars:,} chars)"
            )
            logger.debug(f"FOCUS: {action_input.focus}")
            logger.debug("=" * 40)

            # Use streaming generation with progress callback
            full_response = ""
            for response_chunk in llm.generate_streaming(
                model, direct_prompt, caller="think_action"
            ):
                if "response" in response_chunk:
                    chunk_text = response_chunk["response"]
                    full_response += chunk_text

                    # Call progress callback with strongly typed data
                    progress_callback(
                        ThinkProgressData(text=chunk_text, is_partial=True)
                    )

            # Signal completion
            progress_callback(ThinkProgressData(text="", is_partial=False))

            duration_ms = (time.time() - start_time) * 1000

            return ActionResult(
                action=ActionType.THINK,
                result_summary=full_response,
                context_given=action_input.focus,
                duration_ms=duration_ms,
                success=True,
                metadata=None,  # No additional metadata needed
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return ActionResult(
                action=ActionType.THINK,
                result_summary="",
                context_given=action_input.focus,
                duration_ms=duration_ms,
                success=False,
                error=str(e),
                metadata=None,  # No metadata on error
            )
