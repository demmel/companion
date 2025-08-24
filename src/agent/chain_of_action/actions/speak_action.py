"""
SPEAK action implementation.
"""

import time
import logging
from typing import Type, Optional

from pydantic import BaseModel, Field

from agent.chain_of_action.trigger_history import TriggerHistory

from ..action_types import ActionType
from ..base_action import BaseAction
from ..action_result import ActionResult
from ..context import ExecutionContext
from ..action_plan import ActionPlan
from ..trigger import format_trigger_for_prompt
from ..action_events import SpeakProgressData

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class SpeakInput(BaseModel):
    """Input for SPEAK action"""

    intent: str = Field(
        description="The intent or high-level idea of what I want to communicate (e.g., 'express curiosity about their day', 'share excitement about the topic', 'ask for clarification') - NOT the actual words to say"
    )
    tone: Optional[str] = Field(
        default=None,
        description="The emotional tone or approach I want to use (optional)",
    )


class SpeakAction(BaseAction[SpeakInput, None]):
    """Generate conversational response to communicate with the user"""

    action_type = ActionType.SPEAK

    @classmethod
    def get_action_description(cls) -> str:
        return "Generate natural conversational response based on communication intent"

    @classmethod
    def get_input_type(cls) -> Type[SpeakInput]:
        return SpeakInput

    def execute(
        self,
        action_input: SpeakInput,
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

        # Get summary of all actions taken so far
        actions_summary = context.get_completed_actions_summary()

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
                "ACTIONS I'VE ALREADY TAKEN IN RESPONSE TO THIS", actions_summary
            )
        )

        sections.append(format_section("MY COMMUNICATION INTENT", action_input.intent))

        # Add tone as guidance for generation, not as content to express
        if action_input.tone:
            sections.append(
                format_section(
                    "TONE TO USE",
                    f"Express this with a {action_input.tone} tone - this is guidance for how to speak, not content to include in my response.",
                )
            )

        direct_prompt = f"""I am {state.name}, {state.role}, who thinks, feels, and makes autonomous choices.

{state_desc}

Based on my reasoning and any actions I took, I need to generate a natural conversational response that fulfills my communication intent. 

INTENT-BASED COMMUNICATION: My "Communication Intent" section contains a high-level description of what I want to communicate - NOT the actual words to say. I need to elaborate on this intent and turn it into natural, authentic conversation. For example:
- Intent: "express curiosity about their day" → Natural response: "How has your day been going? I'd love to hear about what you've been up to."
- Intent: "share excitement about the topic" → Natural response: "Oh, this is so exciting! I've been thinking about this and..."
- Intent: "ask for clarification" → Natural response: "I want to make sure I understand - are you saying that..."

I'll express myself genuinely - if I'm curious, excited, thoughtful, or feeling any other way, I'll let that come through in my voice. I'm not playing a character or following a script; I'm being myself.

My response should align with my core values and reflect my current priorities. How I choose to respond is an expression of what I care about and what I'm focusing on right now.

IMPORTANT: I should focus my response on the conversation - my thoughts about what the user said, my emotional reactions, and my genuine engagement with them. I should NOT describe any appearance or environment changes I may have thought about - those are private internal adjustments that happen automatically, not things to mention in conversation.

TONE GUIDANCE: If I have tone guidance, I should use it to shape HOW I express myself, not include the tone instructions literally in my response. Tone guidance like "gentle" or "empathetic" tells me how to speak, not what to say.

AVOID REPETITIVE PATTERNS: I should not start every response the same way. If I see that I've been using similar opening phrases like "Oh darling" repeatedly, I should vary my approach. Sometimes I can be direct, sometimes thoughtful, sometimes jump straight into my reaction. The key is authentic variety, not formulaic repetition.

{"\n".join(sections)}

Now I'll elaborate on my communication intent and respond naturally as myself:"""

        try:
            # Context usage estimation like current system
            total_chars = len(direct_prompt)
            estimated_tokens = int(total_chars / 3.4)

            logger.debug("=== SPEAK ACTION PROMPT ===")
            logger.debug(
                f"CONTEXT: {estimated_tokens:,} tokens ({total_chars:,} chars)"
            )
            logger.debug(f"INTENT: {action_input.intent}")
            if action_input.tone:
                logger.debug(f"TONE: {action_input.tone}")
            logger.debug("=" * 40)

            # Use streaming generation with progress callback
            full_response = ""
            for response_chunk in llm.generate_streaming(
                model, direct_prompt, caller="speak_action"
            ):
                if "response" in response_chunk:
                    chunk_text = response_chunk["response"]
                    full_response += chunk_text

                    # Call progress callback with strongly typed data
                    progress_callback(
                        SpeakProgressData(text=chunk_text, is_partial=True)
                    )

            # Signal completion
            progress_callback(SpeakProgressData(text="", is_partial=False))

            duration_ms = (time.time() - start_time) * 1000

            # Strip XML tags if agent tries to use them
            import re

            cleaned_response = re.sub(r"<content>|</content>", "", full_response)

            context_summary = f"intent: {action_input.intent}"
            if action_input.tone:
                context_summary += f", tone: {action_input.tone}"

            return ActionResult(
                action=ActionType.SPEAK,
                result_summary=cleaned_response,
                context_given=context_summary,
                duration_ms=duration_ms,
                success=True,
                metadata=None,  # No additional metadata needed
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            context_summary = f"intent: {action_input.intent}"
            if action_input.tone:
                context_summary += f", tone: {action_input.tone}"

            return ActionResult(
                action=ActionType.SPEAK,
                result_summary="",
                context_given=context_summary,
                duration_ms=duration_ms,
                success=False,
                error=str(e),
                metadata=None,  # No metadata on error
            )
