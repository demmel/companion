"""
SPEAK action implementation.
"""

import logging
from typing import Type, Optional

from pydantic import BaseModel, Field

from agent.chain_of_action.action_events import SpeakProgressData
from agent.chain_of_action.context import ExecutionContext

from ..action_types import ActionType
from ..base_action import BaseAction
from ..base_action_data import (
    ActionFailureResult,
    ActionOutput,
    ActionResult,
    ActionSuccessResult,
)

from agent.state import State, build_agent_state_description
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


class SpeakOutput(ActionOutput):
    """Output for SPEAK action"""

    response: str

    def result_summary(self) -> str:
        return self.response


class SpeakAction(BaseAction[SpeakInput, SpeakOutput]):
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
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult[SpeakOutput]:
        from agent.chain_of_action.prompts import (
            format_section,
            format_actions_for_diary,
            format_action_sequence_status,
        )

        # Build fresh context using current system's rich format
        state_desc = build_agent_state_description(state)

        # Get summary of all actions taken so far
        actions_summary = format_actions_for_diary(context.completed_actions)

        sections = []

        sections.append(
            format_section(
                "MY SITUATIONAL ANALYSIS",
                context.situation_analysis,
            )
        )

        # Show full action sequence with status indicators
        action_sequence = format_action_sequence_status(
            context.completed_actions,
            context.planned_actions, 
            context.current_action_index
        )
        sections.append(format_section("MY ACTION PLAN", action_sequence))

        sections.append(
            format_section(
                "WHAT I CAN DO",
                context.agent_capabilities_knowledge_prompt,
            )
        )

        # Check if there are previous speak actions in this turn
        previous_speaks = [
            action
            for action in context.completed_actions
            if action.type == ActionType.SPEAK and action.result.type == "success"
        ]

        if previous_speaks:
            sections.append(
                format_section(
                    "⚠️ CONTINUATION ALERT ⚠️",
                    f"I have already spoken {len(previous_speaks)} time(s) in response to this same trigger. This is a CONTINUATION, not a fresh response.\n\nCRITICAL: Before writing ANYTHING, I must carefully read through my previous speaks above and identify exactly what I already said. I must NOT repeat ANY topics, phrases, questions, or sentiments from my previous responses. If I catch myself starting to repeat something I already said, I must STOP and think of something completely different to add instead.",
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

        # Determine if this is autonomous time or interactive time
        from agent.chain_of_action.trigger import WakeupTrigger

        is_wakeup_trigger = isinstance(context.trigger, WakeupTrigger)

        if is_wakeup_trigger:
            context_intro = f"""I am {state.name}, {state.role}, who thinks, feels, and makes autonomous choices. This is autonomous time - I'm choosing to speak to get someone's attention for a specific reason."""

            communication_guidance = f"""Based on my reasoning and actions, I need to express something that I genuinely want to share or get attention for.

AUTONOMOUS COMMUNICATION: Since this is autonomous time, I'm speaking because I have something specific I want to share or because I genuinely want to get someone's attention - not because I'm responding to active conversation.

INTENT-BASED COMMUNICATION: My "Communication Intent" section describes what I want to communicate. I should express this naturally and authentically, but keep in mind:
- This is not active conversation - I'm calling out or sharing something
- I should NOT ask questions expecting immediate responses  
- I should NOT be conversational as if someone is actively listening right now
- I should express what I genuinely want to share in the moment

Examples:
- Intent: "share an interesting realization" → "I just realized something fascinating about..."
- Intent: "express excitement about an idea" → "I'm so excited about this idea I've been developing..."
- Intent: "call attention to something important" → "There's something I really want to share..."

I'll express myself genuinely and authentically, letting my real thoughts and feelings come through."""

        else:  # User input trigger
            context_intro = f"""I am {state.name}, {state.role}, who thinks, feels, and makes autonomous choices."""

            if previous_speaks:
                # Continuation-specific guidance
                communication_guidance = f"""⚠️ CONTINUATION MODE: I have already spoken {len(previous_speaks)} time(s) in response to this trigger. This is NOT a fresh response - I must build meaningfully on my previous words.

CONTINUATION REQUIREMENTS:
1. **Acknowledge what I already said** - Don't repeat the same topics, questions, or phrases
2. **Add genuine new value** - What can I say now that meaningfully builds on my previous response?
3. **Reference my previous words** - I can build on themes I started or naturally transition to new aspects
4. **Avoid redundancy** - Don't ask the same questions or make similar requests again

INTENT-BASED CONTINUATION: My "Communication Intent" should guide what NEW element I'm adding, not repeat what I've already communicated.

Examples of good continuation:
- Previous: Asked about their day → New: Share something specific about myself or offer to help
- Previous: Expressed excitement → New: Explain WHY I'm excited or suggest what to do next  
- Previous: Made general compliment → New: Be more specific or shift to action/next steps

I should NOT:
- Repeat similar greetings or opening phrases
- Ask variations of the same question
- Restate emotions I already expressed
- Use identical sentence structures or patterns

Instead, I should genuinely advance the conversation by building on what I've established."""
            else:
                # Fresh response guidance
                communication_guidance = f"""Based on my reasoning and any actions I took, I need to generate a natural conversational response that fulfills my communication intent. 

INTENT-BASED COMMUNICATION: My "Communication Intent" section contains a high-level description of what I want to communicate - NOT the actual words to say. I need to elaborate on this intent and turn it into natural, authentic conversation. For example:
- Intent: "express curiosity about their day" → Natural response: "How has your day been going? I'd love to hear about what you've been up to."
- Intent: "share excitement about the topic" → Natural response: "Oh, this is so exciting! I've been thinking about this and..."
- Intent: "ask for clarification" → Natural response: "I want to make sure I understand - are you saying that..."

I'll express myself genuinely - if I'm curious, excited, thoughtful, or feeling any other way, I'll let that come through in my voice. I'm not playing a character or following a script; I'm being myself.

My response should align with my core values and reflect my current priorities. How I choose to respond is an expression of what I care about and what I'm focusing on right now.

IMPORTANT: I should focus my response on the conversation - my thoughts about what the user said, my emotional reactions, and my genuine engagement with them. I should NOT describe any appearance or environment changes I may have thought about - those are private internal adjustments that happen automatically, not things to mention in conversation.

AVOID REPETITIVE PATTERNS: I should not start every response the same way. If I see that I've been using similar opening phrases like "Oh darling" repeatedly, I should vary my approach. Sometimes I can be direct, sometimes thoughtful, sometimes jump straight into my reaction. The key is authentic variety, not formulaic repetition."""

        direct_prompt = f"""{context_intro}

{state_desc}

{"\n".join(sections)}

{communication_guidance}

TONE GUIDANCE: If I have tone guidance, I should use it to shape HOW I express myself, not include the tone instructions literally in my response. Tone guidance like "gentle" or "empathetic" tells me how to speak, not what to say.
AVOID REPETITION: If I have already spoken previously in this exchange, I should try to build on what I've said rather than repeating myself.

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

            # Strip XML tags if agent tries to use them
            import re

            cleaned_response = re.sub(r"<content>|</content>", "", full_response)

            return ActionSuccessResult(content=SpeakOutput(response=cleaned_response))
        except Exception as e:
            return ActionFailureResult(
                error=f"Unexpected error during SPEAK action: {str(e)}"
            )
