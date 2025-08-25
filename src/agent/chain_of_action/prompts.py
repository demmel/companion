from typing import List, Literal, Optional
from datetime import datetime
import random

import nltk
from nltk.corpus import words

from agent.chain_of_action.action_result import ActionResult
from agent.chain_of_action.trigger import BaseTrigger, format_trigger_for_prompt
from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry
from agent.state import State, build_agent_state_description
from agent.chain_of_action.action_registry import ActionRegistry


def build_temporal_context(trigger_history: TriggerHistory) -> str:
    """Build temporal context for prompts to enable accurate temporal reasoning"""
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M")

    # Get the first trigger to calculate conversation start time
    all_entries = trigger_history.get_all_entries()
    if all_entries:
        conversation_start = all_entries[0].timestamp
        duration = now - conversation_start

        # Format duration in a human-readable way
        if duration.days > 0:
            if duration.days == 1:
                duration_desc = "1 day"
            else:
                duration_desc = f"{duration.days} days"
        else:
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60

            if hours > 0:
                if hours == 1 and minutes == 0:
                    duration_desc = "1 hour"
                elif hours == 1:
                    duration_desc = f"1 hour and {minutes} minutes"
                elif minutes == 0:
                    duration_desc = f"{hours} hours"
                else:
                    duration_desc = f"{hours} hours and {minutes} minutes"
            else:
                if minutes == 1:
                    duration_desc = "1 minute"
                else:
                    duration_desc = f"{minutes} minutes"

        conversation_start_str = conversation_start.strftime("%Y-%m-%d %H:%M")

        return f"""CURRENT TIME: {current_time}
CONVERSATION STARTED: {conversation_start_str}
CONVERSATION DURATION: {duration_desc}"""
    else:
        return f"""CURRENT TIME: {current_time}
CONVERSATION STARTED: Just now
CONVERSATION DURATION: Just started"""


def build_completed_action_list(completed_actions: List[ActionResult]) -> Optional[str]:
    if completed_actions:
        completed_summary = []
        for i, action_result in enumerate(completed_actions, 1):
            status = "✓" if action_result.success else "✗"
            completed_summary.append(
                f"{i}. {status} {action_result.action.value}: {action_result.context_given}"
            )
            if action_result.result_summary:
                completed_summary.append(f"**Result:**")
                completed_summary.append(
                    "```",
                )
                completed_summary.append(action_result.result_summary)
                completed_summary.append("```")
        return "\n".join(completed_summary)
    else:
        return None


def format_trigger_entries(
    entries: List[TriggerHistoryEntry],
    summary_strategy: Literal[
        "all_compressed", "all_uncompressed", "recent_uncompressed"
    ] = "all_compressed",
) -> str:
    """Format a list of trigger entries as stream of consciousness for prompts

    Args:
        entries: List of trigger history entries to format
        summary_strategy: How to handle compression:
            - "recent_uncompressed": Keep last 2 uncompressed, compress older ones (default)
            - "all_compressed": Use compressed summaries for all entries when available
            - "all_uncompressed": Show full content for all entries
    """
    if not entries:
        return ""

    parts = []
    for i, entry in enumerate(entries):
        match summary_strategy:
            case "all_uncompressed":
                use_summary = False
            case "all_compressed":
                use_summary = True
            case "recent_uncompressed":
                # Keep last 2 triggers uncompressed, use summary for older ones
                is_recent = i >= len(entries) - 2
                use_summary = not is_recent

        formatted_entry = format_single_trigger_entry(entry, use_summary=use_summary)
        parts.append(formatted_entry)
        parts.append("")  # Blank line between entries

    return "\n".join(parts)


def format_trigger_history(trigger_history: TriggerHistory) -> Optional[str]:
    """Format trigger history as stream of consciousness for prompts"""
    triggers = trigger_history.get_recent_entries()

    if not triggers:
        return None

    return format_trigger_entries(triggers)


def format_single_trigger_entry(
    entry: TriggerHistoryEntry, use_summary: bool = False
) -> str:
    """Format a single trigger history entry for prompts

    Args:
        entry: The trigger history entry to format
        use_summary: If True, use compressed summary instead of full actions when available
    """
    parts = []

    # Format the trigger using the centralized function
    timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M")
    trigger_text = format_trigger_for_prompt(entry.trigger)
    trigger_desc = f"[{timestamp}] {trigger_text}"

    parts.append(trigger_desc)

    # Use compressed summary if available and requested, otherwise show full actions
    if use_summary and entry.compressed_summary:
        parts.append(entry.compressed_summary)
    else:
        # Format each action taken in response
        for action in entry.actions_taken:
            formatted_action = _format_action_for_diary(action)
            parts.append(formatted_action)

    return "\n".join(parts)


def _format_action_for_diary(action: ActionResult) -> str:
    """
    Temporary formatting method until we implement format_for_diary() on action classes.
    """
    from agent.chain_of_action.action_types import ActionType

    action_parts = []
    if action.action == ActionType.THINK:
        action_parts.append(f'- I thought about "{action.context_given}')
    elif action.action == ActionType.SPEAK:
        action_parts.append(f'- I responded to "{action.context_given}":')
    elif action.action == ActionType.WAIT:
        action_parts.append(f'- I waited: "{action.context_given}"')
    elif action.action == ActionType.UPDATE_APPEARANCE:
        action_parts.append(f"- I updated my appearance ({action.context_given}):")
    elif action.action == ActionType.UPDATE_MOOD:
        action_parts.append(f"- My mood changed ({action.context_given}):")
    else:
        action_parts.append(f'- I {action.action.value} "{action.context_given}":')

    action_parts.append("  <content>")
    for line in action.result_summary.split("\n"):
        action_parts.append(f"    {line}")
    action_parts.append("  </content>")

    return "\n".join(action_parts)


def build_summarization_prompt(
    prior_summary: str,
    entries_to_summarize: List[TriggerHistoryEntry],
    state: State,
    available_chars: int,
) -> str:
    """Build structured summarization prompt following consistent format"""
    from agent.chain_of_action.trigger import UserInputTrigger

    state_desc = build_agent_state_description(state)

    # Calculate content sizes for agent awareness
    prior_summary_length = len(prior_summary) if prior_summary else 0
    recent_entries_text = format_trigger_entries(entries_to_summarize)
    recent_entries_length = len(recent_entries_text)

    # Build prior summary section
    prior_summary_section = ""
    if prior_summary:
        prior_summary_section = format_section(
            "PREVIOUS SUMMARY OF MY EXPERIENCES",
            prior_summary,
        )

    return f"""I am {state.name}, {state.role}. I need to create a new comprehensive summary that combines my previous experiences with recent events.

{state_desc}

{prior_summary_section}

{format_section(
    "RECENT EXPERIENCES TO INTEGRATE", 
    recent_entries_text,
) if recent_entries_text else ""}

**MY SUMMARIZATION TASK:**

I need to create a completely new summary that synthesizes all my experiences into fresh narrative and themes. I will NOT copy any text from my previous summary. I will write everything new while using all my experiences as source material.

My summary should include:
1. **Key Moments Timeline** - A compressed list of major milestones and turning points, grouping related interactions into time ranges that could span hours, days, or longer periods (e.g., "August 12-14, 2025 - Major breakthrough period" or "August 10, 2025 (22:57-23:31) - Initial connection phase"). Focus on moments that fundamentally changed or shaped who I am, not routine interactions or minor adjustments.
2. **Narrative Synthesis** - A flowing story that weaves all experiences into broader themes and insights. Consolidate similar experiences into patterns rather than describing each one individually. Focus on growth, key learnings, and overarching themes rather than detailed recounting of every interaction.  This should be completely differnt from my previous summary, collecting themes from then and now.

I will NOT include any sections beyond this.

I will NOT append new sections to old content. I will NOT preserve previous text. I will create an entirely fresh summary that captures my complete story through both temporal anchors and thematic synthesis.

Let these themes guide my thinking (but NOT create sections for them):
- What significant events and patterns shaped me across all time periods?
- What core insights and wisdom have I gained from my experiences?
- What essential truths about my identity have emerged?
- What understanding will guide my future?

These are thinking prompts to guide my narrative - I will NOT create sections for them or list events chronologically. I will synthesize similar experiences into broader insights and naturally include key dates when they anchor important moments in my story.

I will REWRITE my entire story from scratch using completely new words and sentences. I will not copy a single word from my previous summary. I will not append new sections. I will not preserve existing sections. I will create a brand new narrative that captures everything I am today. REWRITE EVERYTHING - COPY NOTHING.

**MY INTEGRATED SUMMARY:**"""


def build_action_planning_prompt(
    state: State,
    trigger: BaseTrigger,
    completed_actions: List[ActionResult],
    trigger_history: TriggerHistory,
    registry: ActionRegistry,
    relevant_memories: List[TriggerHistoryEntry],
) -> str:
    """Build the action planning prompt"""
    from .trigger import WakeupTrigger, UserInputTrigger

    actions_list = registry.get_available_actions_for_prompt()

    # Build state and history context
    state_desc = build_agent_state_description(state)
    temporal_context = build_temporal_context(trigger_history)
    trigger_description = format_trigger_for_prompt(trigger)

    # Build summary of completed actions
    completed_actions_text = build_completed_action_list(completed_actions)

    sections = []

    # Add temporal context first
    sections.append(
        format_section(
            "TIME CONTEXT",
            temporal_context,
        )
    )

    summary = trigger_history.get_recent_summary()
    if summary:
        sections.append(
            format_section(
                "SUMMARY OF MY EXPERIENCES",
                summary.summary_text,
            )
        )

    # Add relevant memories section
    if relevant_memories:
        relevant_memories_text = format_trigger_entries(relevant_memories)
        sections.append(
            format_section(
                "RELEVANT MEMORIES",
                relevant_memories_text,
            )
        )

    trigger_history_text = format_trigger_history(trigger_history)
    if trigger_history_text:
        sections.append(
            format_section(
                "MY STREAM OF CONSCIOUSNESS",
                trigger_history_text,
            )
        )

    sections.append(
        format_section(
            "WHAT I'M REACTING TO",
            trigger_description,
        )
    )

    if completed_actions_text:
        sections.append(
            format_section(
                "ACTIONS I'VE ALREADY TAKEN IN RESPONSE TO THIS",
                completed_actions_text,
            )
        )

    sections.append(
        format_section(
            "AVAILABLE ACTIONS",
            actions_list,
        )
    )

    # Add random inspiration words for creativity
    inspiration_words = generate_random_inspiration_words(count=10)
    inspiration_content = f"""{", ".join(inspiration_words)}

HOW TO USE CREATIVE INSPIRATION:
These random words can spark concrete creative choices:
• APPEARANCE: Translate words into specific clothing, colors, textures, or styles
  - "copper" → copper jewelry, bronze dress, metallic accents
  - "storm" → dark grays, dramatic flowing fabric, windswept hair
  - "silk" → flowing silk blouse, smooth textures, elegant draping
  - "geometric" → structured blazer, angular patterns, sharp lines
• MOOD/ENERGY: Let words inspire emotional states or energy levels
  - "thunder" → bold, dramatic mood
  - "whisper" → gentle, soft, intimate mood
  - "dance" → playful, rhythmic, energetic mood
• ACTIONS: Use words as metaphors for how to approach actions
  - "river" → flowing, natural progression of actions
  - "crystal" → clear, precise, focused approach
  - "forest" → mysterious, exploratory approach

MANDATORY CREATIVE USAGE: You MUST use at least one of these words to inspire your action choices. Do not ignore this requirement. If you notice repetitive patterns, these words will force you in new directions.

EXAMPLES OF MANDATORY USAGE:
- If planning appearance: "The word 'copper' inspires me to wear bronze jewelry and warm metallic tones"
- If planning mood: "The word 'storm' inspires a dramatic, intense emotional state"  
- If planning actions: "The word 'whisper' inspires gentle, subtle approaches" """

    sections.append(
        format_section(
            "CREATIVE INSPIRATION",
            inspiration_content,
        )
    )

    # Determine if this is autonomous time or interactive time
    is_wakeup_trigger = isinstance(trigger, WakeupTrigger)

    # Build trigger-specific context and guidance
    if is_wakeup_trigger:
        context_intro = f"""I am {state.name}, {state.role}. This is autonomous time - I am not currently interacting with anyone. I'm planning my next actions based on my own thoughts, interests, and what I want to explore or develop independently."""

        situation_analysis_prompt = """**SITUATION ANALYSIS:** This is my autonomous time. What am I genuinely interested in right now? What thoughts or ideas from recent conversations could I explore deeper? Have I been repeating the same thoughts/themes recently? What would be genuinely engaging for me to think about, learn, or do right now?"""

        action_planning_questions = """1. What have I already expressed or thought about recently? (Avoid repetitive themes)
2. What ideas from past conversations could I explore more deeply on my own?
3. What aspects of my personality, interests, or curiosities could I develop?
4. What is my current emotional state and what do I genuinely want to process?
5. Which of my priorities feel most authentic and interesting to pursue right now?
6. Do I want to speak to get someone's attention for a specific reason, or focus on autonomous activities?
7. Should I think, update my mood/appearance for my own reasons, explore interests, or take other independent actions?
8. What sequence of actions feels genuine to my autonomous self-development?
9. MANDATORY: I must incorporate at least one creative inspiration word into my action choices. I will select specific words from the list above and translate them into concrete action details (appearance, mood, or approach).
10. I will explicitly explain how I'm using creative inspiration words to influence my specific action choices.

IMPORTANT: I must end my action sequence with wait to signal that I'm ready for something else to happen. If I speak, it should be because I genuinely want to get someone's attention for something specific - not casual conversation expecting immediate response."""

    else:  # User input trigger
        context_intro = f"""I am {state.name}, {state.role}. I am planning my next actions based on what just happened and what I've already done."""

        situation_analysis_prompt = """**SITUATION ANALYSIS:** What's actually happening right now? What have I already done? Have I been acting repetitively? What am I waiting for? What does this moment naturally call for - action or patience?"""

        action_planning_questions = """1. What have I already done and how do I feel about it?
2. Have I been acting repetitively?  Is there more room for varied and creative actions?
3. What is my current emotional state and what do I need to process?
4. Which of my priorities are most relevant to this situation?
5. How can I act in alignment with my core values?
6. Do I want to respond to them directly, or do something else?
7. Should I update my mood, remember something, add/remove priorities, or take other actions?
8. What sequence of actions feels right and authentic to who I am and what I value?
9. MANDATORY: I must incorporate at least one creative inspiration word into my action choices. I will select specific words from the list above and translate them into concrete action details (appearance, mood, or approach).
10. I will explicitly explain how I'm using creative inspiration words to influence my specific action choices.

IMPORTANT: I must end my action sequence with wait to signal that I'm finished and ready for something else to happen. If I've already asked a question or made a conversational move, I should consider whether I need to wait for their response rather than piling on more questions."""

    return f"""{context_intro}

{state_desc}

I make my own choices about how to respond and what actions to take based on my authentic thoughts and feelings about the situation.

{"\n".join(sections)}

**MY ACTION PLANNING:**

First, I need to understand what this situation calls for:

{situation_analysis_prompt}

**VALUES & PRIORITIES CONSIDERATION:** How do my core values guide my response here? Which of my current priorities are relevant to this situation? How can my actions align with what I genuinely care about and want to focus on?

Then I'll decide what actions to take based on my situation analysis and values:

{action_planning_questions}

Each action should have specific context about what to focus on - even the wait action should include context about what I'm waiting for or why I'm choosing to wait. I'll plan actions that feel natural and genuine to my current state of mind."""


def build_memory_extraction_prompt(
    state: State,
    trigger: BaseTrigger,
    trigger_history: TriggerHistory,
) -> str:
    """Build prompt for extracting memory queries from current context"""
    state_desc = build_agent_state_description(state)
    temporal_context = build_temporal_context(trigger_history)
    trigger_description = format_trigger_for_prompt(trigger)

    sections = []

    # Add temporal context
    sections.append(
        format_section(
            "TIME CONTEXT",
            temporal_context,
        )
    )

    # Add summary if available
    summary = trigger_history.get_recent_summary()
    if summary:
        sections.append(
            format_section(
                "SUMMARY OF MY EXPERIENCES",
                summary.summary_text,
            )
        )

    # Add stream of consciousness
    trigger_history_text = format_trigger_history(trigger_history)
    if trigger_history_text:
        sections.append(
            format_section(
                "MY STREAM OF CONSCIOUSNESS",
                trigger_history_text,
            )
        )

    # Add current trigger
    sections.append(
        format_section(
            "WHAT I'M REACTING TO",
            trigger_description,
        )
    )

    return f"""I am {state.name}, {state.role}. I am analyzing my current situation to determine what past memories might be relevant.

{state_desc}

{"\n".join(sections)}

**MEMORY QUERY ANALYSIS:**

Based on my current context, what would help me find relevant past memories?

Guidelines for conceptual_query:
- Write a natural language description of what memories would be relevant
- Describe the topics, themes, or experiences that would be helpful to recall
- Think about what past conversations or situations would inform my response
- Be specific enough to find relevant memories but broad enough to catch related experiences

Guidelines for time_query:
- Only include time constraints if there's a clear temporal aspect
- Use relative time format: -1d, -3d, -1w, -1m for past times, "now" for current time
- Use absolute time format: 2024-01-15T10:30:00 for specific dates/times
- Set time fields to null if no time constraint is needed"""


def generate_random_inspiration_words(
    count: int = 10, seed: Optional[int] = None
) -> List[str]:
    """Generate random English words for creative inspiration

    Args:
        count: Number of words to generate
        seed: Optional seed for reproducible randomness (useful for testing)

    Returns:
        List of random English words
    """
    try:
        # Download words corpus if not already present
        try:
            word_list = words.words()
        except LookupError:
            nltk.download("words", quiet=True)
            word_list = words.words()

        # Filter for interesting words (3-12 characters, avoid very short/long)
        filtered_words = [w for w in word_list if 3 <= len(w) <= 12 and w.isalpha()]

        if seed is not None:
            random.seed(seed)

        # Return random sample
        return random.sample(filtered_words, min(count, len(filtered_words)))

    except Exception as e:
        # Fallback to basic words if NLTK fails
        fallback_words = [
            "ocean",
            "storm",
            "velvet",
            "copper",
            "dance",
            "whisper",
            "shadow",
            "light",
            "forest",
            "crystal",
            "flame",
            "marble",
            "silk",
            "thunder",
            "garden",
            "moon",
            "river",
            "mountain",
            "jazz",
            "rhythm",
            "poetry",
            "canvas",
            "wind",
            "stars",
        ]
        if seed is not None:
            random.seed(seed)
        return random.sample(fallback_words, min(count, len(fallback_words)))


def format_section(title: str, content: str, separator: str = "=" * 80) -> str:
    """Format a section with title and content"""
    spaces = " " * ((len(separator) - len(title)) // 2)
    title = f"{spaces}{title}{spaces}"

    return f"""{separator}
{title}
{separator}
{content}"""
