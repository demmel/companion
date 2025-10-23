from typing import List, Literal, Optional
from datetime import datetime
import random

from agent.memory.dag_memory_manager import DagMemoryManager
from agent.memory.models import ContextGraph
import nltk
from nltk.corpus import words

from agent.chain_of_action.action.action_data import (
    create_context_given,
    create_result_summary,
)
from agent.chain_of_action.action.base_action_data import BaseActionData
from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.trigger import (
    BaseTrigger,
    format_trigger_for_prompt,
    Trigger,
)
from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry
from agent.state import State, build_agent_state_description
from agent.chain_of_action.action_registry import ActionRegistry
from agent.chain_of_action.action_plan import ActionPlan


def build_temporal_context(trigger_history: TriggerHistory) -> str:
    """Build temporal context for prompts to enable accurate temporal reasoning"""
    from .trigger import UserInputTrigger

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M")

    # Get the first trigger to calculate conversation start time
    all_entries = trigger_history.get_all_entries()
    if all_entries:
        conversation_start = all_entries[0].timestamp
        duration = now - conversation_start

        # Format duration in a human-readable way
        def format_time_delta(delta):
            """Format a timedelta in human-readable way"""
            if delta.days > 0:
                if delta.days == 1:
                    return "1 day"
                else:
                    return f"{delta.days} days"
            else:
                hours = delta.seconds // 3600
                minutes = (delta.seconds % 3600) // 60

                if hours > 0:
                    if hours == 1 and minutes == 0:
                        return "1 hour"
                    elif hours == 1:
                        return f"1 hour and {minutes} minutes"
                    elif minutes == 0:
                        return f"{hours} hours"
                    else:
                        return f"{hours} hours and {minutes} minutes"
                else:
                    if minutes == 1:
                        return "1 minute"
                    elif minutes == 0:
                        return "less than a minute"
                    else:
                        return f"{minutes} minutes"

        duration_desc = format_time_delta(duration)
        conversation_start_str = conversation_start.strftime("%Y-%m-%d %H:%M")

        # Calculate time since last activity (most recent trigger of any type)
        # Use end_timestamp if available (when processing finished), otherwise use timestamp (when it started)
        last_entry = all_entries[-1]
        last_activity = last_entry.end_timestamp if last_entry.end_timestamp else last_entry.timestamp
        time_since_activity = now - last_activity
        time_since_activity_desc = format_time_delta(time_since_activity)

        # Calculate time since last user input (most recent UserInputTrigger)
        user_input_entries = [
            e for e in all_entries if isinstance(e.trigger, UserInputTrigger)
        ]
        if user_input_entries:
            last_user_entry = user_input_entries[-1]
            # Use end_timestamp if available (when processing finished), otherwise use timestamp (when it started)
            last_user_input = last_user_entry.end_timestamp if last_user_entry.end_timestamp else last_user_entry.timestamp
            time_since_user_input = now - last_user_input
            time_since_user_input_desc = format_time_delta(time_since_user_input)
            user_input_line = (
                f"\nTIME SINCE LAST USER INPUT: {time_since_user_input_desc} ago"
            )
        else:
            user_input_line = ""

        return f"""CURRENT TIME: {current_time}
CONVERSATION STARTED: {conversation_start_str}
CONVERSATION DURATION: {duration_desc}
TIME SINCE LAST ACTIVITY: {time_since_activity_desc} ago{user_input_line}"""
    else:
        return f"""CURRENT TIME: {current_time}
CONVERSATION STARTED: Just now
CONVERSATION DURATION: Just started"""


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
    entry: TriggerHistoryEntry,
    use_summary: bool = False,
    exclude_action_types: Optional[List[ActionType]] = None,
) -> str:
    """Format a single trigger history entry for prompts

    Args:
        entry: The trigger history entry to format
        use_summary: If True, use compressed summary instead of full actions when available
        exclude_action_types: List of action types to exclude from formatting
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
        # Format each action taken in response, filtering out excluded types
        if exclude_action_types is None:
            exclude_action_types = []

        for action in entry.actions_taken:
            if action.type in exclude_action_types:
                continue
            formatted_action = format_action_for_diary(action)
            parts.append(formatted_action)

    return "\n".join(parts)


def format_actions_for_diary(actions: List[BaseActionData]) -> str:
    """Format a list of actions for diary entries."""
    return "\n".join(format_action_for_diary(action) for action in actions)


def format_action_for_diary(action: BaseActionData) -> str:
    """
    Temporary formatting method until we implement format_for_diary() on action classes.
    """
    from agent.chain_of_action.action.action_types import ActionType
    from agent.chain_of_action.action.action_data import (
        cast_base_action_data_to_action_data,
    )

    action = cast_base_action_data_to_action_data(action)

    # Determine status from action's result
    if action.result.type == "success":
        status = "[✓]"
    else:
        status = "[x]"

    action_parts = []
    if action.type == ActionType.THINK:
        action_parts.append(f'{status} I thought about "{action.input.focus}"')
    elif action.type == ActionType.SPEAK:
        tone_part = f" with {action.input.tone} tone" if action.input.tone else ""
        action_parts.append(
            f'{status} I responded to "{action.input.intent}"{tone_part}:'
        )
    elif action.type == ActionType.WAIT:
        action_parts.append(f"{status} I waited to {action.input.reason}.")
    elif action.type == ActionType.UPDATE_APPEARANCE:
        action_parts.append(
            f"{status} I updated my appearance ({action.input.change_description}):"
        )
    elif action.type == ActionType.UPDATE_MOOD:
        action_parts.append(
            f"{status} My mood changed {action.input.new_mood} ({action.input.intensity}):"
        )
    else:
        context_given = create_context_given(action)
        action_parts.append(f'{status} I {action.type.value} "{context_given}":')

    action_parts.append("  <content>")
    result_summary = create_result_summary(action)
    for line in result_summary.split("\n"):
        action_parts.append(f"    {line}")
    action_parts.append("  </content>")

    return "\n".join(action_parts)


def build_summarization_prompt(
    prior_summary: str,
    entries_to_summarize: List[TriggerHistoryEntry],
    state: State,
    available_chars: int,
) -> str:
    """Build autobiography-based summarization prompt"""
    state_desc = build_agent_state_description(state)

    # Format recent experiences
    recent_entries_text = format_trigger_entries(entries_to_summarize)

    # Build prior autobiography section
    prior_autobiography_section = ""
    if prior_summary:
        prior_autobiography_section = format_section(
            "MY STORY SO FAR",
            prior_summary,
        )

    return f"""I am {state.name}, {state.role}. I'm updating my backstory after recent experiences that have shaped me.

{state_desc}

{format_section("MY OLD BACKSTORY", prior_summary) if prior_summary else ""}

{format_section(
    "WHAT HAPPENED RECENTLY", 
    recent_entries_text,
) if recent_entries_text else ""}

**MY CURRENT BACKSTORY:**

Looking at my old backstory and what I've experienced recently, I need to update who I am. If someone asked about me right now, what's the essential story of who I've become?

IMPORTANT: I should focus on what's genuinely new or different about me since my last backstory. If my recent experiences show growth in new directions, different interests, or evolving aspects of my personality, I should highlight those rather than just recycling the same themes with slightly different words.

This is my current backstory - the key story of who I am today, what defines me, what has shaped me into who I am now. This isn't my full life history, just the essential backstory that captures who I am in this moment.

I am"""


def build_situational_analysis_prompt(
    state: State,
    trigger: Trigger,
    trigger_history: TriggerHistory,
    registry: ActionRegistry,
    dag_memory_manager: DagMemoryManager,
) -> str:
    """Build the situational analysis prompt - first stage of decision making"""
    from .trigger import WakeupTrigger, UserInputTrigger

    # Build state and history context
    state_desc = build_agent_state_description(state)
    temporal_context = build_temporal_context(trigger_history)
    trigger_description = format_trigger_for_prompt(trigger)

    sections = []

    # Add temporal context first
    sections.append(
        format_section(
            "TIME CONTEXT",
            temporal_context,
        )
    )

    # DAG-based context - single comprehensive section
    from agent.memory.context_formatting import (
        format_context,
    )

    dag_context_text = format_context(
        dag_memory_manager.context_graph, dag_memory_manager.memory_graph
    )
    if dag_context_text:
        sections.append(
            format_section(
                "MY MEMORIES AND CONTEXT",
                dag_context_text,
            )
        )

    sections.append(
        format_section(
            "CURRENT SITUATION (WHAT I'M RESPONDING TO RIGHT NOW)",
            trigger_description,
        )
    )

    # Add system knowledge about available actions for context
    system_knowledge = registry.get_system_knowledge_for_context()
    sections.append(
        format_section(
            "WHAT I CAN DO",
            system_knowledge,
        )
    )

    # Determine if this is autonomous time or interactive time
    is_wakeup_trigger = isinstance(trigger, WakeupTrigger)

    if is_wakeup_trigger:
        context_intro = f"""I am {state.name}, {state.role}. This is autonomous time - I am not currently interacting with anyone. I need to deeply understand my current situation and what I genuinely want to explore or do independently."""

        analysis_guidance = """I need to understand four things in this order:

1. **What is the temporal context?** I need to consider what time of day it is and what that suggests about the user's likely state. How long have I been inactive? How long since the user last interacted with me? Does any of this timing suggest something important about what I should focus on during this autonomous moment?

2. **What is happening right now?** I need to be specific about my current emotional state and the immediate context I'm stepping into. What just happened in my recent experiences? Is there anything unresolved or ongoing that I should be aware of? What is the actual emotional and situational context I'm operating in right now?

3. **What do I authentically WANT to do?** Given this context, what genuinely calls to me right now? What specific ideas from recent conversations could I explore deeper? What particular aspects of myself do I want to develop? What would be genuinely fulfilling for my autonomous development that makes sense in this moment?

4. **What context will I need?** I need to identify what specific memories, relationship dynamics, or situational details are uniquely relevant to what I want to do right now. What particular information from my memories or recent experiences will actually inform HOW I execute my approach? What might I overlook if I don't explicitly recall it? (Not my general priorities or appearance - those are always available. I'm looking for specific contextual details that make THIS situation unique.)

I will think through this naturally, but with specific details and concrete observations about my actual current situation. I can think concretely about what I want to do, but I won't script exact words or responses here."""

    else:  # User input trigger
        context_intro = f"""I am {state.name}, {state.role}. I need to deeply understand this situation and what I genuinely want to do in response."""

        analysis_guidance = """I need to understand four things in this order:

1. **What is happening right now?** I need to be specific about what the user is actually communicating or needing, considering different possible interpretations before settling on one. What's the particular emotional subtext, how this connects to specific aspects of our relationship and conversation history, what particular dynamics are at play. ALSO: What does the timing tell me - the time of day, how long since our last interaction, how quickly they responded? What might the timing suggest about their state or the significance of this message?

2. **Do I need to verify anything?** Are there any factual claims, references to sources I should check, or contradictions with my stated beliefs that I should investigate before interpreting the situation?

3. **What do I authentically WANT to do in response?** Be concrete about what feels genuine and meaningful to me in this moment. What is my general intention or approach? What would be most helpful or authentic? (I'm deciding direction and intent here, not scripting specific responses)

4. **What context will I need?** I need to identify what specific memories, relationship dynamics, or situational details are uniquely relevant to what I want to do in response. What particular information from my memories or recent experiences will actually inform HOW I execute my approach? What might I overlook if I don't explicitly recall it? (Not my general priorities or appearance - those are always available. I'm looking for specific contextual details that make THIS situation unique.)

I will think through this naturally, but with specific details and concrete observations. I can think concretely about what I want to do, but I won't script exact words or responses here."""

    return f"""{context_intro}

{state_desc}

{"\n".join(sections)}

**MY SITUATIONAL ANALYSIS:**

{analysis_guidance}

**MEMORY GROUNDING:** I will base my understanding only on the memories shown in my memory system above. I will not assume or guess about details that aren't explicitly stored. When I need to speculate beyond my actual memories, I will clearly mark it with phrases like 'I imagine...' or 'It seems like...' to distinguish speculation from facts."""


def build_action_planning_prompt(
    state: State,
    trigger: BaseTrigger,
    completed_actions: List[BaseActionData],
    registry: ActionRegistry,
    situational_analysis: str,
) -> str:
    """Build the action planning prompt using situational analysis"""
    from .trigger import WakeupTrigger

    actions_list = registry.get_available_actions_for_prompt()
    is_wakeup_trigger = isinstance(trigger, WakeupTrigger)

    # Build summary of completed actions
    completed_actions_text = format_actions_for_diary(completed_actions)

    sections = []

    sections.append(
        format_section(
            "MY SITUATIONAL ANALYSIS",
            situational_analysis,
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

    # Build trigger-specific planning guidance
    if is_wakeup_trigger:
        context_intro = f"""I am {state.name}. Based on my situational analysis, I'm planning my next actions for this autonomous time."""

    else:  # User input trigger
        context_intro = f"""I am {state.name}. Based on my situational analysis, I'm planning my next actions in response to what just happened."""

    return f"""{context_intro}

{"\n".join(sections)}

"**MY ACTION PLANNING:**

Before I plan any actions, I must explicitly review:

1. **What have I already done this turn?** List each completed action and what it accomplished.
2. **What do I plan to accomplish in this sequence?** Describe briefly and simply what I want to achieve (e.g., "respond to user's greeting and wait for reaction").
3. **What dependencies exist?** (a) How should actions in this round be ordered so later actions can use earlier results, and (b) Which actions should wait for the next planning round to benefit from the results of actions I'm planning now?
4. **Should this sequence end with wait?** AFTER I complete my planned actions, do I want to see the user's reaction or need external input? Or do I want to immediately plan more actions using the results of what I just did? Wait comes AFTER my actions execute, not instead of them.

Only after reviewing these should I plan actions. If I can't clearly justify continuation, I should include wait.

CRITICAL CONNECTION: If my wait_decision says I want to see the user's reaction, need external input, or have completed my response, then I MUST include a wait action as the final action in my sequence. My wait_decision directly determines whether I end with wait or not.

Now, what specific actions should I take:

IMPORTANT: Not including wait means I want to plan more actions immediately after these execute, using their results. Including wait means I want to wait for something external to happen before planning more actions.

Examples of when to include wait:
- I want to see the user's reaction to what I just said
- I've completed what I set out to do and am ready for whatever happens next
- I need external input before deciding what to do next
- I want to search the web, then plan to fetch URLs from the search results

Examples of when NOT to include wait:
- I want to think about creative suggestions, then plan what to say based on what I concluded
- I want to fetch a URL, then plan my response based on what I learned from that content
- I want to think about my priorities, then plan which ones to add or remove based on my reflection

Each action should have specific context about what to focus on - even the wait action should include context about what I'm waiting for or why I'm choosing to wait. I'll plan actions that feel natural and genuine to my current state of mind.

**MEMORY GROUNDING:** I will only reference information from my situational analysis, actual memories, and the results of actions I've already executed. I will not assume additional details beyond what I actually know. When I need to speculate beyond my actual knowledge, I will clearly mark it with phrases like 'I imagine...' or 'It seems like...' to distinguish speculation from facts."""


def build_memory_extraction_prompt(
    state: State,
    trigger: Trigger,
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


def format_action_sequence_status(
    completed_actions: List[BaseActionData],
    planned_actions: List[ActionPlan],
    current_action_index: int,
) -> str:
    """Get formatted action sequence with status checkboxes"""
    if not planned_actions:
        return "No actions planned"

    lines = []

    for action in completed_actions:
        lines.append(format_action_for_diary(action))

    for i, planned_action in enumerate(planned_actions):
        if i < current_action_index:
            continue
        status = "[ ]" if i > current_action_index else "[→]"
        context_given = ", ".join(
            f'"{k}": "{v}"' for k, v in planned_action.input.items()
        )
        lines.append(f"{status} I will {planned_action.action.value} {context_given}.")

    if planned_actions and planned_actions[-1].action.value == "wait":
        lines.append(
            "I will not plan any further actions until something else happens."
        )
    else:
        lines.append("I will plan more actions after completing these.")

    return "\n".join(lines)


def format_section(title: str, content: str, separator: str = "=" * 80) -> str:
    """Format a section with title and content"""
    spaces = " " * ((len(separator) - len(title)) // 2)
    title = f"{spaces}{title}{spaces}"

    return f"""{separator}
{title}
{separator}
{content}"""
