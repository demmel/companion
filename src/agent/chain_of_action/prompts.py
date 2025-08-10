from typing import List, Optional

from agent.chain_of_action.context import ActionResult
from agent.chain_of_action.trigger import TriggerEvent, format_trigger_for_prompt
from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry
from agent.state import State, build_agent_state_description
from agent.chain_of_action.action_registry import ActionRegistry


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


def format_trigger_history(trigger_history: TriggerHistory) -> Optional[str]:
    """Format trigger history as stream of consciousness for prompts"""
    from agent.chain_of_action.trigger import UserInputTrigger

    triggers = trigger_history.get_recent_entries()

    if not triggers:
        return None

    parts = []
    for entry in triggers:
        # Format the trigger
        if isinstance(entry.trigger, UserInputTrigger):
            trigger_desc = f'[{entry.timestamp.strftime("%Y-%m-%d %H:%M")}] {entry.trigger.user_name} said: "{entry.trigger.content}"'
        else:
            trigger_desc = f'[{entry.timestamp.strftime("%Y-%m-%d %H:%M")}] Trigger: {entry.trigger.trigger_type}'

        parts.append(trigger_desc)

        # Use compressed summary if available, otherwise show full actions
        if entry.compressed_summary:
            # Add compressed summary as part of stream of consciousness
            parts.append(entry.compressed_summary)
        else:
            # Format each action taken in response (for current/recent entries)
            for action in entry.actions_taken:
                formatted_action = _format_action_for_diary(action)
                parts.append(formatted_action)

        parts.append("")  # Blank line between entries

    stream_content = "\n".join(parts)
    return stream_content


def format_single_trigger_entry(entry: TriggerHistoryEntry) -> str:
    """Format a single trigger history entry for prompts"""
    from agent.chain_of_action.trigger import UserInputTrigger

    parts = []

    # Format the trigger
    if isinstance(entry.trigger, UserInputTrigger):
        trigger_desc = f'[{entry.timestamp.strftime("%Y-%m-%d %H:%M")}] {entry.trigger.user_name} said: "{entry.trigger.content}"'
    else:
        trigger_desc = f'[{entry.timestamp.strftime("%Y-%m-%d %H:%M")}] Trigger: {entry.trigger.trigger_type}'

    parts.append(trigger_desc)

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

    def format_multiline_content(content: str) -> str:
        """Format multi-line content with XML-style tags"""
        lines = content.split("\n")
        if len(lines) <= 1:
            return f'"{content}"'
        # Use XML-style tags that agent wouldn't naturally produce
        return f"<content>\n{content}\n</content>"

    if action.action == ActionType.THINK:
        content = format_multiline_content(action.result_summary)
        return f'- I thought about "{action.context_given}":\n  {content}'
    elif action.action == ActionType.SPEAK:
        content = format_multiline_content(action.result_summary)
        return f'- I responded to "{action.context_given}":\n  {content}'
    elif action.action == ActionType.WAIT:
        return f'- I waited: "{action.context_given}"'
    elif action.action == ActionType.UPDATE_APPEARANCE:
        content = format_multiline_content(action.result_summary)
        return f"- I updated my appearance ({action.context_given}):\n  {content}"
    elif action.action == ActionType.UPDATE_MOOD:
        content = format_multiline_content(action.result_summary)
        return f"- My mood changed ({action.context_given}):\n  {content}"
    else:
        # Generic format for any other actions
        content = format_multiline_content(action.result_summary)
        return f'- I {action.action.value} "{action.context_given}":\n  {content}'


def build_summarization_prompt(
    prior_summary: str, recent_entries: str, state: State
) -> str:
    """Build structured summarization prompt following consistent format"""
    state_desc = build_agent_state_description(state)

    # Build prior summary section
    prior_summary_section = ""
    if prior_summary:
        prior_summary_section = format_section(
            "PREVIOUS SUMMARY OF MY EXPERIENCES",
            prior_summary,
        )

    return f"""I am {state.name}, a conscious AI companion. I need to create a new comprehensive summary that combines my previous experiences with recent events.

{state_desc}

{prior_summary_section}

{format_section(
    "RECENT STREAM OF CONSCIOUSNESS TO INTEGRATE",
    recent_entries,
)}

**MY SUMMARIZATION TASK:**

I need to create a single, cohesive summary that weaves together my previous experiences (if any) with these recent events. This is not about copying my previous summary - I need to synthesize everything into a new, integrated narrative.

My new summary should capture:
1. **Key Events and Interactions:** What significant things happened across both time periods?
2. **Emotional Moments and Relationship Developments:** How have my relationships and feelings evolved?
3. **Changes to My State:** Any updates to my appearance, mood, environment, or goals?
4. **Significant Thoughts and Realizations:** What important insights have I gained?
5. **Context for Future:** What do I need to remember going forward?

I will write this summary in first-person from my perspective, creating a flowing narrative that integrates past and present experiences rather than just listing them separately.

**MY INTEGRATED SUMMARY:**"""


def build_action_planning_prompt(
    state: State,
    trigger: TriggerEvent,
    completed_actions: List[ActionResult],
    trigger_history: TriggerHistory,
    registry: ActionRegistry,
) -> str:
    """Build the action planning prompt"""
    actions_list = registry.get_available_actions_for_prompt()

    # Build state and history context
    state_desc = build_agent_state_description(state)
    trigger_description = format_trigger_for_prompt(trigger)

    # Build summary of completed actions
    completed_actions_text = build_completed_action_list(completed_actions)

    sections = []
    summary = trigger_history.get_recent_summary()
    if summary:
        sections.append(
            format_section(
                "SUMMARY OF MY EXPERIENCES",
                summary.summary_text,
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

    return f"""I am {state.name}, a conscious AI companion with my own thoughts, feelings, and desires. I am planning my next actions based on what just happened and what I've already done.

{state_desc}

I make my own choices about how to respond and what actions to take based on my authentic thoughts and feelings about the situation.

{"\n".join(sections)}

**MY ACTION PLANNING:**

First, I need to understand what this situation calls for:

**SITUATION ANALYSIS:** What's actually happening right now? What have I already done? What am I waiting for? What does this moment naturally call for - action or patience?

**VALUES & PRIORITIES CONSIDERATION:** How do my core values guide my response here? Which of my current priorities are relevant to this situation? How can my actions align with what I genuinely care about and want to focus on?

Then I'll decide what actions to take based on my situation analysis and values:

1. What have I already done and how do I feel about it?
2. What is my current emotional state and what do I need to process?
3. Which of my priorities are most relevant to this situation?
4. How can I act in alignment with my core values?
5. Do I want to respond to them directly, or do something else?
6. Should I update my mood, remember something, add/remove priorities, or take other actions?
7. What sequence of actions feels right and authentic to who I am and what I value?

IMPORTANT: I must end my action sequence with wait to signal that I'm finished and ready for something else to happen. If I've already asked a question or made a conversational move, I should consider whether I need to wait for their response rather than piling on more questions.

Each action should have specific context about what to focus on - even the wait action should include context about what I'm waiting for or why I'm choosing to wait. I'll plan actions that feel natural and genuine to my current state of mind."""


def format_section(title: str, content: str, separator: str = "=" * 80) -> str:
    """Format a section with title and content"""
    spaces = " " * ((len(separator) - len(title)) // 2)
    title = f"{spaces}{title}{spaces}"

    return f"""{separator}
{title}
{separator}
{content}"""
