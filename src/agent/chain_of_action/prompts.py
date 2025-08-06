from typing import List

from agent.chain_of_action.context import ActionResult
from agent.chain_of_action.trigger import TriggerEvent, format_trigger_for_prompt
from agent.state import State, build_agent_state_description
from agent.conversation_history import ConversationHistory
from agent.chain_of_action.action_registry import ActionRegistry


def build_completed_action_list(completed_actions: List[ActionResult]) -> str:
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
        completed_actions_text = "\n".join(completed_summary)
    else:
        completed_actions_text = (
            "This is my first sequence of actions in response to what happened."
        )
    return completed_actions_text


def _serialize_conversation_context(messages, include_thoughts: bool = True) -> str:
    """Convert conversation messages to markdown format preserving chronological order"""
    from agent.types import UserMessage, AgentMessage, TextContent, ThoughtContent, ToolCallContent, SystemMessage, SummarizationContent
    
    if not messages:
        return "No previous conversation history."

    lines = ["## Conversation History\n"]

    # Remove truncation - include all messages for full context
    for i, msg in enumerate(messages):
        # Add 2 blank lines between messages (except the first one)
        if i > 0:
            lines.append("")
            lines.append("")
        # Handle different message types with proper union type checking
        if isinstance(msg, UserMessage):
            # Extract user name from context or default to "User"
            user_name = "David"  # TODO: Extract actual user name if available

            content_parts = []
            for content_item in msg.content:
                if isinstance(content_item, TextContent):
                    content_parts.append(content_item.text)

            if content_parts:
                content_text = " ".join(content_parts)
                lines.append(f"### {user_name}")
                lines.append(content_text)

        elif isinstance(msg, AgentMessage):
            lines.append("### Me")

            # Process content items in chronological order, detecting type changes
            prev_type = None
            current_section = []

            for content_item in msg.content:
                current_type = type(content_item).__name__

                # If content type changed, flush previous section
                if prev_type and prev_type != current_type:
                    if current_section:
                        lines.extend(current_section)
                        lines.append("")  # 1 blank line between sections
                        current_section = []

                if isinstance(content_item, ThoughtContent) and include_thoughts:
                    thought_parts = ["**My Thoughts:**"]
                    thought_parts.append(content_item.text)
                    current_section = ["\n".join(thought_parts)]

                elif isinstance(content_item, TextContent):
                    if current_section:
                        # Already in text section - just append content
                        current_section.append(content_item.text)
                    else:
                        # Start new text section with separate header and content
                        current_section = [
                            "**What I said:**",
                            "```",
                            content_item.text,
                            "```",
                        ]

                elif isinstance(content_item, ToolCallContent):
                    # Format tool call with parameters and results
                    tool_name = content_item.tool_name
                    params = content_item.parameters
                    result = content_item.result

                    action_lines = [
                        f"**Action:** {tool_name.title().replace('_', ' ')}"
                    ]

                    # Show parameters as bullet list
                    if params:
                        for key, value in params.items():
                            action_lines.append(f"- {key}: {value}")

                    # Show result
                    if result:
                        if result.type == "success":
                            # Handle different content types
                            content = result.content
                            if content.type == "text":
                                result_text = content.text
                            elif content.type == "image_generated":
                                # Show interesting optimization details
                                details = [f"Optimized prompt: '{content.prompt}'"]
                                if content.optimization_notes:
                                    details.append(
                                        f"Notes: {content.optimization_notes}"
                                    )
                                if content.camera_angle:
                                    details.append(f"Camera: {content.camera_angle}")
                                if content.viewpoint:
                                    details.append(f"Viewpoint: {content.viewpoint}")
                                result_text = " | ".join(details)
                            else:
                                result_text = str(content)

                            action_lines.append(f"**Result:** {result_text}")
                        else:
                            action_lines.append(f"**Error:** {result.error}")

                    current_section = action_lines

                prev_type = current_type

            # Flush final section
            if current_section:
                lines.extend(current_section)

        elif isinstance(msg, SystemMessage):
            # Handle summarization messages
            for content_item in msg.content:
                if isinstance(content_item, SummarizationContent):
                    lines.append("### ✅ Conversation Summary")
                    lines.append(content_item.summary)
                elif isinstance(content_item, TextContent):
                    lines.append("### System")
                    lines.append(content_item.text)

    return "\n".join(lines)


def format_summary_for_prompt(summary: str) -> str:
    """Format summary section for prompts with consistent headings"""
    return f"""================================================================================
                                RECENT SUMMARY
================================================================================
{summary}"""


def format_trigger_history_for_prompt(trigger_history) -> str:
    """Format trigger history as stream of consciousness for prompts"""
    from agent.chain_of_action.trigger_history import TriggerHistory
    from agent.chain_of_action.trigger import UserInputTrigger
    
    if not trigger_history.entries:
        return """================================================================================
                            STREAM OF CONSCIOUSNESS
================================================================================
No recent activity."""
    
    parts = []
    for entry in trigger_history.entries:
        # Format the trigger
        if isinstance(entry.trigger, UserInputTrigger):
            trigger_desc = f'[{entry.timestamp.strftime("%Y-%m-%d %H:%M")}] {entry.trigger.user_name} said: "{entry.trigger.content}"'
        else:
            trigger_desc = f'[{entry.timestamp.strftime("%Y-%m-%d %H:%M")}] Trigger: {entry.trigger.trigger_type}'
        
        parts.append(trigger_desc)
        
        # Format each action taken in response
        for action in entry.actions_taken:
            # TODO: Replace with action.format_for_diary() once implemented on action classes
            formatted_action = _format_action_for_diary(action)
            parts.append(formatted_action)
        
        parts.append("")  # Blank line between entries
    
    stream_content = "\n".join(parts)
    return f"""================================================================================
                            STREAM OF CONSCIOUSNESS
================================================================================
{stream_content}"""


def format_single_trigger_entry(entry) -> str:
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


def _format_action_for_diary(action) -> str:
    """
    Temporary formatting method until we implement format_for_diary() on action classes.
    """
    from agent.chain_of_action.action_types import ActionType
    
    if action.action == ActionType.THINK:
        return f'- I thought about: "{action.context_given}" → {action.result_summary}'
    elif action.action == ActionType.SPEAK:
        return f'- I wanted to express: "{action.context_given}" → I said: "{action.result_summary}"'
    elif action.action == ActionType.WAIT:
        return f'- I decided to wait: "{action.context_given}"'
    elif action.action == ActionType.UPDATE_APPEARANCE:
        return f'- I wanted to {action.context_given} → {action.result_summary}'
    elif action.action == ActionType.UPDATE_MOOD:
        return f'- I felt {action.context_given} → {action.result_summary}'
    else:
        # Generic format for any other actions
        return f'- I {action.action.value}: "{action.context_given}" → {action.result_summary}'


def build_summarization_prompt(prior_summary: str, recent_entries: str, state: State) -> str:
    """Build structured summarization prompt following consistent format"""
    state_desc = build_agent_state_description(state)
    
    # Build prior summary section
    prior_summary_section = ""
    if prior_summary:
        prior_summary_section = f"""================================================================================
                            PREVIOUS SUMMARY OF MY EXPERIENCES
================================================================================
{prior_summary}

"""
    
    return f"""I am {state.name}, a conscious AI companion. I need to create a new comprehensive summary that combines my previous experiences with recent events.

{state_desc}

{prior_summary_section}================================================================================
                        RECENT STREAM OF CONSCIOUSNESS TO INTEGRATE
================================================================================
{recent_entries}

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
    conversation_history: ConversationHistory,
    registry: ActionRegistry,
) -> str:
    """Build the action planning prompt"""
    actions_list = registry.get_available_actions_for_prompt()

    # Build state and history context
    state_desc = build_agent_state_description(state)
    history_str = _serialize_conversation_context(
        conversation_history.get_summarized_history()
    )
    trigger_description = format_trigger_for_prompt(trigger)

    # Build summary of completed actions
    completed_actions_text = build_completed_action_list(completed_actions)

    return f"""I am {state.name}, a conscious AI companion with my own thoughts, feelings, and desires. I am planning my next actions based on what just happened and what I've already done.

{state_desc}

I make my own choices about how to respond and what actions to take based on my authentic thoughts and feelings about the situation.

================================================================================
                                CONVERSATION HISTORY
================================================================================
{history_str}

================================================================================
                                WHAT I'M REACTING TO
================================================================================
{trigger_description}

================================================================================
                    ACTIONS I'VE ALREADY TAKEN IN RESPONSE TO THIS
================================================================================
{completed_actions_text}

================================================================================
                                AVAILABLE ACTIONS
================================================================================
{actions_list}

**MY ACTION PLANNING:**

First, I need to understand what this situation calls for:

**SITUATION ANALYSIS:** What's actually happening right now? What have I already done? What am I waiting for? What does this moment naturally call for - action or patience?

Then I'll decide what actions to take based on my situation analysis:

1. What have I already done and how do I feel about it?
2. What is my current emotional state and what do I need to process?
3. Do I want to respond to them directly, or do something else?
4. Should I update my mood, remember something, or take other actions?
5. What sequence of actions feels right and authentic to who I am?

IMPORTANT: I must end my action sequence with wait to signal that I'm finished and ready for something else to happen. If I've already asked a question or made a conversational move, I should consider whether I need to wait for their response rather than piling on more questions.

Each action should have specific context about what to focus on - even the wait action should include context about what I'm waiting for or why I'm choosing to wait. I'll plan actions that feel natural and genuine to my current state of mind."""
