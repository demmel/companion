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
    """Serialize conversation history for prompts"""
    # Import the real serializer
    from agent.reasoning.analyze import (
        _serialize_conversation_context as real_serializer,
    )

    return real_serializer(messages, include_thoughts)


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
