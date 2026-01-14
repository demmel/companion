"""
State replay system for deterministic conversation replay.

This module enables replaying conversations without re-executing side effects
(LLM calls, image generation, etc.) by using the recorded action results
to reconstruct state at each point in the conversation.
"""

from typing import Iterator

from agent.state import State, Priority, Value
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.chain_of_action.trigger import BirthTrigger
from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.action.actions.update_mood_action import UpdateMoodAction
from agent.chain_of_action.action.actions.visual_actions import (
    UpdateAppearanceAction,
    UpdateEnvironmentAction,
)
from agent.chain_of_action.action.actions.priority_actions import (
    AddPriorityAction,
    RemovePriorityAction,
)
from agent.chain_of_action.action.actions.evaluate_priorities_action import (
    EvaluatePrioritiesAction,
)
from agent.chain_of_action.action.action_data import ActionData


# Map action types to their action classes for apply_state_change
_ACTION_CLASSES = {
    ActionType.UPDATE_MOOD: UpdateMoodAction(),
    ActionType.UPDATE_APPEARANCE: UpdateAppearanceAction(enable_image_generation=False),
    ActionType.UPDATE_ENVIRONMENT: UpdateEnvironmentAction(enable_image_generation=False),
    ActionType.ADD_PRIORITY: AddPriorityAction(),
    ActionType.REMOVE_PRIORITY: RemovePriorityAction(),
    ActionType.EVALUATE_PRIORITIES: EvaluatePrioritiesAction(),
}


def derive_initial_state(first_entry: TriggerHistoryEntry) -> State:
    """
    Derive the initial state from the first trigger entry.

    First checks if the BirthTrigger has initial_state set directly (new format).
    Falls back to parsing the THINK action output for backward compatibility.

    Args:
        first_entry: The first trigger history entry (should be a birth trigger)

    Returns:
        The initial State for the conversation

    Raises:
        ValueError: If the first trigger is not a BirthTrigger or has no initial state
    """
    if not isinstance(first_entry.trigger, BirthTrigger):
        raise ValueError(
            f"First trigger must be a BirthTrigger, got {type(first_entry.trigger).__name__}"
        )

    # New format: initial_state set directly on BirthTrigger
    if first_entry.trigger.initial_state is not None:
        return first_entry.trigger.initial_state

    # Find the THINK action that contains the derived state
    think_action = None
    for action in first_entry.actions_taken:
        if action.type == ActionType.THINK:
            think_action = action
            break

    if think_action is None or think_action.result.type != "success":
        raise ValueError("Birth trigger must have a successful THINK action with derived state")

    # Parse the state from the thoughts text
    # Format from core.py:
    # Name: {name}
    # Role: {role}
    # Mood: {mood}
    # Environment: {environment}
    # Appearance: {appearance}
    # Backstory: {backstory}
    # Core Values:
    # - {value1}
    # - {value2}
    # Priorities:
    # - {priority1}
    # - {priority2}

    thoughts = think_action.result.content.thoughts
    lines = thoughts.split("\n")

    name = ""
    role = ""
    mood = ""
    environment = ""
    appearance = ""
    core_values = []
    priorities = []

    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Name:"):
            name = line[5:].strip()
        elif line.startswith("Role:"):
            role = line[5:].strip()
        elif line.startswith("Mood:"):
            mood = line[5:].strip()
        elif line.startswith("Environment:"):
            environment = line[12:].strip()
        elif line.startswith("Appearance:"):
            appearance = line[11:].strip()
        elif line.startswith("Backstory:"):
            # Skip backstory, not part of State
            current_section = "backstory"
        elif line == "Core Values:":
            current_section = "values"
        elif line == "Priorities:":
            current_section = "priorities"
        elif line.startswith("- "):
            item = line[2:].strip()
            if current_section == "values":
                core_values.append(Value(content=item))
            elif current_section == "priorities":
                priorities.append(item)

    # Create priorities with sequential IDs
    priority_objects = [
        Priority(id=f"p{i}", content=content)
        for i, content in enumerate(priorities, 1)
    ]

    return State(
        name=name,
        role=role,
        current_mood=mood,
        mood_intensity="medium",  # Default, not stored in text format
        current_appearance=appearance,
        current_environment=environment,
        core_values=core_values,
        current_priorities=priority_objects,
        next_priority_id=len(priority_objects) + 1,
    )


def derive_initial_state_or_default(
    first_entry: TriggerHistoryEntry,
    default_state: State,
) -> State:
    """
    Derive initial state from first entry, or use provided default.

    This is useful for eval scenarios where we want to provide a custom
    initial state rather than relying on the birth trigger.

    Args:
        first_entry: The first trigger history entry
        default_state: State to use if derivation isn't possible

    Returns:
        The initial State for the conversation
    """
    if isinstance(first_entry.trigger, BirthTrigger):
        try:
            return derive_initial_state(first_entry)
        except ValueError:
            return default_state
    return default_state


def apply_action_state_change(state: State, action_data: ActionData) -> None:
    """
    Apply a single action's state changes to the current state.

    Mutates state in place. Uses the action's apply_state_change method
    if it modifies state, otherwise does nothing.

    Args:
        state: State to mutate
        action_data: The action data containing input and result
    """
    # Skip failed actions
    if action_data.result.type != "success":
        return

    action_type = action_data.type
    action_instance = _ACTION_CLASSES.get(action_type)

    if action_instance is None:
        # Action doesn't modify state (THINK, SPEAK, etc.)
        return

    action_instance.apply_state_change(
        state=state,
        action_input=action_data.input,
        output=action_data.result.content,
    )


def replay_state(
    entries: list[TriggerHistoryEntry],
    initial_state: State,
) -> Iterator[tuple[TriggerHistoryEntry, State]]:
    """
    Replay a conversation and yield state at each trigger.

    This is the core replay function that enables deterministic state
    reconstruction from recorded conversation history.

    Args:
        entries: List of trigger history entries in chronological order
        initial_state: The state at the start of the conversation (will be mutated)

    Yields:
        Tuples of (entry, state_at_that_point) for each trigger.
        The state is the state BEFORE processing that entry's actions.
    """
    state = initial_state

    for entry in entries:
        # Yield state before processing this entry's actions
        yield (entry, state)

        # Apply all actions from this entry to get state for next entry
        for action_data in entry.actions_taken:
            apply_action_state_change(state, action_data)


def replay_state_from_birth(
    entries: list[TriggerHistoryEntry],
) -> Iterator[tuple[TriggerHistoryEntry, State]]:
    """
    Replay a conversation starting from a birth trigger.

    Convenience wrapper that derives initial state from the first entry.
    Use replay_state() directly if you have a custom initial state.

    Args:
        entries: List of trigger history entries starting with BirthTrigger

    Yields:
        Tuples of (entry, state_at_that_point) for each trigger.

    Raises:
        ValueError: If entries is empty or first entry isn't a BirthTrigger
    """
    if not entries:
        raise ValueError("Cannot replay empty conversation")

    initial_state = derive_initial_state(entries[0])
    yield from replay_state(entries, initial_state)


def get_final_state(
    entries: list[TriggerHistoryEntry],
    initial_state: State,
) -> State:
    """
    Get the final state after replaying all entries.

    Args:
        entries: List of trigger history entries
        initial_state: The state at the start of the conversation (will be mutated)

    Returns:
        The state after all actions have been applied (same object as initial_state)
    """
    for entry in entries:
        for action_data in entry.actions_taken:
            apply_action_state_change(initial_state, action_data)
    return initial_state
