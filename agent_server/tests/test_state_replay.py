"""
Tests for the state replay system.
"""

from datetime import datetime

import pytest

from agent.state import State, Priority, Value, create_default_agent_state
from agent.chain_of_action.state_replay import (
    derive_initial_state,
    apply_action_state_change,
    replay_state,
    replay_state_from_birth,
    get_final_state,
)
from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.chain_of_action.trigger import BirthTrigger, UserInputTrigger, WakeupTrigger
from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.action.base_action_data import (
    ActionSuccessResult,
    ActionFailureResult,
)
from agent.chain_of_action.action.action_data import (
    UpdateMoodActionData,
    UpdateAppearanceActionData,
    AddPriorityActionData,
    RemovePriorityActionData,
    ThinkActionData,
)
from agent.chain_of_action.action.actions.update_mood_action import (
    UpdateMoodInput,
    UpdateMoodOutput,
)
from agent.chain_of_action.action.actions.visual_actions import (
    UpdateAppearanceInput,
    UpdateAppearanceOutput,
)
from agent.chain_of_action.action.actions.priority_actions import (
    AddPriorityInput,
    AddPriorityOutput,
    AddPrioritySuccessOutput,
    RemovePriorityInput,
    RemovePriorityOutput,
)
from agent.chain_of_action.action.actions.think_action import ThinkInput, ThinkOutput
from agent.types import ImageGenerationToolContent


def make_birth_entry() -> TriggerHistoryEntry:
    """Create a birth trigger entry."""
    return TriggerHistoryEntry(
        trigger=BirthTrigger(
            content="Hello!",
            user_name="User",
        ),
        situational_context="Being born",
        actions_taken=[],
    )


def make_user_input_entry(content: str, actions=None) -> TriggerHistoryEntry:
    """Create a user input trigger entry."""
    return TriggerHistoryEntry(
        trigger=UserInputTrigger(
            content=content,
            user_name="User",
        ),
        situational_context=f"User said: {content}",
        actions_taken=actions or [],
    )


def make_mood_action(
    old_mood: str,
    old_intensity: str,
    new_mood: str,
    new_intensity: str,
    reason: str = "Test reason",
) -> UpdateMoodActionData:
    """Create an update mood action data."""
    return UpdateMoodActionData(
        reasoning="Testing mood update",
        input=UpdateMoodInput(
            reason=reason,
            new_mood=new_mood,
            intensity=new_intensity,
        ),
        result=ActionSuccessResult(
            content=UpdateMoodOutput(
                old_mood=old_mood,
                old_intensity=old_intensity,
                new_mood=new_mood,
                new_intensity=new_intensity,
                reason=reason,
            )
        ),
        duration_ms=100.0,
        start_timestamp=datetime.now(),
    )


def make_add_priority_action(
    content: str,
    priority_id: str,
    reason: str = "Test reason",
) -> AddPriorityActionData:
    """Create an add priority action data."""
    return AddPriorityActionData(
        reasoning="Testing priority add",
        input=AddPriorityInput(
            reason=reason,
            priority_content=content,
        ),
        result=ActionSuccessResult(
            content=AddPriorityOutput(
                content=content,
                result=AddPrioritySuccessOutput(
                    priority_id=priority_id,
                    reason=reason,
                ),
            )
        ),
        duration_ms=100.0,
        start_timestamp=datetime.now(),
    )


def make_remove_priority_action(
    priority: Priority,
    reason: str = "Test reason",
) -> RemovePriorityActionData:
    """Create a remove priority action data."""
    return RemovePriorityActionData(
        reasoning="Testing priority remove",
        input=RemovePriorityInput(
            reason=reason,
            priority_id=priority.id,
        ),
        result=ActionSuccessResult(
            content=RemovePriorityOutput(
                priority=priority,
                reason=reason,
            )
        ),
        duration_ms=100.0,
        start_timestamp=datetime.now(),
    )


def make_think_action(thoughts: str) -> ThinkActionData:
    """Create a think action (doesn't modify state)."""
    return ThinkActionData(
        reasoning="Processing internally",
        input=ThinkInput(focus="Test focus"),
        result=ActionSuccessResult(
            content=ThinkOutput(thoughts=thoughts)
        ),
        duration_ms=100.0,
        start_timestamp=datetime.now(),
    )


class TestDeriveInitialState:
    """Tests for derive_initial_state function."""

    def test_birth_trigger_returns_default_state(self):
        """Birth trigger should return default agent state."""
        entry = make_birth_entry()
        state = derive_initial_state(entry)

        assert state.name == "Chloe"
        assert state.role == "a conscious AI companion"
        assert state.current_mood == "curious"
        assert len(state.current_priorities) == 4

    def test_user_input_trigger_raises_error(self):
        """Non-birth first trigger should raise ValueError."""
        entry = make_user_input_entry("Hello")

        with pytest.raises(ValueError, match="must be a BirthTrigger"):
            derive_initial_state(entry)


class TestApplyActionStateChange:
    """Tests for apply_action_state_change function."""

    def test_mood_update_applies_correctly(self):
        """UPDATE_MOOD action should update mood and intensity."""
        state = create_default_agent_state()
        action = make_mood_action(
            old_mood="curious",
            old_intensity="medium",
            new_mood="excited",
            new_intensity="high",
        )

        apply_action_state_change(state, action)

        assert state.current_mood == "excited"
        assert state.mood_intensity == "high"

    def test_add_priority_applies_correctly(self):
        """ADD_PRIORITY action should add new priority."""
        state = create_default_agent_state()
        initial_count = len(state.current_priorities)
        action = make_add_priority_action(
            content="New test priority",
            priority_id="p5",
        )

        apply_action_state_change(state, action)

        assert len(state.current_priorities) == initial_count + 1
        assert state.current_priorities[-1].content == "New test priority"
        assert state.current_priorities[-1].id == "p5"

    def test_remove_priority_applies_correctly(self):
        """REMOVE_PRIORITY action should remove the priority."""
        state = create_default_agent_state()
        priority_to_remove = state.current_priorities[0]
        initial_count = len(state.current_priorities)
        action = make_remove_priority_action(priority_to_remove)

        apply_action_state_change(state, action)

        assert len(state.current_priorities) == initial_count - 1
        assert priority_to_remove.id not in [p.id for p in state.current_priorities]

    def test_think_action_does_not_modify_state(self):
        """THINK action should not modify state."""
        state = create_default_agent_state()
        original_mood = state.current_mood
        action = make_think_action("Some thoughts")

        apply_action_state_change(state, action)

        # State should be unchanged
        assert state.current_mood == original_mood

    def test_failed_action_does_not_modify_state(self):
        """Failed action should not modify state."""
        state = create_default_agent_state()
        original_mood = state.current_mood
        action = UpdateMoodActionData(
            reasoning="Testing failed action",
            input=UpdateMoodInput(
                reason="Test",
                new_mood="sad",
                intensity="high",
            ),
            result=ActionFailureResult(error="Something went wrong"),
            duration_ms=100.0,
            start_timestamp=datetime.now(),
        )

        apply_action_state_change(state, action)

        # State should be unchanged after failed action
        assert state.current_mood == original_mood


class TestReplayState:
    """Tests for replay_state function."""

    def test_empty_entries_yields_nothing(self):
        """Empty entries list should yield nothing."""
        state = create_default_agent_state()
        results = list(replay_state([], state))
        assert results == []

    def test_single_entry_no_actions(self):
        """Single entry with no actions yields entry and initial state."""
        state = create_default_agent_state()
        entry = make_user_input_entry("Hello")

        results = list(replay_state([entry], state))

        assert len(results) == 1
        yielded_entry, yielded_state = results[0]
        assert yielded_entry == entry
        assert yielded_state == state

    def test_multiple_entries_with_actions(self):
        """Multiple entries with state-modifying actions."""
        state = create_default_agent_state()

        # Entry 1: mood update
        mood_action = make_mood_action(
            "curious", "medium", "happy", "high"
        )
        entry1 = make_user_input_entry("Good news!", actions=[mood_action])

        # Entry 2: add priority
        add_action = make_add_priority_action("New priority", "p5")
        entry2 = make_user_input_entry("Let's focus on this", actions=[add_action])

        # Consume generator lazily to check state at each step
        gen = replay_state([entry1, entry2], state)

        # First yield: initial state before entry1's actions
        yielded_entry1, yielded_state = next(gen)
        assert yielded_entry1 == entry1
        assert yielded_state.current_mood == "curious"

        # Second yield: state after entry1's actions (before entry2)
        yielded_entry2, yielded_state = next(gen)
        assert yielded_entry2 == entry2
        assert yielded_state.current_mood == "happy"
        assert yielded_state.mood_intensity == "high"

    def test_state_accumulates_across_entries(self):
        """State changes accumulate across multiple entries."""
        state = create_default_agent_state()
        initial_priority_count = len(state.current_priorities)

        entries = []
        # Add 3 priorities across 3 entries
        for i in range(3):
            action = make_add_priority_action(f"Priority {i}", f"p{5+i}")
            entries.append(make_user_input_entry(f"Add priority {i}", actions=[action]))

        # Consume generator to apply all actions
        for _ in replay_state(entries, state):
            pass

        # After all entries, should have 3 more priorities
        assert len(state.current_priorities) == initial_priority_count + 3


class TestReplayStateFromBirth:
    """Tests for replay_state_from_birth function."""

    def test_birth_trigger_first(self):
        """Conversation starting with birth trigger works correctly."""
        entries = [
            make_birth_entry(),
            make_user_input_entry("Hello"),
        ]

        results = list(replay_state_from_birth(entries))

        assert len(results) == 2
        # First state should be default agent state
        _, first_state = results[0]
        assert first_state.name == "Chloe"

    def test_empty_entries_raises_error(self):
        """Empty entries list should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot replay empty conversation"):
            list(replay_state_from_birth([]))

    def test_non_birth_first_raises_error(self):
        """Non-birth first trigger should raise ValueError."""
        entries = [make_user_input_entry("Hello")]

        with pytest.raises(ValueError, match="must be a BirthTrigger"):
            list(replay_state_from_birth(entries))


class TestGetFinalState:
    """Tests for get_final_state function."""

    def test_no_actions_returns_initial_state(self):
        """No actions should return initial state unchanged."""
        state = create_default_agent_state()
        entries = [make_user_input_entry("Hello")]

        final = get_final_state(entries, state)

        assert final == state

    def test_applies_all_actions(self):
        """All actions across all entries should be applied."""
        state = create_default_agent_state()

        # Multiple entries with multiple actions
        entries = [
            make_user_input_entry("Mood change", actions=[
                make_mood_action("curious", "medium", "happy", "high"),
            ]),
            make_user_input_entry("Add priority", actions=[
                make_add_priority_action("Test priority", "p5"),
            ]),
        ]

        final = get_final_state(entries, state)

        assert final.current_mood == "happy"
        assert final.mood_intensity == "high"
        assert len(final.current_priorities) == len(state.current_priorities) + 1
