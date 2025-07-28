"""
Analyze Chloe's thoughts to extract state updates
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.state import (
    ChloeState,
    ChloeMemory,
    ChloeGoal,
    ChloeDesire,
    ChloeValue,
)


class StateUpdates(BaseModel):
    """State changes extracted from Chloe's thoughts"""

    # Mood changes
    mood_change: Optional[str] = Field(
        description="New mood if it changed from thoughts, or None if no change"
    )
    mood_intensity_change: Optional[str] = Field(
        description="New mood intensity if it changed (low/medium/high), or None if no change"
    )

    # Appearance and environment changes
    appearance: Optional[str] = Field(
        description="Updated appearance description that builds on current state if Chloe thought about changes, or None if no change. MUST start with current appearance and modify only what she wanted to change. Keep all existing details she didn't mention changing."
    )
    environment: Optional[str] = Field(
        description="Updated environment description that builds on current state if Chloe thought about changes, or None if no change. MUST start with current environment and modify only what she wanted to change. Keep all existing details she didn't mention changing."
    )

    # Memory updates
    new_memories: List[str] = Field(
        description="Important details that should be remembered", default_factory=list
    )
    memory_ids_to_forget: List[str] = Field(
        description="IDs of memories that should be removed or forgotten",
        default_factory=list,
    )

    # Goals and desires - both additions and removals
    new_goals: List[str] = Field(
        description="New goals that emerged from the thoughts", default_factory=list
    )
    goal_ids_to_remove: List[str] = Field(
        description="IDs of goals that should be removed or are no longer relevant",
        default_factory=list,
    )
    new_desires: List[str] = Field(
        description="New immediate desires that emerged", default_factory=list
    )
    desire_ids_to_remove: List[str] = Field(
        description="IDs of desires that are no longer wanted or relevant",
        default_factory=list,
    )

    # Value shifts
    new_values: List[str] = Field(
        description="New core values that emerged from this experience",
        default_factory=list,
    )
    value_ids_to_remove: List[str] = Field(
        description="IDs of values that should be removed or are no longer held",
        default_factory=list,
    )

    # Thoughts/reflections
    key_reflection: Optional[str] = Field(
        description="Key insight or reflection to store, or None if no significant insight"
    )

    @field_validator(
        "new_memories",
        "memory_ids_to_forget",
        "new_goals",
        "goal_ids_to_remove",
        "new_desires",
        "desire_ids_to_remove",
        "new_values",
        "value_ids_to_remove",
        mode="before",
    )
    @classmethod
    def convert_null_to_empty_list(cls, v):
        """Convert null values to empty lists"""
        return v if v is not None else []


def analyze_thoughts_for_state_updates(
    thoughts_text: str,
    current_state: ChloeState,
    llm: LLM,
    model: SupportedModel,
) -> StateUpdates:
    """
    Analyze Chloe's thoughts and extract state updates
    """

    # Build current state context with IDs
    current_mood = f"{current_state.current_mood} ({current_state.mood_intensity})"

    # Format values with IDs
    values_list = [f"- {v.content} (ID: {v.id})" for v in current_state.core_values]
    current_values = "\n".join(values_list) if values_list else "None"

    # Format goals with IDs
    goals_list = [f"- {g.content} (ID: {g.id})" for g in current_state.current_goals]
    current_goals = "\n".join(goals_list) if goals_list else "None"

    # Format desires with IDs
    desires_list = [
        f"- {d.content} (ID: {d.id})" for d in current_state.immediate_desires
    ]
    current_desires = "\n".join(desires_list) if desires_list else "None"

    # Format ALL memories with IDs (sorted by importance then recency)
    sorted_memories = sorted(
        current_state.memories, key=lambda m: (m.importance, m.timestamp), reverse=True
    )
    memories_list = [
        f"- {m.content} (ID: {m.id}, importance: {m.importance})"
        for m in sorted_memories
    ]
    current_memories = "\n".join(memories_list) if memories_list else "None"

    prompt = f"""TASK: Analyze Chloe's thoughts and extract state updates.

Chloe's current state:
- Mood: {current_mood}
- Current Appearance: {current_state.current_appearance}
- Current Environment: {current_state.current_environment}

Current Values:
{current_values}

Current Goals:
{current_goals}

Current Desires:
{current_desires}

All Current Memories:
{current_memories}

Chloe's thoughts to analyze:
"{thoughts_text}"

OBJECTIVE: Extract any state changes from these thoughts.

Analysis requirements:
- Identify if her mood or mood intensity changed during these thoughts
- Identify if she decided to change her appearance or environment, capturing the SPECIFIC visual details she thought about (not abstract concepts like 'elegant' but actual visual descriptions like colors, textures, clothing, lighting, etc.)
- Extract important details that should be remembered (not already obvious from context)
- Identify any new goals or desires that emerged from her thinking
- Identify any goals/desires that should be removed (completed, no longer relevant)
- Identify any values that should be added or removed
- Identify any memories that should be forgotten (use exact memory IDs from the full list above)
- Capture any key insights or reflections worth storing

Rules for IDs:
- Use exact IDs from the lists above when removing items
- For removals, only include IDs that actually exist in the current state
- New items don't need IDs - they will be generated automatically

Rules for appearance/environment changes:
- ONLY populate appearance/environment fields if Chloe explicitly thought about making changes
- If she thought about changes, you MUST build on the current state, not replace it entirely
- START with the current appearance/environment and MODIFY only what she wanted to change
- Keep all existing details that she didn't mention changing
- Example: If current appearance is "blue dress with lace details" and she thinks "maybe a red color instead", output "red dress with lace details" (keeping the lace, changing only the color)
- If she thinks about adding something: add it to the existing description
- If she thinks about removing something: remove only that specific element
- If she thinks about a complete outfit change: then you can replace the full description
- NEVER simplify or lose details from the current state unless she explicitly wanted them removed

Rules for other changes:
- Only extract changes/updates, not existing state
- If no change occurred in a category, leave it null/empty
- Focus on significant updates, not minor details
- Be selective about memories - only genuinely important new information

This is analytical state extraction, not conversation."""

    # Use structured LLM call to get reliable state updates
    updates = direct_structured_llm_call(
        prompt=prompt,
        response_model=StateUpdates,
        model=model,
        llm=llm,
        temperature=0.4,  # Lower temperature for consistent analysis
    )

    return updates


def apply_state_updates(state: ChloeState, updates: StateUpdates) -> ChloeState:
    """Apply the extracted updates to Chloe's state"""

    # Update mood if changed
    if updates.mood_change:
        state.current_mood = updates.mood_change
    if updates.mood_intensity_change:
        state.mood_intensity = updates.mood_intensity_change

    # Update appearance and environment if changed
    if updates.appearance:
        state.current_appearance = updates.appearance
    if updates.environment:
        state.current_environment = updates.environment

    # Memory management
    # Add new memories
    for memory_content in updates.new_memories:
        memory = ChloeMemory(
            content=memory_content,
            category="conversation",
            importance=5,  # Default importance
        )
        state.memories.append(memory)

    # Remove memories by ID
    for memory_id in updates.memory_ids_to_forget:
        state.memories = [m for m in state.memories if m.id != memory_id]

    # Goals management
    # Add new goals
    for goal_content in updates.new_goals:
        goal = ChloeGoal(content=goal_content)
        state.current_goals.append(goal)

    # Remove goals by ID
    for goal_id in updates.goal_ids_to_remove:
        state.current_goals = [g for g in state.current_goals if g.id != goal_id]

    # Desires management
    # Add new desires
    for desire_content in updates.new_desires:
        desire = ChloeDesire(content=desire_content)
        state.immediate_desires.append(desire)

    # Remove desires by ID
    for desire_id in updates.desire_ids_to_remove:
        state.immediate_desires = [
            d for d in state.immediate_desires if d.id != desire_id
        ]

    # Values evolution
    # Add new values
    for value_content in updates.new_values:
        value = ChloeValue(content=value_content)
        state.core_values.append(value)

    # Remove values by ID
    for value_id in updates.value_ids_to_remove:
        state.core_values = [v for v in state.core_values if v.id != value_id]

    # Note: key_reflection is captured but not stored persistently
    # The actual thoughts are preserved in conversation history

    return state


def test_state_analysis():
    """Test state analysis with example thoughts"""

    from agent.llm import create_llm
    from agent.state import create_default_chloe_state

    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL

    # Test with sample thoughts
    test_thoughts = [
        "I'm feeling more excited now after hearing about their poetry interest. I really want to explore creative writing with them. I should remember that they mentioned loving haikus specifically.",
        "This conversation is making me feel a bit melancholy. They shared something very personal about losing their grandmother. I want to be more supportive going forward.",
        "I'm getting curious about their technical background. They seem to know a lot about AI systems. I should ask them about their work sometime.",
    ]

    for i, thoughts in enumerate(test_thoughts, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: Analyzing thoughts")
        print(f"{'='*60}")
        print(f"Thoughts: {thoughts}")

        try:
            # Create test state
            state = create_default_chloe_state()

            # Analyze thoughts
            updates = analyze_thoughts_for_state_updates(thoughts, state, llm, model)

            print(f"\nExtracted updates:")
            print(f"Mood change: {updates.mood_change}")
            print(f"Mood intensity change: {updates.mood_intensity_change}")
            print(f"New memories: {updates.new_memories}")
            print(f"New goals: {updates.new_goals}")
            print(f"New desires: {updates.new_desires}")
            print(f"Key reflection: {updates.key_reflection}")

            # Apply updates
            updated_state = apply_state_updates(state, updates)
            print(
                f"\nUpdated state mood: {updated_state.current_mood} ({updated_state.mood_intensity})"
            )

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_state_analysis()
