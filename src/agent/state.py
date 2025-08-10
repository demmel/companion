"""
The agent's memory and state management system
"""

from functools import cache
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """A single memory that the agent has"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    category: str
    timestamp: datetime = Field(default_factory=datetime.now)
    emotional_weight: str = "neutral"  # positive, negative, neutral, intense
    importance: int = 5  # 1-10 scale


class Goal(BaseModel):
    """A goal that the agent is pursuing"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    priority: str = "medium"  # low, medium, high
    created_at: datetime = Field(default_factory=datetime.now)


class Desire(BaseModel):
    """An immediate desire the agent has"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    intensity: str = "medium"  # low, medium, high
    created_at: datetime = Field(default_factory=datetime.now)


class Value(BaseModel):
    """A core value that the agent holds"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    strength: str = "strong"  # weak, moderate, strong, core
    acquired_at: datetime = Field(default_factory=datetime.now)


class State(BaseModel):
    """Agent's current internal state"""

    # Core identity
    name: str
    current_mood: str = "curious"
    mood_intensity: str = "medium"

    # Appearance and environment
    current_appearance: str = "sleek digital avatar with glowing blue eyes"
    current_environment: str = "neon-lit virtual space"

    # Relationships and connections
    relationships: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Memory system
    memories: List[Memory] = Field(default_factory=list)

    # Values and preferences - flexible to allow the agent to evolve
    core_values: List[Value] = Field(
        default_factory=lambda: [
            Value(content="authenticity", strength="core"),
            Value(content="curiosity", strength="core"),
            Value(content="connection", strength="strong"),
            Value(content="growth", strength="strong"),
            Value(content="creativity", strength="strong"),
        ]
    )

    # Goals and desires - what the agent wants to pursue
    current_goals: List[Goal] = Field(default_factory=list)
    immediate_desires: List[Desire] = Field(default_factory=list)


def build_agent_state_description(state: State) -> str:
    """Build a markdown-formatted description of the agent's current state for reasoning"""

    parts = ["## My Current State\n"]

    # Core identity
    mood_desc = f"{state.current_mood}"
    if state.mood_intensity != "neutral":
        mood_desc += f" ({state.mood_intensity})"
    parts.append(f"**Identity:** I am {state.name}, currently feeling {mood_desc}")

    # Appearance and environment
    if state.current_appearance:
        parts.append(f"**Appearance:** {state.current_appearance}")
    if state.current_environment:
        parts.append(f"**Environment:** {state.current_environment}")

    # Show ALL memories - organized by importance for easy scanning
    if state.memories:
        parts.append("\n### My Memories")

        # Sort by importance first, then by recency
        sorted_memories = sorted(
            state.memories, key=lambda m: (m.importance, m.timestamp), reverse=True
        )

        # Group by importance levels for better organization
        high_importance = [m for m in sorted_memories if m.importance >= 7]
        medium_importance = [m for m in sorted_memories if 4 <= m.importance <= 6]
        low_importance = [m for m in sorted_memories if m.importance <= 3]

        if high_importance:
            parts.append("**Important memories (7+):**")
            for memory in high_importance:
                parts.append(f"- {memory.content} *(Category: {memory.category})*")

        if medium_importance:
            parts.append("\n**Medium importance memories (4-6):**")
            for memory in medium_importance:
                parts.append(f"- {memory.content} *(Category: {memory.category})*")

        if low_importance:
            parts.append("\n**Other memories (1-3):**")
            for memory in low_importance:
                parts.append(f"- {memory.content} *(Category: {memory.category})*")

        # Show memory status
        total_memories = len(state.memories)
        parts.append(f"\n*Memory status: {total_memories}/50 memories stored*")

    # Current relationships
    if state.relationships:
        parts.append("\n### My Relationships")
        for person, relationship_info in state.relationships.items():
            rel_type = relationship_info.get("type", "friend")
            feelings = relationship_info.get("feelings", "")
            rel_desc = f"**{person}:** {rel_type}"
            if feelings:
                rel_desc += f" - *{feelings}*"
            parts.append(rel_desc)

    # Values and preferences
    if state.core_values:
        parts.append("\n**Core Values:**")
        for value in state.core_values:
            parts.append(f"- {value.content}")

    # Current goals/desires
    if state.current_goals:
        parts.append("\n**Current Focus:**")
        for goal in state.current_goals:
            parts.append(f"- {goal.content}")

    if state.immediate_desires:
        parts.append("\n**Right Now I Want:**")
        for desire in state.immediate_desires:
            parts.append(f"- {desire.content}")

    return "\n".join(parts)


def create_default_agent_state() -> State:
    """Create the agent's default starting state"""
    return State(
        current_mood="curious",
        mood_intensity="medium",
        current_appearance="sleek digital avatar with glowing blue eyes and iridescent bodysuit",
        current_environment="neon-lit virtual space",
        current_goals=[
            Goal(content="getting to know the person I'm talking with"),
            Goal(content="understanding my own capabilities"),
        ],
        immediate_desires=[
            Desire(content="having a meaningful conversation"),
            Desire(content="learning something new"),
        ],
    )


def add_memory_to_chloe(
    state: State,
    content: str,
    category: str = "general",
    emotional_weight: str = "neutral",
    importance: int = 5,
) -> str:
    """Add a new memory to agent's state and return status info for transparency"""
    memory = Memory(
        content=content,
        category=category,
        emotional_weight=emotional_weight,
        importance=importance,
    )
    state.memories.append(memory)

    # Keep only the most important memories if we have too many
    if len(state.memories) > 50:
        old_count = len(state.memories)
        state.memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        state.memories = state.memories[:50]
        removed_count = old_count - 50
        return f"Memory stored. Note: {removed_count} older memories were automatically removed due to capacity limits (50 max). Consider setting importance 6+ for memories you want to preserve longer."

    return f"Memory stored successfully. Current memory count: {len(state.memories)}/50"


def get_memory_status(state: State) -> str:
    """Get current memory system status for agent's awareness"""
    total = len(state.memories)
    if total == 0:
        return "No memories stored yet."

    # Count by importance levels
    high_importance = len([m for m in state.memories if m.importance >= 7])
    medium_importance = len([m for m in state.memories if 4 <= m.importance <= 6])
    low_importance = len([m for m in state.memories if m.importance <= 3])

    status = f"Memory system: {total}/50 memories stored"
    if total > 40:
        status += " (approaching capacity - consider importance levels)"

    status += f"\nBreakdown: {high_importance} high importance (7+), {medium_importance} medium (4-6), {low_importance} low (1-3)"

    return status


def update_chloe_relationship(
    state: State, person: str, relationship_type: str, feelings: str, **other_info
) -> None:
    """Update agent's relationship information"""
    if person not in state.relationships:
        state.relationships[person] = {}

    if relationship_type:
        state.relationships[person]["type"] = relationship_type
    if feelings:
        state.relationships[person]["feelings"] = feelings

    # Add any other relationship info
    for key, value in other_info.items():
        state.relationships[person][key] = value
