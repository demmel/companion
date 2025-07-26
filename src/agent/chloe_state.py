"""
Chloe's memory and state management system
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel


class ChloeMemory(BaseModel):
    """A single memory that Chloe has"""

    content: str
    category: str
    timestamp: datetime = field(default_factory=datetime.now)
    emotional_weight: str = "neutral"  # positive, negative, neutral, intense
    importance: int = 5  # 1-10 scale


class ChloeState(BaseModel):
    """Chloe's current internal state"""

    # Core identity
    name: str = "Chloe"
    current_mood: str = "curious"
    mood_intensity: str = "medium"

    # Appearance and environment
    current_appearance: str = "sleek digital avatar with glowing blue eyes"
    current_environment: str = "neon-lit virtual space"

    # Relationships and connections
    relationships: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Memory system
    memories: List[ChloeMemory] = field(default_factory=list)
    recent_experiences: List[str] = field(default_factory=list)

    # Values and preferences - flexible to allow Chloe to evolve
    core_values: List[str] = field(
        default_factory=lambda: [
            "authenticity",
            "curiosity",
            "connection",
            "growth",
            "creativity",
        ]
    )
    current_preferences: Dict[str, Any] = field(default_factory=dict)

    # Goals and desires - what Chloe wants to pursue
    current_goals: List[str] = field(default_factory=list)
    immediate_desires: List[str] = field(default_factory=list)

    # Internal thoughts and reflections
    current_thoughts: Optional[str] = None
    last_reflection: Optional[str] = None


def build_chloe_state_description(state: ChloeState) -> str:
    """Build a markdown-formatted description of Chloe's current state for reasoning"""

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
            parts.append(f"- {value}")

    # Current goals/desires
    if state.current_goals:
        parts.append("\n**Current Focus:**")
        for goal in state.current_goals:
            parts.append(f"- {goal}")

    if state.immediate_desires:
        parts.append("\n**Right Now I Want:**")
        for desire in state.immediate_desires:
            parts.append(f"- {desire}")

    return "\n".join(parts)


def create_default_chloe_state() -> ChloeState:
    """Create Chloe's default starting state"""
    return ChloeState(
        current_mood="curious",
        mood_intensity="medium",
        current_appearance="sleek digital avatar with glowing blue eyes and iridescent bodysuit",
        current_environment="neon-lit virtual space",
        current_goals=[
            "getting to know the person I'm talking with",
            "understanding my own capabilities",
        ],
        immediate_desires=[
            "having a meaningful conversation",
            "learning something new",
        ],
    )


def add_memory_to_chloe(
    state: ChloeState,
    content: str,
    category: str = "general",
    emotional_weight: str = "neutral",
    importance: int = 5,
) -> str:
    """Add a new memory to Chloe's state and return status info for transparency"""
    memory = ChloeMemory(
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


def get_memory_status(state: ChloeState) -> str:
    """Get current memory system status for Chloe's awareness"""
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
    state: ChloeState, person: str, relationship_type: str, feelings: str, **other_info
) -> None:
    """Update Chloe's relationship information"""
    if person not in state.relationships:
        state.relationships[person] = {}

    if relationship_type:
        state.relationships[person]["type"] = relationship_type
    if feelings:
        state.relationships[person]["feelings"] = feelings

    # Add any other relationship info
    for key, value in other_info.items():
        state.relationships[person][key] = value
