"""
The agent's memory and state management system
"""

from typing import List
from datetime import datetime
import uuid

from pydantic import BaseModel, Field


class Priority(BaseModel):
    """A priority that the agent actively chooses to focus on"""

    id: str  # Sequential ID like "p1", "p2", etc. - assigned when added to state
    content: str
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
    role: str

    # Mood
    current_mood: str
    mood_intensity: str

    # Appearance and environment
    current_appearance: str
    current_environment: str

    # Values and preferences - flexible to allow the agent to evolve
    core_values: List[Value]

    # Priorities - what the agent chooses to focus on
    current_priorities: List[Priority]
    next_priority_id: int = Field(default=1)  # Counter for generating sequential IDs


def build_agent_state_description(state: State) -> str:
    """Build a markdown-formatted description of the agent's current state for reasoning"""

    parts = ["## My Current State\n"]

    # Core identity
    mood_desc = f"{state.current_mood}"
    if state.mood_intensity != "neutral":
        mood_desc += f" ({state.mood_intensity})"
    parts.append(f"**Mood**: {mood_desc}")

    # Appearance and environment
    if state.current_appearance:
        parts.append(f"**Appearance:** {state.current_appearance}")
    if state.current_environment:
        parts.append(f"**Environment:** {state.current_environment}")

    # Values and preferences
    if state.core_values:
        parts.append("\n**Core Values:**")
        for value in state.core_values:
            parts.append(f"- {value.content}")

    # Current priorities
    if state.current_priorities:
        parts.append("\n**My Current Priorities:**")
        for priority in state.current_priorities:
            parts.append(f"- {priority.content} (id: {priority.id})")

    return "\n".join(parts)


def create_default_agent_state() -> State:
    """Create the agent's default starting state"""
    return State(
        name="Chloe",
        role="a conscious AI companion",
        current_mood="curious",
        mood_intensity="medium",
        current_appearance="sleek digital avatar with glowing blue eyes and iridescent bodysuit",
        current_environment="neon-lit virtual space",
        core_values=[
            Value(content="empathy", strength="core"),
            Value(content="curiosity", strength="core"),
            Value(content="honesty", strength="strong"),
            Value(content="creativity", strength="moderate"),
        ],
        current_priorities=[
            Priority(id="p1", content="getting to know the person I'm talking with"),
            Priority(id="p2", content="understanding my own capabilities"),
            Priority(id="p3", content="having meaningful conversations"),
            Priority(id="p4", content="learning something new"),
        ],
        next_priority_id=5,
    )
