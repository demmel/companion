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
    max_priorities: int = 25  # Maximum number of priorities
    next_priority_id: int = Field(default=1)  # Counter for generating sequential IDs


def build_agent_state_description(state: State) -> str:
    """Build a markdown-formatted description of the agent's current state for reasoning"""
    from agent.chain_of_action.prompts import format_section

    sections = []

    # Mood and context section
    context_parts = []
    mood_desc = f"{state.current_mood}"
    if state.mood_intensity != "neutral":
        mood_desc += f" ({state.mood_intensity})"
    context_parts.append(f"**Mood**: {mood_desc}")

    if state.current_appearance:
        context_parts.append(f"**Appearance:** {state.current_appearance}")
    if state.current_environment:
        context_parts.append(f"**Environment:** {state.current_environment}")

    sections.append(format_section("MY MOOD AND CONTEXT", "\n".join(context_parts)))

    # Core values section
    if state.core_values:
        values_parts = []
        for value in state.core_values:
            values_parts.append(f"- {value.content}")
        sections.append(format_section("MY CORE VALUES", "\n".join(values_parts)))

    # Current priorities section
    if state.current_priorities:
        priority_parts = []
        priority_parts.append("These are ordered by importance (most important first).")
        for i, priority in enumerate(state.current_priorities, 1):
            priority_parts.append(f"{i}. [id: {priority.id}] - {priority.content}")

        # Add grounding instruction for priority IDs
        existing_ids = ", ".join([p.id for p in state.current_priorities])
        priority_parts.append(f"\n**IMPORTANT:** The ONLY priority IDs that currently exist are: {existing_ids}")
        priority_parts.append("Any other priority ID (not listed above) does NOT exist and cannot be used in operations.")

        sections.append(format_section("MY CURRENT PRIORITIES", "\n".join(priority_parts)))

    return "\n\n".join(sections)


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
        max_priorities=25,
        next_priority_id=5,
    )
