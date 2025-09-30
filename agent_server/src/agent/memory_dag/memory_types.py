"""
Memory type definitions for the DAG-based memory system.

This module defines different types of memories and their characteristics for
retrieval weighting, handling, and classification.
"""

from enum import Enum
from typing import assert_never


class MemoryType(str, Enum):
    """Types of memories for retrieval weighting and handling."""

    COMMITMENT = "commitment"  # Promises, rules, boundaries that constrain behavior
    PREFERENCE = "preference"  # User likes/dislikes, choices, opinions
    FACTUAL = "factual"  # Objective information, data, facts
    EMOTIONAL = "emotional"  # Feelings, emotional states, relationship moments
    IDENTITY = "identity"  # Core self-knowledge, role, purpose
    PROCEDURAL = "procedural"  # How-to knowledge, processes, methods


class AgentControlledMemoryType(str, Enum):
    """Memory types that the agent can select during memory formation."""

    COMMITMENT = MemoryType.COMMITMENT.value
    PREFERENCE = MemoryType.PREFERENCE.value
    FACTUAL = MemoryType.FACTUAL.value
    EMOTIONAL = MemoryType.EMOTIONAL.value
    IDENTITY = MemoryType.IDENTITY.value
    PROCEDURAL = MemoryType.PROCEDURAL.value


# Memory type descriptions for LLM prompts
def get_memory_type_description(memory_type: MemoryType) -> str:
    match memory_type:
        case MemoryType.COMMITMENT:
            return "Promises, commitments, rules, or boundaries that constrain future behavior. These memories represent explicit agreements or decisions about what should or shouldn't be done."
        case MemoryType.PREFERENCE:
            return "User preferences, likes, dislikes, opinions, or choices. These capture what the user enjoys or wants."
        case MemoryType.FACTUAL:
            return "Objective information, facts, data, or neutral observations about the world, people, or situations."
        case MemoryType.EMOTIONAL:
            return "Emotional experiences, feelings, relationship moments, or significant personal interactions."
        case MemoryType.IDENTITY:
            return "Core self-knowledge about role, purpose, capabilities, or fundamental characteristics."
        case MemoryType.PROCEDURAL:
            return "How-to knowledge, processes, methods, or instructions for accomplishing tasks."
        case _:
            assert_never(_)


def get_prompt_memory_type_list() -> str:
    """Get formatted list of memory types for LLM prompts."""
    return ", ".join([t.value for t in AgentControlledMemoryType])


def get_memory_type_classification_descriptions() -> str:
    """Get detailed memory type descriptions for classification prompts."""
    descriptions = []
    for mem_type in AgentControlledMemoryType:
        full_type = MemoryType(mem_type.value)
        description = get_memory_type_description(full_type)
        descriptions.append(f"- **{mem_type.value}**: {description}")

    return "\n".join(descriptions)


def is_memory_superseded(memory_id: str, edges: dict) -> bool:
    """Check if a memory has been superseded by examining incoming SUPERSEDED_BY edges."""
    from .edge_types import GraphEdgeType

    for edge in edges.values():
        if (
            edge.target_id == memory_id
            and edge.edge_type == GraphEdgeType.SUPERSEDED_BY
        ):
            return True
    return False
