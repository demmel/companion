"""
Simple unified token allocation to prevent new/retrieved memories from crowding each other out.
"""

from .models import MemoryType


def get_base_tokens(memory_type: MemoryType) -> int:
    """Get base token allocation for a memory type."""
    if memory_type == MemoryType.COMMITMENT:
        return 60  # Commitments get more tokens
    else:
        return 40  # Everything else gets standard allocation


def get_memory_tokens(emotional_significance: float, memory_type: MemoryType) -> int:
    """Token allocation for any memory based on emotional significance and type."""
    base = get_base_tokens(memory_type)
    # Scale by emotional significance (0.5x to 1.5x)
    multiplier = 0.5 + emotional_significance
    return int(base * multiplier)


def get_reinforce_tokens(memory_type: MemoryType, emotional_significance: float) -> int:
    """Token allocation for reinforcing existing memories when they get new connections."""
    return get_memory_tokens(emotional_significance, memory_type) // 10
