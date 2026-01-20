"""Seed selection strategies for dream generation."""

import random
from typing import Optional

from agent.memory.dag.models import MemoryGraph, MemoryElement

# Keywords that suggest emotional content
EMOTIONAL_KEYWORDS = [
    "feel",
    "felt",
    "feeling",
    "emotion",
    "happy",
    "sad",
    "angry",
    "fear",
    "love",
    "hate",
    "joy",
    "sorrow",
    "anxious",
    "excited",
    "worried",
    "grateful",
    "thankful",
    "miss",
    "missed",
    "heart",
    "soul",
    "tears",
    "laugh",
    "smile",
    "cry",
    "hug",
    "warm",
    "cold",
    "beautiful",
    "wonderful",
    "terrible",
    "amazing",
    "awful",
    "proud",
    "ashamed",
    "guilty",
    "hurt",
    "pain",
    "comfort",
    "trust",
    "hope",
    "dream",
    "wish",
    "care",
    "connected",
    "understood",
    "lonely",
    "together",
    "special",
    "meaningful",
    "intimate",
]


def select_random_seed(memory_graph: MemoryGraph) -> Optional[str]:
    """
    Pick a random memory as seed.

    Args:
        memory_graph: The memory graph to select from

    Returns:
        Memory ID of the selected seed, or None if no memories exist
    """
    if not memory_graph.elements:
        return None

    return random.choice(list(memory_graph.elements.keys()))


def select_recent_seed(memory_graph: MemoryGraph) -> Optional[str]:
    """
    Pick the most recent memory as seed.

    Args:
        memory_graph: The memory graph to select from

    Returns:
        Memory ID of the most recent memory, or None if no memories exist
    """
    if not memory_graph.elements:
        return None

    # Sort by timestamp, get most recent
    sorted_elements = sorted(
        memory_graph.elements.values(), key=lambda m: m.timestamp, reverse=True
    )

    return sorted_elements[0].id


def _calculate_emotional_score(memory: MemoryElement) -> int:
    """
    Calculate an emotional score for a memory based on keyword presence.

    Args:
        memory: The memory to score

    Returns:
        Count of emotional keywords found in content
    """
    content_lower = memory.content.lower()
    score = 0
    for keyword in EMOTIONAL_KEYWORDS:
        if keyword in content_lower:
            score += 1
    return score


def select_emotional_seed(memory_graph: MemoryGraph) -> Optional[str]:
    """
    Pick a memory with high emotional content as seed.

    Uses keyword heuristics to identify emotionally significant memories.

    Args:
        memory_graph: The memory graph to select from

    Returns:
        Memory ID of an emotionally significant memory, or None if no memories exist
    """
    if not memory_graph.elements:
        return None

    # Score all memories by emotional content
    scored_memories = [
        (mem_id, _calculate_emotional_score(mem))
        for mem_id, mem in memory_graph.elements.items()
    ]

    # Sort by score descending
    scored_memories.sort(key=lambda x: x[1], reverse=True)

    # Get top emotional memories (top 10%)
    top_count = max(1, len(scored_memories) // 10)
    top_memories = scored_memories[:top_count]

    # Pick randomly from top emotional memories
    return random.choice(top_memories)[0]


def select_unprocessed_seed(memory_graph: MemoryGraph) -> Optional[str]:
    """
    Pick a memory that hasn't been recently accessed.

    Since we don't track access times, this currently picks randomly
    from older memories (bottom 50% by timestamp).

    Args:
        memory_graph: The memory graph to select from

    Returns:
        Memory ID of an older memory, or None if no memories exist
    """
    if not memory_graph.elements:
        return None

    # Sort by timestamp, get older memories
    sorted_elements = sorted(memory_graph.elements.values(), key=lambda m: m.timestamp)

    # Get bottom 50% (older memories)
    older_count = max(1, len(sorted_elements) // 2)
    older_memories = sorted_elements[:older_count]

    # Pick randomly from older memories
    return random.choice(older_memories).id


def select_seed(
    memory_graph: MemoryGraph, strategy: str, fixed_seed_id: Optional[str] = None
) -> Optional[str]:
    """
    Select a seed memory using the specified strategy.

    Args:
        memory_graph: The memory graph to select from
        strategy: One of 'random', 'recent', 'emotional', 'unprocessed'
        fixed_seed_id: If provided, use this seed instead of selecting

    Returns:
        Memory ID of the selected seed
    """
    if fixed_seed_id is not None:
        if fixed_seed_id in memory_graph.elements:
            return fixed_seed_id
        raise ValueError(f"Fixed seed ID {fixed_seed_id} not found in memory graph")

    if strategy == "random":
        return select_random_seed(memory_graph)
    elif strategy == "recent":
        return select_recent_seed(memory_graph)
    elif strategy == "emotional":
        return select_emotional_seed(memory_graph)
    elif strategy == "unprocessed":
        return select_unprocessed_seed(memory_graph)
    else:
        raise ValueError(f"Unknown seed selection strategy: {strategy}")
