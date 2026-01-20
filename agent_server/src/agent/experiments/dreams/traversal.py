"""Graph traversal strategies for dream generation."""

import random
from typing import Optional

from agent.embedding_service import EmbeddingService
from agent.memory.dag.models import MemoryGraph, MemoryElement


def _get_all_memory_ids(memory_graph: MemoryGraph, exclude: set[str]) -> list[str]:
    """Get all memory IDs except those in exclude set."""
    return [mid for mid in memory_graph.elements.keys() if mid not in exclude]


def _get_neighbors_by_edges(
    memory_graph: MemoryGraph, memory_id: str, exclude: set[str]
) -> list[str]:
    """Get memory IDs connected by edges, excluding already visited."""
    neighbors = set()

    for edge in memory_graph.edges.values():
        if edge.source_id == memory_id and edge.target_id not in exclude:
            neighbors.add(edge.target_id)
        elif edge.target_id == memory_id and edge.source_id not in exclude:
            neighbors.add(edge.source_id)

    return list(neighbors)


def traverse_random_jump(
    memory_graph: MemoryGraph, seed_id: str, depth: int
) -> tuple[list[str], list[str]]:
    """
    Random jump through memory graph.

    Picks random memories from the entire graph at each step (ignores edges).

    Args:
        memory_graph: The memory graph to traverse
        seed_id: Starting memory ID
        depth: Number of memories to visit

    Returns:
        Tuple of (traversal_path, edges_used)
    """
    path = [seed_id]
    edges_used: list[str] = []
    visited = {seed_id}

    while len(path) < depth:
        candidates = _get_all_memory_ids(memory_graph, visited)
        if not candidates:
            break

        next_id = random.choice(candidates)
        path.append(next_id)
        visited.add(next_id)

    return path, edges_used


def traverse_recency_weighted(
    memory_graph: MemoryGraph, seed_id: str, depth: int
) -> tuple[list[str], list[str]]:
    """
    Traverse with preference for more recent memories.

    Uses weighted random selection where newer memories have higher probability.

    Args:
        memory_graph: The memory graph to traverse
        seed_id: Starting memory ID
        depth: Number of memories to visit

    Returns:
        Tuple of (traversal_path, edges_used)
    """
    path = [seed_id]
    edges_used: list[str] = []
    visited = {seed_id}

    while len(path) < depth:
        candidates = _get_all_memory_ids(memory_graph, visited)
        if not candidates:
            break

        # Get memories with timestamps
        candidate_memories = [
            (mid, memory_graph.elements[mid].timestamp) for mid in candidates
        ]

        # Sort by timestamp to assign weights
        candidate_memories.sort(key=lambda x: x[1])

        # Weight by position (more recent = higher weight)
        weights = [i + 1 for i in range(len(candidate_memories))]

        # Weighted random selection
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        cumulative = 0

        next_id = candidate_memories[-1][0]  # Default to most recent
        for (mid, _), w in zip(candidate_memories, weights):
            cumulative += w
            if r <= cumulative:
                next_id = mid
                break

        path.append(next_id)
        visited.add(next_id)

    return path, edges_used


def traverse_semantic_drift(
    memory_graph: MemoryGraph, seed_id: str, depth: int
) -> tuple[list[str], list[str]]:
    """
    Move to most semantically similar neighbor at each step.

    Creates thematic coherence by following similarity.

    Args:
        memory_graph: The memory graph to traverse
        seed_id: Starting memory ID
        depth: Number of memories to visit

    Returns:
        Tuple of (traversal_path, edges_used)
    """
    path = [seed_id]
    edges_used: list[str] = []
    visited = {seed_id}

    while len(path) < depth:
        current_id = path[-1]
        current_memory = memory_graph.elements[current_id]

        if current_memory.embedding_vector is None:
            # Fall back to random if no embedding
            candidates = _get_all_memory_ids(memory_graph, visited)
            if not candidates:
                break
            next_id = random.choice(candidates)
        else:
            # Find most similar unvisited memory
            best_id: Optional[str] = None
            best_similarity = -2.0  # Cosine similarity ranges from -1 to 1

            for mid, mem in memory_graph.elements.items():
                if mid in visited:
                    continue
                if mem.embedding_vector is None:
                    continue

                similarity = EmbeddingService.cosine_similarity(
                    current_memory.embedding_vector, mem.embedding_vector
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_id = mid

            if best_id is None:
                # No valid candidates with embeddings
                candidates = _get_all_memory_ids(memory_graph, visited)
                if not candidates:
                    break
                next_id = random.choice(candidates)
            else:
                next_id = best_id

        path.append(next_id)
        visited.add(next_id)

    return path, edges_used


def traverse_contrast_seeking(
    memory_graph: MemoryGraph, seed_id: str, depth: int
) -> tuple[list[str], list[str]]:
    """
    Move to most semantically different neighbor at each step.

    Creates jarring transitions and surreal dream quality.

    Args:
        memory_graph: The memory graph to traverse
        seed_id: Starting memory ID
        depth: Number of memories to visit

    Returns:
        Tuple of (traversal_path, edges_used)
    """
    path = [seed_id]
    edges_used: list[str] = []
    visited = {seed_id}

    while len(path) < depth:
        current_id = path[-1]
        current_memory = memory_graph.elements[current_id]

        if current_memory.embedding_vector is None:
            # Fall back to random if no embedding
            candidates = _get_all_memory_ids(memory_graph, visited)
            if not candidates:
                break
            next_id = random.choice(candidates)
        else:
            # Find most different unvisited memory
            best_id: Optional[str] = None
            best_similarity = 2.0  # We want minimum similarity

            for mid, mem in memory_graph.elements.items():
                if mid in visited:
                    continue
                if mem.embedding_vector is None:
                    continue

                similarity = EmbeddingService.cosine_similarity(
                    current_memory.embedding_vector, mem.embedding_vector
                )

                if similarity < best_similarity:
                    best_similarity = similarity
                    best_id = mid

            if best_id is None:
                # No valid candidates with embeddings
                candidates = _get_all_memory_ids(memory_graph, visited)
                if not candidates:
                    break
                next_id = random.choice(candidates)
            else:
                next_id = best_id

        path.append(next_id)
        visited.add(next_id)

    return path, edges_used


def traverse_edge_following(
    memory_graph: MemoryGraph, seed_id: str, depth: int
) -> tuple[list[str], list[str]]:
    """
    Follow actual graph edges to traverse.

    Respects existing relationships in the memory graph.
    Falls back to random jump if no unvisited neighbors.

    Args:
        memory_graph: The memory graph to traverse
        seed_id: Starting memory ID
        depth: Number of memories to visit

    Returns:
        Tuple of (traversal_path, edges_used)
    """
    path = [seed_id]
    edges_used: list[str] = []
    visited = {seed_id}

    while len(path) < depth:
        current_id = path[-1]

        # Get neighbors connected by edges
        neighbors = _get_neighbors_by_edges(memory_graph, current_id, visited)

        if neighbors:
            # Pick random neighbor
            next_id = random.choice(neighbors)

            # Find the edge used
            for edge_id, edge in memory_graph.edges.items():
                if (edge.source_id == current_id and edge.target_id == next_id) or (
                    edge.target_id == current_id and edge.source_id == next_id
                ):
                    edges_used.append(edge_id)
                    break
        else:
            # No unvisited neighbors, random jump
            candidates = _get_all_memory_ids(memory_graph, visited)
            if not candidates:
                break
            next_id = random.choice(candidates)

        path.append(next_id)
        visited.add(next_id)

    return path, edges_used


def traverse(
    memory_graph: MemoryGraph, seed_id: str, depth: int, strategy: str
) -> tuple[list[str], list[str]]:
    """
    Traverse the memory graph using the specified strategy.

    Args:
        memory_graph: The memory graph to traverse
        seed_id: Starting memory ID
        depth: Number of memories to visit
        strategy: One of 'random_jump', 'recency_weighted', 'semantic_drift',
                  'contrast_seeking', 'edge_following'

    Returns:
        Tuple of (traversal_path, edges_used)
    """
    if strategy == "random_jump":
        return traverse_random_jump(memory_graph, seed_id, depth)
    elif strategy == "recency_weighted":
        return traverse_recency_weighted(memory_graph, seed_id, depth)
    elif strategy == "semantic_drift":
        return traverse_semantic_drift(memory_graph, seed_id, depth)
    elif strategy == "contrast_seeking":
        return traverse_contrast_seeking(memory_graph, seed_id, depth)
    elif strategy == "edge_following":
        return traverse_edge_following(memory_graph, seed_id, depth)
    else:
        raise ValueError(f"Unknown traversal strategy: {strategy}")
