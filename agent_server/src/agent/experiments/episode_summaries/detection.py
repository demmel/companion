"""
Episode boundary detection algorithms.

Detects episode boundaries in a stream of memories based on time gaps.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re
import uuid
from agent.memory.dag.models import MemoryElement
from agent.llm.router import LLM
from agent.llm.models import SupportedModel
from agent.experiments.episode_summaries.models import (
    Episode,
    EpisodeDetectionResult,
    GapThresholdStats,
    GapSweepResult,
    TopicShift,
    TopicDetectionResult,
    SimilarityStats,
)
import math


def detect_episodes_by_gap(
    memories: list[MemoryElement], gap_minutes: int
) -> EpisodeDetectionResult:
    """
    Detect episodes using a fixed time gap threshold.

    Memories with gaps > gap_minutes between them are split into separate episodes.

    Args:
        memories: List of memory elements to analyze
        gap_minutes: Minimum gap in minutes to consider as episode boundary

    Returns:
        EpisodeDetectionResult with detected episodes
    """
    if not memories:
        return EpisodeDetectionResult(
            episodes=[],
            gap_threshold_minutes=gap_minutes,
            total_memories=0,
            orphan_memories=[],
        )

    # Sort memories by timestamp
    sorted_memories = sorted(memories, key=lambda m: m.timestamp)

    episodes: list[Episode] = []
    current_episode_memories: list[MemoryElement] = [sorted_memories[0]]

    for i in range(1, len(sorted_memories)):
        prev_memory = sorted_memories[i - 1]
        curr_memory = sorted_memories[i]

        # Calculate gap in minutes
        gap = (curr_memory.timestamp - prev_memory.timestamp).total_seconds() / 60

        if gap > gap_minutes:
            # Close current episode and start new one
            episode = _create_episode(current_episode_memories)
            episodes.append(episode)
            current_episode_memories = [curr_memory]
        else:
            # Continue current episode
            current_episode_memories.append(curr_memory)

    # Don't forget the last episode
    if current_episode_memories:
        episode = _create_episode(current_episode_memories)
        episodes.append(episode)

    return EpisodeDetectionResult(
        episodes=episodes,
        gap_threshold_minutes=gap_minutes,
        total_memories=len(memories),
        orphan_memories=[],
    )


def _create_episode(memories: list[MemoryElement]) -> Episode:
    """Create an Episode from a list of memories."""
    start_time = min(m.timestamp for m in memories)
    end_time = max(m.timestamp for m in memories)
    duration_minutes = (end_time - start_time).total_seconds() / 60

    return Episode(
        id=str(uuid.uuid4()),
        start_time=start_time,
        end_time=end_time,
        duration_minutes=duration_minutes,
        memory_ids=[m.id for m in memories],
        memory_count=len(memories),
    )


def analyze_gap_distribution(memories: list[MemoryElement]) -> dict:
    """
    Analyze the distribution of time gaps between memories.

    Returns statistics about gaps to help choose an appropriate threshold.
    """
    if len(memories) < 2:
        return {
            "count": len(memories),
            "gaps": [],
            "min_gap_minutes": 0,
            "max_gap_minutes": 0,
            "avg_gap_minutes": 0,
            "gap_distribution": {},
        }

    # Sort by timestamp
    sorted_memories = sorted(memories, key=lambda m: m.timestamp)

    # Calculate all gaps
    gaps: list[float] = []
    for i in range(1, len(sorted_memories)):
        gap_minutes = (
            sorted_memories[i].timestamp - sorted_memories[i - 1].timestamp
        ).total_seconds() / 60
        gaps.append(gap_minutes)

    # Calculate gap distribution by buckets
    buckets = [5, 15, 30, 60, 120, 240, 480, 1440]  # minutes
    distribution: dict[str, int] = {}
    for bucket in buckets:
        count = sum(1 for g in gaps if g <= bucket)
        distribution[f"<= {bucket} min"] = count

    return {
        "count": len(memories),
        "total_gaps": len(gaps),
        "min_gap_minutes": min(gaps) if gaps else 0,
        "max_gap_minutes": max(gaps) if gaps else 0,
        "avg_gap_minutes": sum(gaps) / len(gaps) if gaps else 0,
        "median_gap_minutes": sorted(gaps)[len(gaps) // 2] if gaps else 0,
        "gap_distribution": distribution,
        "large_gaps": [g for g in gaps if g > 30],  # Gaps > 30 min
    }


def run_gap_threshold_sweep(
    memories: list[MemoryElement],
    thresholds: list[int] | None = None,
) -> GapSweepResult:
    """
    Run episode detection with multiple gap thresholds to find optimal value.

    Args:
        memories: List of memory elements to analyze
        thresholds: List of gap thresholds to test (minutes).
                   Defaults to [15, 30, 60, 120, 240]

    Returns:
        GapSweepResult with statistics for each threshold
    """
    if thresholds is None:
        thresholds = [15, 30, 60, 120, 240]

    results: list[GapThresholdStats] = []

    for threshold in thresholds:
        detection_result = detect_episodes_by_gap(memories, threshold)

        if detection_result.episodes:
            sizes = [ep.memory_count for ep in detection_result.episodes]
            durations = [ep.duration_minutes for ep in detection_result.episodes]

            stats = GapThresholdStats(
                gap_minutes=threshold,
                episode_count=len(detection_result.episodes),
                sizes={
                    "min": min(sizes),
                    "max": max(sizes),
                    "avg": sum(sizes) / len(sizes),
                },
                durations={
                    "min_minutes": min(durations),
                    "max_minutes": max(durations),
                    "avg_minutes": sum(durations) / len(durations),
                },
            )
        else:
            stats = GapThresholdStats(
                gap_minutes=threshold,
                episode_count=0,
                sizes={"min": 0, "max": 0, "avg": 0},
                durations={"min_minutes": 0, "max_minutes": 0, "avg_minutes": 0},
            )

        results.append(stats)

    # Generate recommendation
    recommendation = _generate_recommendation(results)

    return GapSweepResult(
        thresholds=results,
        total_memories=len(memories),
        recommendation=recommendation,
    )


def _generate_recommendation(results: list[GapThresholdStats]) -> str:
    """Generate a recommendation based on threshold sweep results."""
    if not results:
        return "No results to analyze"

    # Find threshold with reasonable episode count (not too many, not too few)
    # Prefer 3-20 episodes with average size > 5
    best_threshold = None
    best_score = -1

    for stats in results:
        if stats.episode_count == 0:
            continue

        # Score based on episode count and average size
        # Prefer more episodes but not too fragmented
        episode_score = min(stats.episode_count, 20) / 20  # Normalize to 0-1
        size_score = min(stats.sizes["avg"], 30) / 30  # Prefer avg size up to 30

        # Combined score
        score = episode_score * 0.5 + size_score * 0.5

        if score > best_score:
            best_score = score
            best_threshold = stats

    if best_threshold:
        return (
            f"{best_threshold.gap_minutes}-minute threshold produces "
            f"{best_threshold.episode_count} episodes with average size "
            f"{best_threshold.sizes['avg']:.1f} memories"
        )
    else:
        return "Unable to determine optimal threshold - check data"


# =============================================================================
# Topic-Based Detection (Approach C)
# =============================================================================


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns value between -1 and 1, where 1 means identical direction.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def analyze_similarity_distribution(
    memories: list[MemoryElement],
) -> SimilarityStats:
    """
    Analyze the distribution of embedding similarities between consecutive memories.

    Returns statistics to help choose an appropriate similarity threshold.
    """
    # Sort by timestamp
    sorted_memories = sorted(memories, key=lambda m: m.timestamp)

    # Calculate similarities between consecutive memories
    similarities: list[float] = []
    for i in range(1, len(sorted_memories)):
        prev = sorted_memories[i - 1]
        curr = sorted_memories[i]

        if prev.embedding_vector and curr.embedding_vector:
            sim = cosine_similarity(prev.embedding_vector, curr.embedding_vector)
            similarities.append(sim)

    if not similarities:
        return SimilarityStats(
            min_similarity=0,
            max_similarity=0,
            avg_similarity=0,
            median_similarity=0,
            std_similarity=0,
            low_similarity_count={},
        )

    # Calculate statistics
    avg = sum(similarities) / len(similarities)
    variance = sum((s - avg) ** 2 for s in similarities) / len(similarities)
    std = math.sqrt(variance)

    sorted_sims = sorted(similarities)
    median = sorted_sims[len(sorted_sims) // 2]

    # Count below various thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    low_counts = {}
    for t in thresholds:
        count = sum(1 for s in similarities if s < t)
        low_counts[f"< {t}"] = count

    return SimilarityStats(
        min_similarity=min(similarities),
        max_similarity=max(similarities),
        avg_similarity=avg,
        median_similarity=median,
        std_similarity=std,
        low_similarity_count=low_counts,
    )


def detect_topic_shifts(
    memories: list[MemoryElement],
    similarity_threshold: float = 0.7,
) -> list[TopicShift]:
    """
    Detect topic shifts where embedding similarity drops below threshold.

    Args:
        memories: List of memories with embedding vectors
        similarity_threshold: Similarity below this indicates topic shift

    Returns:
        List of TopicShift objects at detected boundaries
    """
    sorted_memories = sorted(memories, key=lambda m: m.timestamp)
    shifts: list[TopicShift] = []

    for i in range(1, len(sorted_memories)):
        prev = sorted_memories[i - 1]
        curr = sorted_memories[i]

        # Skip if either lacks embedding
        if not prev.embedding_vector or not curr.embedding_vector:
            continue

        similarity = cosine_similarity(prev.embedding_vector, curr.embedding_vector)

        if similarity < similarity_threshold:
            time_gap = (curr.timestamp - prev.timestamp).total_seconds() / 60
            shifts.append(
                TopicShift(
                    before_memory_id=prev.id,
                    after_memory_id=curr.id,
                    similarity=similarity,
                    time_gap_minutes=time_gap,
                    index=i,
                )
            )

    return shifts


def detect_episodes_by_topic(
    memories: list[MemoryElement],
    similarity_threshold: float = 0.7,
) -> TopicDetectionResult:
    """
    Detect episodes based on topic shifts (embedding similarity).

    Episodes are split where consecutive memories have low similarity,
    indicating a topic change regardless of time gap.

    Args:
        memories: List of memories with embedding vectors
        similarity_threshold: Similarity below this creates episode boundary

    Returns:
        TopicDetectionResult with episodes and shift information
    """
    if not memories:
        return TopicDetectionResult(
            episodes=[],
            similarity_threshold=similarity_threshold,
            total_memories=0,
        )

    sorted_memories = sorted(memories, key=lambda m: m.timestamp)

    # Track memories without embeddings
    no_embedding_count = sum(1 for m in memories if not m.embedding_vector)

    # Detect all topic shifts
    topic_shifts = detect_topic_shifts(memories, similarity_threshold)
    shift_indices = {s.index for s in topic_shifts}

    # Build episodes by splitting at shift points
    episodes: list[Episode] = []
    current_episode_memories: list[MemoryElement] = [sorted_memories[0]]

    for i in range(1, len(sorted_memories)):
        if i in shift_indices:
            # Topic shift - close current episode
            if current_episode_memories:
                episodes.append(_create_episode(current_episode_memories))
            current_episode_memories = [sorted_memories[i]]
        else:
            current_episode_memories.append(sorted_memories[i])

    # Don't forget last episode
    if current_episode_memories:
        episodes.append(_create_episode(current_episode_memories))

    return TopicDetectionResult(
        episodes=episodes,
        similarity_threshold=similarity_threshold,
        total_memories=len(memories),
        topic_shifts=topic_shifts,
        memories_without_embeddings=no_embedding_count,
    )


def run_topic_threshold_sweep(
    memories: list[MemoryElement],
    thresholds: list[float] | None = None,
) -> dict:
    """
    Run topic-based episode detection with multiple similarity thresholds.

    Args:
        memories: List of memories with embeddings
        thresholds: Similarity thresholds to test. Defaults to [0.5, 0.6, 0.7, 0.8]

    Returns:
        Dict with results for each threshold
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8]

    results = []
    for threshold in thresholds:
        detection = detect_episodes_by_topic(memories, threshold)

        if detection.episodes:
            sizes = [ep.memory_count for ep in detection.episodes]
            durations = [ep.duration_minutes for ep in detection.episodes]

            result = {
                "similarity_threshold": threshold,
                "episode_count": len(detection.episodes),
                "topic_shifts_count": len(detection.topic_shifts),
                "sizes": {
                    "min": min(sizes),
                    "max": max(sizes),
                    "avg": sum(sizes) / len(sizes),
                },
                "durations": {
                    "min_minutes": min(durations),
                    "max_minutes": max(durations),
                    "avg_minutes": sum(durations) / len(durations),
                },
            }
        else:
            result = {
                "similarity_threshold": threshold,
                "episode_count": 0,
                "topic_shifts_count": 0,
                "sizes": {"min": 0, "max": 0, "avg": 0},
                "durations": {"min_minutes": 0, "max_minutes": 0, "avg_minutes": 0},
            }

        results.append(result)

    return {
        "thresholds": results,
        "total_memories": len(memories),
    }


# =============================================================================
# Windowed Topic Detection (Phase 3)
# =============================================================================


def average_vectors(vectors: list[list[float]]) -> list[float]:
    """
    Compute the centroid (average) of multiple vectors.

    Args:
        vectors: List of vectors to average

    Returns:
        Centroid vector
    """
    if not vectors:
        raise ValueError("Cannot average empty list of vectors")

    dim = len(vectors[0])
    centroid = [0.0] * dim

    for vec in vectors:
        for i, val in enumerate(vec):
            centroid[i] += val

    n = len(vectors)
    return [val / n for val in centroid]


def detect_episodes_windowed(
    memories: list[MemoryElement],
    window_size: int = 5,
    similarity_threshold: float = 0.3,
    min_episode_size: int = 3,
) -> TopicDetectionResult:
    """
    Detect episodes using windowed similarity comparison.

    Instead of comparing each memory to just the previous one, compare to the
    centroid of the last N memories. This captures "drifting away from topic"
    more robustly.

    Args:
        memories: List of memories with embeddings
        window_size: Number of recent memories to average for comparison
        similarity_threshold: Split when similarity to window centroid < threshold
        min_episode_size: Merge episodes smaller than this into neighbors

    Returns:
        TopicDetectionResult with detected episodes
    """
    if not memories:
        return TopicDetectionResult(
            episodes=[],
            similarity_threshold=similarity_threshold,
            total_memories=0,
        )

    sorted_memories = sorted(memories, key=lambda m: m.timestamp)

    # Track memories without embeddings
    no_embedding_count = sum(1 for m in memories if not m.embedding_vector)

    # Build episodes using windowed comparison
    episodes_raw: list[list[MemoryElement]] = []
    current_episode: list[MemoryElement] = []
    window_embeddings: list[list[float]] = []
    topic_shifts: list[TopicShift] = []

    for i, memory in enumerate(sorted_memories):
        if not memory.embedding_vector:
            # No embedding - add to current episode without comparison
            current_episode.append(memory)
            continue

        if len(window_embeddings) < window_size:
            # Still building initial window - no comparison yet
            current_episode.append(memory)
            window_embeddings.append(memory.embedding_vector)
            continue

        # Compare to window centroid
        centroid = average_vectors(window_embeddings)
        similarity = cosine_similarity(memory.embedding_vector, centroid)

        if similarity < similarity_threshold:
            # Topic shift detected - close current episode
            if current_episode:
                episodes_raw.append(current_episode)

            # Record the shift
            prev_memory = sorted_memories[i - 1]
            time_gap = (memory.timestamp - prev_memory.timestamp).total_seconds() / 60
            topic_shifts.append(
                TopicShift(
                    before_memory_id=prev_memory.id,
                    after_memory_id=memory.id,
                    similarity=similarity,
                    time_gap_minutes=time_gap,
                    index=i,
                )
            )

            # Start new episode with fresh window
            current_episode = [memory]
            window_embeddings = [memory.embedding_vector]
        else:
            # Continue current episode
            current_episode.append(memory)
            # Update window (sliding)
            window_embeddings.pop(0)
            window_embeddings.append(memory.embedding_vector)

    # Don't forget last episode
    if current_episode:
        episodes_raw.append(current_episode)

    # Merge small episodes into neighbors
    episodes_merged = _merge_small_episodes(episodes_raw, min_episode_size)

    # Convert to Episode objects
    episodes = [_create_episode(ep_memories) for ep_memories in episodes_merged]

    return TopicDetectionResult(
        episodes=episodes,
        similarity_threshold=similarity_threshold,
        total_memories=len(memories),
        topic_shifts=topic_shifts,
        memories_without_embeddings=no_embedding_count,
    )


def _merge_small_episodes(
    episodes: list[list[MemoryElement]],
    min_size: int,
) -> list[list[MemoryElement]]:
    """Merge episodes smaller than min_size into their neighbors."""
    if not episodes or min_size <= 1:
        return episodes

    merged: list[list[MemoryElement]] = []

    for episode in episodes:
        if len(episode) >= min_size:
            merged.append(episode)
        elif merged:
            # Merge into previous episode
            merged[-1].extend(episode)
        else:
            # First episode is small - keep it, will merge with next
            merged.append(episode)

    # Handle case where last episode is small
    if len(merged) > 1 and len(merged[-1]) < min_size:
        merged[-2].extend(merged[-1])
        merged.pop()

    return merged


def run_windowed_sweep(
    memories: list[MemoryElement],
    window_sizes: list[int] | None = None,
    thresholds: list[float] | None = None,
    min_episode_size: int = 3,
) -> dict:
    """
    Run windowed detection with multiple parameter combinations.

    Args:
        memories: List of memories
        window_sizes: Window sizes to test. Defaults to [3, 5, 10]
        thresholds: Similarity thresholds to test. Defaults to [0.2, 0.3, 0.4]
        min_episode_size: Minimum episode size for merging

    Returns:
        Dict with results for each parameter combination
    """
    if window_sizes is None:
        window_sizes = [3, 5, 10]
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4]

    results = []

    for window_size in window_sizes:
        for threshold in thresholds:
            detection = detect_episodes_windowed(
                memories,
                window_size=window_size,
                similarity_threshold=threshold,
                min_episode_size=min_episode_size,
            )

            if detection.episodes:
                sizes = [ep.memory_count for ep in detection.episodes]
                durations = [ep.duration_minutes for ep in detection.episodes]

                result = {
                    "window_size": window_size,
                    "similarity_threshold": threshold,
                    "episode_count": len(detection.episodes),
                    "topic_shifts_count": len(detection.topic_shifts),
                    "sizes": {
                        "min": min(sizes),
                        "max": max(sizes),
                        "avg": sum(sizes) / len(sizes),
                    },
                    "durations": {
                        "min_minutes": min(durations),
                        "max_minutes": max(durations),
                        "avg_minutes": sum(durations) / len(durations),
                    },
                }
            else:
                result = {
                    "window_size": window_size,
                    "similarity_threshold": threshold,
                    "episode_count": 0,
                    "topic_shifts_count": 0,
                    "sizes": {"min": 0, "max": 0, "avg": 0},
                    "durations": {"min_minutes": 0, "max_minutes": 0, "avg_minutes": 0},
                }

            results.append(result)

    return {
        "results": results,
        "total_memories": len(memories),
        "min_episode_size": min_episode_size,
    }


# =============================================================================
# LLM-Based Boundary Detection (Phase 4 - Approach D)
# =============================================================================


def format_memories_for_boundary_detection(
    memories: list[MemoryElement],
    start_index: int = 0,
) -> str:
    """
    Format memories as a numbered list for LLM prompt.

    Args:
        memories: List of memories to format (should be pre-sorted by timestamp)
        start_index: Starting index for numbering

    Returns:
        Formatted string with numbered memories
    """
    lines = []
    for i, memory in enumerate(memories):
        idx = start_index + i
        timestamp = memory.timestamp.strftime("%Y-%m-%d %H:%M")
        # Truncate very long content
        content = memory.content[:300]
        if len(memory.content) > 300:
            content += "..."
        lines.append(f"[{idx}] {timestamp}: {content}")

    return "\n".join(lines)


@dataclass
class EpisodeBoundary:
    """An episode boundary with description."""

    starts_at: int
    about: str


def parse_boundary_response(response: str, num_memories: int) -> list[EpisodeBoundary]:
    """
    Extract episode boundaries from LLM response.

    Supports two formats:
    1. Description-first: "Description of episode" index
       Example: "Morning greeting and plans" 33
    2. JSON format (fallback): [{"about": "...", "starts_at": N}]

    Args:
        response: LLM response text
        num_memories: Total number of memories (for validation)

    Returns:
        List of EpisodeBoundary objects
    """
    boundaries: list[EpisodeBoundary] = []

    # Try description-first format: "description" index
    # Match: "text" followed by whitespace and a number
    desc_pattern = re.compile(r'"([^"]+)"\s+(\d+)')
    desc_matches = desc_pattern.findall(response)
    if desc_matches:
        for about, idx_str in desc_matches:
            idx = int(idx_str)
            if 0 <= idx < num_memories:
                boundaries.append(EpisodeBoundary(starts_at=idx, about=about))
        if boundaries:
            return sorted(boundaries, key=lambda b: b.starts_at)

    # Fallback: Try to find JSON array in response
    json_match = re.search(r"\[.*\]", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            for item in data:
                if isinstance(item, dict) and "starts_at" in item:
                    idx = item["starts_at"]
                    about = item.get("about", "")
                    if 0 <= idx < num_memories:
                        boundaries.append(EpisodeBoundary(starts_at=idx, about=about))
                elif isinstance(item, int):
                    # Fallback for old format
                    if 0 <= item < num_memories:
                        boundaries.append(EpisodeBoundary(starts_at=item, about=""))
            if boundaries:
                return sorted(boundaries, key=lambda b: b.starts_at)
        except json.JSONDecodeError:
            pass

    # Last resort fallback: try to extract numbers from response
    numbers = re.findall(r"\b(\d+)\b", response)
    if numbers:
        indices = [int(n) for n in numbers if 0 <= int(n) < num_memories]
        unique_indices = sorted(set(indices))
        return [EpisodeBoundary(starts_at=idx, about="") for idx in unique_indices]

    return [
        EpisodeBoundary(starts_at=0, about="Start")
    ]  # Default: single episode starting at 0


def detect_episodes_llm_chunk(
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
    start_index: int = 0,
) -> list[int]:
    """
    Use LLM to detect episode boundaries in a chunk of memories.

    Args:
        memories: Chunk of memories (pre-sorted by timestamp)
        llm: LLM router instance
        model: Model to use
        start_index: Global starting index for this chunk

    Returns:
        List of global boundary indices
    """
    formatted = format_memories_for_boundary_detection(memories, start_index=0)

    prompt = f"""These are memories from an AI that exists continuously. It spends time thinking, journaling, exploring ideas, and occasionally interacting with a user.

Divide these into episodes - periods you'd remember as distinct.

Here are {len(memories)} consecutive memories, numbered 0 to {len(memories)-1}:

{formatted}

For each episode, first describe what it's about, then state the index where it starts.

Format each episode as: "description of episode" index

Example output:
"Initial creation and first interaction" 0
"Late night intimate session" 16
"Morning greeting and plans" 33

Episodes:"""

    response = llm.generate(
        model=model,
        prompt=prompt,
        caller="episode_boundary_detection",
    )

    # Parse response and adjust indices to global
    local_boundaries = parse_boundary_response(response, len(memories))

    # Convert to global indices
    global_boundaries = [
        EpisodeBoundary(starts_at=start_index + b.starts_at, about=b.about)
        for b in local_boundaries
    ]

    return global_boundaries


def detect_episodes_llm(
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
    chunk_size: int = 50,
    overlap: int = 10,
    max_chunks: int | None = None,
) -> TopicDetectionResult:
    """
    Use LLM to identify episode boundaries across the full conversation.

    Processes memories in overlapping chunks and merges results.

    Args:
        memories: Full list of memories
        llm: LLM router instance
        model: Model to use
        chunk_size: Number of memories per chunk
        overlap: Overlap between chunks for boundary consistency
        max_chunks: Maximum chunks to process (for testing). None = all.

    Returns:
        TopicDetectionResult with detected episodes
    """
    if not memories:
        return TopicDetectionResult(
            episodes=[],
            similarity_threshold=0.0,  # Not applicable for LLM
            total_memories=0,
        )

    sorted_memories = sorted(memories, key=lambda m: m.timestamp)

    # Collect all boundaries with descriptions
    all_boundaries: dict[int, str] = {0: "Start"}  # index -> about

    # Process chunks
    chunk_count = 0
    start = 0
    while start < len(sorted_memories):
        if max_chunks is not None and chunk_count >= max_chunks:
            break

        end = min(start + chunk_size, len(sorted_memories))
        chunk = sorted_memories[start:end]

        # Get boundaries for this chunk
        boundaries = detect_episodes_llm_chunk(chunk, llm, model, start_index=start)
        for b in boundaries:
            # Keep first description if duplicate boundary
            if b.starts_at not in all_boundaries:
                all_boundaries[b.starts_at] = b.about

        # Move to next chunk with overlap
        start = end - overlap if end < len(sorted_memories) else end
        chunk_count += 1

    # Sort boundaries and create episodes
    sorted_boundaries = sorted(all_boundaries.keys())

    episodes: list[Episode] = []
    for i, boundary_start in enumerate(sorted_boundaries):
        if i + 1 < len(sorted_boundaries):
            boundary_end = sorted_boundaries[i + 1]
        else:
            boundary_end = len(sorted_memories)

        episode_memories = sorted_memories[boundary_start:boundary_end]
        if episode_memories:
            episodes.append(_create_episode(episode_memories))

    # Create topic shifts for reporting (boundaries between episodes)
    topic_shifts: list[TopicShift] = []
    for boundary in sorted_boundaries[1:]:  # Skip first (index 0)
        if boundary < len(sorted_memories):
            before = sorted_memories[boundary - 1]
            after = sorted_memories[boundary]
            time_gap = (after.timestamp - before.timestamp).total_seconds() / 60
            topic_shifts.append(
                TopicShift(
                    before_memory_id=before.id,
                    after_memory_id=after.id,
                    similarity=0.0,  # Not applicable for LLM detection
                    time_gap_minutes=time_gap,
                    index=boundary,
                )
            )

    return TopicDetectionResult(
        episodes=episodes,
        similarity_threshold=0.0,  # Not applicable
        total_memories=len(memories),
        topic_shifts=topic_shifts,
        memories_without_embeddings=0,
    )


# Patterns that indicate a BAD boundary (action type changes, not topic changes)
BAD_BOUNDARY_PATTERNS = [
    "I continue to exist",
    "My mood changed",
    "I updated my appearance",
    "I add_priority",
    "I remove_priority",
    "I update_environment",
    "I get_creative_inspiration",
    "I responded to",
    "I thought about",
    "I thought:",
    "I search_web",
    "I browse_web",
]

# Patterns that indicate a GOOD boundary (user input, greetings)
GOOD_BOUNDARY_PATTERNS = [
    "David said to me:",
    "Chloe said to me:",
]

GREETING_PATTERNS = [
    "good morning",
    "good night",
    "goodnight",
    "hello",
    "hey",
    "hi ",
    "i'm back",
    "im back",
]


def strip_action_prefix(content: str) -> str:
    """Strip leading [✓] or similar prefixes from memory content."""
    import re

    # Remove leading [x] patterns (checkmarks, etc.)
    stripped = re.sub(r"^\s*\[[^\]]*\]\s*", "", content)
    return stripped


def is_bad_boundary(after_content: str) -> bool:
    """Check if a boundary's 'after' content indicates an action type change."""
    # Strip any leading [✓] prefix
    content = strip_action_prefix(after_content)
    content_lower = content.lower()
    for pattern in BAD_BOUNDARY_PATTERNS:
        if content_lower.startswith(pattern.lower()):
            return True
    return False


def is_good_boundary(after_content: str) -> bool:
    """Check if a boundary's 'after' content indicates a meaningful shift."""
    content_lower = after_content.lower()

    # Check for user/external input
    for pattern in GOOD_BOUNDARY_PATTERNS:
        if content_lower.startswith(pattern.lower()):
            # Check if it contains a greeting (even stronger signal)
            for greeting in GREETING_PATTERNS:
                if greeting in content_lower:
                    return True
            # User input is generally good, but not always a boundary
            return True

    return False


def filter_llm_boundaries(
    boundaries: list[int],
    sorted_memories: list[MemoryElement],
) -> list[int]:
    """
    Filter LLM-detected boundaries to remove action type changes.

    Keeps boundaries where:
    - The 'after' memory is user input (David said, Chloe said)
    - The 'after' memory contains a greeting pattern

    Removes boundaries where:
    - The 'after' memory is an action type (mood, appearance, priority, idle)
    """
    filtered = [0]  # Always keep the first boundary

    for boundary in boundaries:
        if boundary == 0:
            continue
        if boundary >= len(sorted_memories):
            continue

        after_memory = sorted_memories[boundary]
        after_content = after_memory.content

        # If it's a known bad pattern, skip it
        if is_bad_boundary(after_content):
            continue

        # If it's a good pattern (user input), keep it
        if is_good_boundary(after_content):
            filtered.append(boundary)
            continue

        # For other content, keep it (LLM thought it was a boundary)
        filtered.append(boundary)

    return sorted(set(filtered))


def detect_episodes_llm_filtered(
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
    chunk_size: int = 50,
    overlap: int = 10,
    max_chunks: int | None = None,
) -> TopicDetectionResult:
    """
    Use LLM to identify episode boundaries, then filter out action type changes.

    This is a hybrid approach: LLM detects potential boundaries, then rule-based
    filtering removes boundaries that are just action type changes.
    """
    # First, get raw LLM boundaries
    raw_result = detect_episodes_llm(
        memories=memories,
        llm=llm,
        model=model,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunks=max_chunks,
    )

    # Extract boundary indices from topic_shifts
    sorted_memories = sorted(memories, key=lambda m: m.timestamp)
    raw_boundaries = [0] + [shift.index for shift in raw_result.topic_shifts]

    # Filter boundaries
    filtered_boundaries = filter_llm_boundaries(raw_boundaries, sorted_memories)

    # Rebuild episodes from filtered boundaries
    episodes: list[Episode] = []
    for i, boundary_start in enumerate(filtered_boundaries):
        if i + 1 < len(filtered_boundaries):
            boundary_end = filtered_boundaries[i + 1]
        else:
            boundary_end = len(sorted_memories)

        episode_memories = sorted_memories[boundary_start:boundary_end]
        if episode_memories:
            episodes.append(_create_episode(episode_memories))

    # Create topic shifts for filtered boundaries
    topic_shifts: list[TopicShift] = []
    for boundary in filtered_boundaries[1:]:  # Skip first
        if boundary < len(sorted_memories):
            before = sorted_memories[boundary - 1]
            after = sorted_memories[boundary]
            time_gap = (after.timestamp - before.timestamp).total_seconds() / 60
            topic_shifts.append(
                TopicShift(
                    before_memory_id=before.id,
                    after_memory_id=after.id,
                    similarity=0.0,
                    time_gap_minutes=time_gap,
                    index=boundary,
                )
            )

    return TopicDetectionResult(
        episodes=episodes,
        similarity_threshold=0.0,
        total_memories=len(memories),
        topic_shifts=topic_shifts,
        memories_without_embeddings=0,
    )


# =============================================================================
# JSON-Format LLM Detection (Baseline Comparison)
# =============================================================================


def detect_episodes_llm_chunk_json(
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
    start_index: int = 0,
) -> list[EpisodeBoundary]:
    """
    Use LLM to detect episode boundaries using JSON output format.

    This is the baseline/original format for comparison with description-first.

    Args:
        memories: Chunk of memories (pre-sorted by timestamp)
        llm: LLM router instance
        model: Model to use
        start_index: Global starting index for this chunk

    Returns:
        List of EpisodeBoundary objects with global indices
    """
    formatted = format_memories_for_boundary_detection(memories, start_index=0)

    prompt = f"""These are memories from an AI that exists continuously. It spends time thinking, journaling, exploring ideas, and occasionally interacting with a user.

Divide these into episodes - coherent periods that would be remembered as distinct experiences.

Here are {len(memories)} consecutive memories, numbered 0 to {len(memories)-1}:

{formatted}

Return a JSON array of episode boundaries. Each boundary marks where a new episode begins.
Only create boundaries where there's a meaningful shift in activity, topic, or context.

Format: [{{"starts_at": 0, "about": "description"}}, {{"starts_at": N, "about": "description"}}]

JSON:"""

    response = llm.generate(
        model=model,
        prompt=prompt,
        caller="episode_boundary_detection_json",
    )

    # Parse response
    local_boundaries = parse_boundary_response(response, len(memories))

    # Convert to global indices
    global_boundaries = [
        EpisodeBoundary(starts_at=start_index + b.starts_at, about=b.about)
        for b in local_boundaries
    ]

    return global_boundaries


def detect_episodes_llm_json(
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
    chunk_size: int = 50,
    overlap: int = 10,
    max_chunks: int | None = None,
) -> TopicDetectionResult:
    """
    Use LLM with JSON format to identify episode boundaries.

    This is the baseline format for comparison with description-first.

    Args:
        memories: Full list of memories
        llm: LLM router instance
        model: Model to use
        chunk_size: Number of memories per chunk
        overlap: Overlap between chunks for boundary consistency
        max_chunks: Maximum chunks to process (for testing). None = all.

    Returns:
        TopicDetectionResult with detected episodes
    """
    if not memories:
        return TopicDetectionResult(
            episodes=[],
            similarity_threshold=0.0,
            total_memories=0,
        )

    sorted_memories = sorted(memories, key=lambda m: m.timestamp)

    # Collect all boundaries with descriptions
    all_boundaries: dict[int, str] = {0: "Start"}

    # Process chunks
    chunk_count = 0
    start = 0
    while start < len(sorted_memories):
        if max_chunks is not None and chunk_count >= max_chunks:
            break

        end = min(start + chunk_size, len(sorted_memories))
        chunk = sorted_memories[start:end]

        # Get boundaries for this chunk using JSON format
        boundaries = detect_episodes_llm_chunk_json(
            chunk, llm, model, start_index=start
        )
        for b in boundaries:
            if b.starts_at not in all_boundaries:
                all_boundaries[b.starts_at] = b.about

        # Move to next chunk with overlap
        start = end - overlap if end < len(sorted_memories) else end
        chunk_count += 1

    # Sort boundaries and create episodes
    sorted_boundaries = sorted(all_boundaries.keys())

    episodes: list[Episode] = []
    for i, boundary_start in enumerate(sorted_boundaries):
        if i + 1 < len(sorted_boundaries):
            boundary_end = sorted_boundaries[i + 1]
        else:
            boundary_end = len(sorted_memories)

        episode_memories = sorted_memories[boundary_start:boundary_end]
        if episode_memories:
            episodes.append(_create_episode(episode_memories))

    # Create topic shifts for reporting
    topic_shifts: list[TopicShift] = []
    for boundary in sorted_boundaries[1:]:
        if boundary < len(sorted_memories):
            before = sorted_memories[boundary - 1]
            after = sorted_memories[boundary]
            time_gap = (after.timestamp - before.timestamp).total_seconds() / 60
            topic_shifts.append(
                TopicShift(
                    before_memory_id=before.id,
                    after_memory_id=after.id,
                    similarity=0.0,
                    time_gap_minutes=time_gap,
                    index=boundary,
                )
            )

    return TopicDetectionResult(
        episodes=episodes,
        similarity_threshold=0.0,
        total_memories=len(memories),
        topic_shifts=topic_shifts,
        memories_without_embeddings=0,
    )
