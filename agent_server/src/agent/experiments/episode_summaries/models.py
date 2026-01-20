"""
Data models for episode detection and summarization.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Episode:
    """A contiguous conversation session detected by time gaps."""

    id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    memory_ids: list[str]
    memory_count: int

    # Generated content (populated by summarization)
    title: str | None = None
    summary: str | None = None
    key_events: list[str] | None = None
    emotional_arc: str | None = None
    topics_discussed: list[str] | None = None


@dataclass
class EpisodeDetectionResult:
    """Result of episode detection."""

    episodes: list[Episode]
    gap_threshold_minutes: int
    total_memories: int
    orphan_memories: list[str] = field(default_factory=list)


@dataclass
class GapThresholdStats:
    """Statistics for a single gap threshold."""

    gap_minutes: int
    episode_count: int
    sizes: dict[str, float]  # min, max, avg
    durations: dict[str, float]  # min_minutes, max_minutes, avg_minutes


@dataclass
class GapSweepResult:
    """Result of gap threshold sweep experiment."""

    thresholds: list[GapThresholdStats]
    total_memories: int
    recommendation: str


@dataclass
class TopicShift:
    """A detected topic shift between memories."""

    before_memory_id: str
    after_memory_id: str
    similarity: float
    time_gap_minutes: float
    index: int  # Index in sorted memory list


@dataclass
class TopicDetectionResult:
    """Result of topic-based episode detection."""

    episodes: list[Episode]
    similarity_threshold: float
    total_memories: int
    topic_shifts: list[TopicShift] = field(default_factory=list)
    memories_without_embeddings: int = 0


@dataclass
class SimilarityStats:
    """Statistics for similarity distribution analysis."""

    min_similarity: float
    max_similarity: float
    avg_similarity: float
    median_similarity: float
    std_similarity: float
    low_similarity_count: dict[str, int]  # count below various thresholds
