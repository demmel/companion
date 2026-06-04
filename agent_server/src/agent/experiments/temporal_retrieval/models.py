"""
Data models for temporal retrieval experiment.

Defines TimeReference, TemporalQuery, RetrievalResult, and IndexedEpisode.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class TimeReference(BaseModel):
    """A parsed time reference from a query."""

    raw_text: str
    ref_type: Literal["relative", "absolute", "emotional"]

    # For relative/absolute time references
    start_time: datetime | None = None
    end_time: datetime | None = None

    # For emotional/contextual time references
    mood_filter: str | None = None
    topic_filter: str | None = None
    event_anchor: str | None = None


class TemporalQuery(BaseModel):
    """A test query with expected results for evaluation."""

    query_text: str
    time_ref: TimeReference
    expected_episode_ids: list[str]
    expected_content_keywords: list[str]


@dataclass
class IndexedEpisode:
    """An episode with extracted metadata for indexing."""

    id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    memory_ids: list[str]
    memory_count: int

    # Generated content
    title: str | None = None
    summary: str | None = None

    # Extracted metadata for indexing
    topics: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    emotional_intensity: float = 0.0
    key_events: list[str] = field(default_factory=list)


class RetrievalResult(BaseModel):
    """Result of a temporal retrieval query."""

    query_text: str
    time_ref: TimeReference
    retrieved_episode_ids: list[str]
    retrieved_summaries: list[str]
    strategy: str
    latency_ms: float

    # For evaluation
    expected_episode_ids: list[str] = []


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating temporal retrieval."""

    # Time parsing accuracy
    time_parse_accuracy: float = 0.0
    relative_time_accuracy: float = 0.0
    absolute_time_accuracy: float = 0.0
    emotional_time_accuracy: float = 0.0

    # Episode retrieval metrics
    episode_precision: float = 0.0
    episode_recall: float = 0.0
    episode_f1: float = 0.0

    # Content relevance (LLM-judged)
    content_relevance: float = 0.0

    # Latency
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


@dataclass
class StrategyComparison:
    """Comparison of multiple retrieval strategies."""

    strategy_name: str
    metrics: EvaluationMetrics
    num_queries: int
    timestamp: datetime = field(default_factory=datetime.now)
