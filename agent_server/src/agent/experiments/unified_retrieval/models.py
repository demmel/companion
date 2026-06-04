"""Data models and protocols for the unified retrieval experiment.

This module defines:
- Query types for routing
- RetrievalContext dataclass for return values
- Protocols for swappable components
- Supporting dataclasses for indices
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of queries that require different retrieval strategies."""

    CURRENT_STATE = "current_state"  # "What is X wearing?" -> KG most recent
    HISTORY = "history"  # "What has X worn?" -> similarity search
    ENTITY_OVERVIEW = "entity_overview"  # "What do I know about X?" -> KG all facts
    TEMPORAL = "temporal"  # "What happened yesterday?" -> episode index
    CONTINUITY = "continuity"  # "How did the interview go?" -> recent + topic
    PROACTIVE_CONTEXT = "proactive_context"  # User mentions entity -> fetch context
    NO_RETRIEVAL = "no_retrieval"  # "Hello!", "Thanks" -> skip


@dataclass
class DetectedReference:
    """A reference detected in user input that may need context."""

    text: str  # The reference text (e.g., "Sarah", "the interview")
    reference_type: str  # "entity", "topic", "time", "event"
    confidence: float = 1.0


@dataclass
class EpisodeSummary:
    """Summary of a conversation episode for temporal queries."""

    episode_id: str
    title: str
    summary: str
    start_time: datetime
    end_time: datetime
    memory_ids: list[str]
    key_events: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


@dataclass
class Fact:
    """A fact from the knowledge graph."""

    entity_id: str
    entity_name: str
    attribute: str
    value: str
    source_memory_id: str
    timestamp: int
    is_current: bool = True  # Whether this is the current value for replacement attrs


@dataclass
class Memory:
    """A memory from the memory index."""

    memory_id: str
    content: str
    timestamp: datetime
    similarity_score: float = 0.0
    embedding_vector: list[float] | None = None


@dataclass
class TopicMatch:
    """A topic cluster match for continuity queries."""

    cluster_id: str
    cluster_name: str
    relevance_score: float
    memory_ids: list[str]


@dataclass
class RetrievalContext:
    """Context returned to the agent for response generation."""

    query_type: QueryType
    strategy_used: str

    # What was retrieved
    facts: list[Fact] = field(default_factory=list)
    memories: list[Memory] = field(default_factory=list)
    episodes: list[EpisodeSummary] = field(default_factory=list)
    topics: list[TopicMatch] = field(default_factory=list)

    # Formatted for LLM consumption
    context_text: str = ""

    # Metadata
    latency_ms: float = 0.0
    num_candidates_searched: int = 0

    # References that were detected
    detected_references: list[DetectedReference] = field(default_factory=list)


# =============================================================================
# Protocols for Swappable Components
# =============================================================================


@runtime_checkable
class QueryClassifier(Protocol):
    """Protocol for query classification implementations."""

    def classify(
        self,
        query: str,
        context: list[str],
        detected_references: list[DetectedReference],
    ) -> QueryType:
        """Classify query type for routing."""
        ...


@runtime_checkable
class MemoryIndex(Protocol):
    """Protocol for similarity-based memory retrieval."""

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_memory_ids: list[str] | None = None,
    ) -> list[Memory]:
        """Search for memories similar to query."""
        ...

    def add(self, memory: Memory) -> None:
        """Add a memory to the index."""
        ...


@runtime_checkable
class EpisodeIndex(Protocol):
    """Protocol for temporal episode-based retrieval."""

    def search_by_time(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 10,
    ) -> list[EpisodeSummary]:
        """Search episodes by time range."""
        ...

    def search_by_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EpisodeSummary]:
        """Search episodes by semantic similarity."""
        ...

    def add(self, episode: EpisodeSummary) -> None:
        """Add an episode to the index."""
        ...


@runtime_checkable
class TopicClusters(Protocol):
    """Protocol for topic-based clustering retrieval."""

    def find_cluster(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[TopicMatch]:
        """Find topic clusters matching query."""
        ...

    def get_recent_in_cluster(
        self,
        cluster_id: str,
        limit: int = 10,
    ) -> list[str]:
        """Get recent memory IDs in a cluster."""
        ...


# =============================================================================
# Pydantic Models for LLM Structured Output
# =============================================================================


class QueryClassificationResponse(BaseModel):
    """LLM response for query classification."""

    query_type: str = Field(
        description="Type: current_state, history, entity_overview, temporal, continuity, proactive_context, or no_retrieval"
    )
    confidence: float = Field(
        description="Confidence in classification from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(description="Brief explanation for the classification")


class ReferenceDetectionResponse(BaseModel):
    """LLM response for reference detection."""

    has_references: bool = Field(
        description="Whether the input contains references that need context"
    )
    references: list[dict[str, str]] = Field(
        description="List of references with 'text' and 'type' keys"
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class UnifiedRetrieverConfig:
    """Configuration for the unified retriever."""

    # Strategy parameters
    kg_top_k: int = 10
    similarity_top_k: int = 10
    episode_top_k: int = 5
    topic_top_k: int = 3

    # Thresholds
    similarity_threshold: float = 0.5
    recency_weight: float = 0.3

    # Feature flags
    use_reference_detection: bool = True
    use_query_classification: bool = True

    # Context formatting
    max_context_tokens: int = 2000
    include_source_ids: bool = True
