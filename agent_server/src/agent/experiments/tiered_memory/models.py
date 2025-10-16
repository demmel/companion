"""
Data models for tiered memory experiment.

Defines tier 3 (temporal conversations) and tier 4 (semantic clusters)
with drill-down capabilities to lower tiers.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MemoryTier(str, Enum):
    """Different tiers of memory granularity."""

    ATOMIC = "atomic"  # Tier 1: Individual actions/triggers
    TRIGGER_RESPONSE = "trigger_response"  # Tier 2: Trigger-response pairs
    CONVERSATION = "conversation"  # Tier 3: Temporal conversation boundaries
    SEMANTIC_CLUSTER = "semantic_cluster"  # Tier 4: Topic-based clusters


class ConversationBoundary(BaseModel):
    """
    Tier 3: Temporal conversation boundary.

    Represents a contiguous sequence of trigger-response pairs that form
    a coherent conversation or interaction session.
    """

    id: str
    trigger_entry_ids: List[str] = Field(
        description="IDs of TriggerHistoryEntry objects in this conversation"
    )
    start_timestamp: datetime = Field(
        description="Start time of the conversation"
    )
    end_timestamp: datetime = Field(
        description="End time of the conversation"
    )
    summary: str = Field(
        description="Compressed summary of the entire conversation"
    )
    embedding_vector: Optional[List[float]] = Field(
        default=None,
        description="Embedding of the summary for similarity search"
    )
    topic_tags: List[str] = Field(
        default_factory=list,
        description="Topic tags extracted from the conversation"
    )

    @property
    def duration_seconds(self) -> float:
        """Calculate conversation duration in seconds."""
        return (self.end_timestamp - self.start_timestamp).total_seconds()


class SemanticCluster(BaseModel):
    """
    Tier 4: Semantic topic cluster.

    Groups related memories across time by topic/theme using embedding similarity.
    Can contain references to any lower tier.
    """

    id: str
    cluster_topic: str = Field(
        description="Main topic or theme of this cluster"
    )
    summary: str = Field(
        description="Compressed summary of the cluster content"
    )
    embedding_vector: Optional[List[float]] = Field(
        default=None,
        description="Embedding of the summary for similarity search"
    )

    # Drill-down references to lower tiers
    conversation_ids: List[str] = Field(
        default_factory=list,
        description="Tier 3 conversation IDs in this cluster"
    )
    trigger_entry_ids: List[str] = Field(
        default_factory=list,
        description="Tier 2 trigger entry IDs in this cluster"
    )
    memory_element_ids: List[str] = Field(
        default_factory=list,
        description="Tier 1 atomic memory IDs in this cluster"
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this cluster was created"
    )
    cluster_size: int = Field(
        default=0,
        description="Total number of elements across all tiers"
    )

    def get_all_element_ids(self) -> List[str]:
        """Get all element IDs across all tiers."""
        return (
            self.conversation_ids +
            self.trigger_entry_ids +
            self.memory_element_ids
        )


class TieredMemoryGraph(BaseModel):
    """
    Extended memory graph with all 4 tiers.

    Wraps the existing MemoryGraph and adds tier 3 & 4 structures.
    """

    # Reference to existing tiers (stored separately)
    memory_graph_path: Optional[str] = Field(
        default=None,
        description="Path to serialized MemoryGraph (tier 1 & 2)"
    )

    # New tiers
    conversations: Dict[str, ConversationBoundary] = Field(
        default_factory=dict,
        description="Tier 3: Conversation boundaries by ID"
    )
    semantic_clusters: Dict[str, SemanticCluster] = Field(
        default_factory=dict,
        description="Tier 4: Semantic clusters by ID"
    )

    created_at: datetime = Field(
        default_factory=datetime.now
    )
    last_updated: datetime = Field(
        default_factory=datetime.now
    )

    def get_tier_counts(self) -> Dict[str, int]:
        """Get counts of elements in each tier."""
        return {
            "tier_3_conversations": len(self.conversations),
            "tier_4_clusters": len(self.semantic_clusters),
        }


class RetrievalResult(BaseModel):
    """
    Result from tiered retrieval containing mixed granularities.
    """

    tier: MemoryTier
    element_id: str
    score: float
    summary: str = Field(description="Content summary at this tier")
    drill_down_ids: List[str] = Field(
        default_factory=list,
        description="IDs of lower-tier elements for drill-down"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (timestamps, tags, etc.)"
    )


class TieredRetrievalResults(BaseModel):
    """
    Complete results from a tiered retrieval query.
    """

    query: str
    results: List[RetrievalResult] = Field(
        description="Retrieved results sorted by relevance"
    )
    total_results: int
    tiers_used: List[MemoryTier] = Field(
        description="Which tiers were queried"
    )
    retrieval_strategy: str = Field(
        description="Strategy used for this retrieval"
    )

    def get_results_by_tier(self, tier: MemoryTier) -> List[RetrievalResult]:
        """Filter results by tier."""
        return [r for r in self.results if r.tier == tier]

    def get_top_k(self, k: int) -> List[RetrievalResult]:
        """Get top k results by score."""
        return sorted(self.results, key=lambda r: r.score, reverse=True)[:k]
