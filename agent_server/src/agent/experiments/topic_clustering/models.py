"""
Data models for topic clustering experiment.

Uses dataclasses for internal processing and Pydantic models for serialization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field


class ClusteringMethod(str, Enum):
    """Supported clustering algorithms."""

    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    GMM = "gmm"  # Gaussian Mixture Model for soft clustering


class TopicNamingApproach(str, Enum):
    """Approaches for LLM-based topic naming."""

    SIMPLE = "simple"
    STRUCTURED = "structured"
    CONTRASTIVE = "contrastive"


@dataclass
class TopicCluster:
    """A group of semantically related memories."""

    id: str
    name: str  # LLM-generated topic name
    description: str  # LLM-generated description
    memory_ids: List[str]  # Memory element IDs in this cluster
    centroid: List[float]  # Average embedding vector
    coherence_score: float  # Intra-cluster tightness metric
    keywords: List[str]  # Key terms extracted from topic


@dataclass
class SoftClusterAssignment:
    """Soft cluster assignment for GMM-based clustering."""

    memory_id: str
    cluster_probabilities: Dict[str, float]  # cluster_id -> probability

    def primary_cluster(self) -> str:
        """Return cluster with highest probability."""
        return max(self.cluster_probabilities, key=self.cluster_probabilities.get)

    def is_multi_topic(self, threshold: float = 0.3) -> bool:
        """Check if memory belongs to multiple topics above threshold."""
        above_threshold = [
            p for p in self.cluster_probabilities.values() if p >= threshold
        ]
        return len(above_threshold) > 1


@dataclass
class ClusteringResult:
    """Result of clustering all memories."""

    clusters: List[TopicCluster]
    unclustered: List[str]  # Memory IDs not assigned to any cluster
    silhouette_score: float  # Overall clustering quality [-1, 1]
    davies_bouldin_score: float  # Cluster separation (lower is better)
    calinski_harabasz_score: float  # Cluster dispersion (higher is better)
    method: ClusteringMethod
    parameters: Dict[str, object]  # Algorithm-specific parameters

    # Optional for GMM soft clustering
    soft_assignments: Optional[List[SoftClusterAssignment]] = None


@dataclass
class AlgorithmComparisonResult:
    """Result of comparing multiple clustering algorithms."""

    results: Dict[str, ClusteringResult]  # method_name -> result
    best_method: str
    comparison_summary: str


@dataclass
class OptimalKResult:
    """Result of optimal K discovery."""

    k_values: List[int]
    silhouette_scores: List[float]
    optimal_k: int
    elbow_k: Optional[int]  # Elbow point if detected
    analysis: str


@dataclass
class TopicNamingResult:
    """Result of topic naming quality evaluation."""

    cluster_id: str
    names_by_approach: Dict[TopicNamingApproach, str]
    descriptions_by_approach: Dict[TopicNamingApproach, str]
    best_approach: TopicNamingApproach
    evaluation_notes: str


@dataclass
class ClusterCoherenceReview:
    """Manual review data for cluster coherence."""

    cluster_id: str
    cluster_name: str
    sample_memories: List[str]  # Sample memory contents
    coherence_rating: Optional[int] = None  # 1-5 scale, filled during review
    outliers_identified: List[str] = field(default_factory=list)
    notes: str = ""


# Pydantic models for JSON serialization
class TopicClusterSchema(BaseModel):
    """Pydantic schema for TopicCluster serialization."""

    id: str
    name: str
    description: str
    memory_ids: List[str]
    centroid: List[float]
    coherence_score: float
    keywords: List[str]


class ClusteringResultSchema(BaseModel):
    """Pydantic schema for ClusteringResult serialization."""

    clusters: List[TopicClusterSchema]
    unclustered: List[str]
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float
    method: str
    parameters: Dict[str, object]
    soft_assignments: Optional[List[Dict[str, object]]] = None


class ExperimentResultSchema(BaseModel):
    """Pydantic schema for overall experiment results."""

    experiment_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    conversation_prefix: str
    total_memories: int
    memories_with_embeddings: int
    results: Dict[str, object]  # Experiment-specific results
