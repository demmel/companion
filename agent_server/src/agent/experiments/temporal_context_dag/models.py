"""
Core Pydantic models for the DAG-based memory system.
"""

from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import uuid
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.experiments.temporal_context_dag.edge_types import GraphEdgeType
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Confidence levels for memory reliability."""

    USER_CONFIRMED = "user_confirmed"
    STRONG_INFERENCE = "strong_inference"
    REASONABLE_ASSUMPTION = "reasonable_assumption"
    SPECULATIVE = "speculative"
    LIKELY_ERROR = "likely_error"
    KNOWN_FALSE = "known_false"


class MemoryElement(BaseModel):
    """Individual memory element containing a specific fact, event, or insight."""

    id: str
    content: str
    evidence: str
    timestamp: datetime
    emotional_significance: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.REASONABLE_ASSUMPTION
    )
    embedding_vector: Optional[List[float]] = None


class MemoryContainer(BaseModel):
    """Container grouping memory elements from a single conversation turn/interaction."""

    trigger: TriggerHistoryEntry
    element_ids: List[str]  # References to elements in graph


class MemoryEdge(BaseModel):
    """Directed edge connecting two memory elements with relationship type."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    edge_type: GraphEdgeType
    created_at: datetime = Field(default_factory=datetime.now)


class MemoryGraph(BaseModel):
    """Main DAG structure managing all memory elements, containers, and edges."""

    elements: Dict[str, MemoryElement] = Field(default_factory=dict)
    containers: Dict[str, MemoryContainer] = Field(default_factory=dict)
    edges: Dict[str, MemoryEdge] = Field(default_factory=dict)


class ContextElement(BaseModel):
    memory: MemoryElement
    tokens: int


class ContextGraph(BaseModel):
    """Subgraph extracted for context generation within token budget."""

    elements: List[ContextElement] = Field(default_factory=list)
    edges: List[MemoryEdge] = Field(default_factory=list)
