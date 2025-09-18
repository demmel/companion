"""
Core Pydantic models for the DAG-based memory system.
"""

from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import uuid
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from pydantic import BaseModel, Field


class MemoryElement(BaseModel):
    """Individual memory element containing a specific fact, event, or insight."""

    id: str
    content: str
    timestamp: datetime
    emotional_significance: float = Field(ge=0.0, le=1.0)
    embedding_vector: Optional[List[float]] = None


class MemoryContainer(BaseModel):
    """Container grouping memory elements from a single conversation turn/interaction."""

    trigger: TriggerHistoryEntry
    element_ids: List[str]  # References to elements in graph


class MemoryEdgeType(str, Enum):
    FOLLOWS = "follows"
    UPDATES = "updates"
    RELATES_TO = "relates_to"
    EXPLAINS = "explains"


class MemoryEdge(BaseModel):
    """Directed edge connecting two memory elements with relationship type."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    edge_type: MemoryEdgeType
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
