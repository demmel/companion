"""Data models for retrieval experiments."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of queries that need different retrieval strategies."""

    FACT = "fact"  # "What's my dog's name?"
    STATE = "state"  # "What am I wearing?"
    EPISODIC = "episodic"  # "Remember when we talked about X?"
    RELATIONSHIP = "relationship"  # "Who is Sarah?"
    PATTERN = "pattern"  # "When do I usually feel tired?"
    PROACTIVE = "proactive"  # No explicit query, context-based


@dataclass
class Memory:
    """A single memory in the system."""

    memory_id: str
    content: str
    timestamp: int  # Unix timestamp or sequence number
    entities: list[str] = field(default_factory=list)
    emotional_context: Optional[str] = None


@dataclass
class StateChange:
    """A change in state captured in a memory."""

    attribute: str  # e.g., "appearance", "location", "mood"
    old_value: Optional[str]
    new_value: str
    memory_id: str
    timestamp: int


@dataclass
class TestQuery:
    """A test query with ground truth."""

    query_text: str
    query_type: QueryType
    expected_memory_ids: list[str]  # Can have multiple valid answers
    expected_answer: Optional[str] = None  # For fact/state queries
    notes: str = ""


@dataclass
class TemporalSequence:
    """A sequence of memories with state changes for temporal testing."""

    memories: list[Memory]
    state_changes: list[StateChange]
    test_queries: list[TestQuery]
    description: str


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    query: TestQuery
    retrieved_memory_ids: list[str]  # Ranked by relevance
    scores: list[float]
    correct: bool
    reciprocal_rank: float  # 1/rank of first correct answer, 0 if not found


# Pydantic models for LLM responses


class QueryClassificationResponse(BaseModel):
    """LLM response for query classification."""

    query_type: str = Field(
        description="Type of query: fact, state, episodic, relationship, pattern, or proactive"
    )
    confidence: float = Field(
        description="Confidence in classification from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(description="Brief explanation for the classification")


class StateExtractionResponse(BaseModel):
    """LLM response for extracting state from a memory."""

    has_state_change: bool = Field(
        description="Whether this memory contains a state change"
    )
    attribute: str = Field(
        description="The attribute that changed (e.g., appearance, location, mood)"
    )
    new_value: str = Field(description="The new value of the attribute")
    entities: list[str] = Field(
        description="Entities this state applies to (e.g., 'user', 'Sarah')"
    )
