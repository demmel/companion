"""
Memory query value types.

These are pure data containers (pydantic + enum, no other imports), so any layer — including
the action *data* layer — can import them without dragging in the memory implementation or the
action-execution stack. ``agent.memory.memory`` re-exports them for existing call sites.
"""

from enum import Enum

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of memory retrieval queries."""

    FACTUAL = "factual"
    EMOTIONAL = "emotional"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    RELATIONSHIP = "relationship"
    DECISION = "decision"
    PATTERN = "pattern"


class MemoryQuery(BaseModel):
    """A single memory retrieval query."""

    reasoning: str = Field(description="Why this query is relevant for current context")
    query_type: QueryType = Field(description="Type of query for categorization")
    query_text: str = Field(description="The actual search query text")
    importance: float = Field(
        description="Importance weight for this query (0.0-1.0)", ge=0.0, le=1.0
    )


class MemoryQueries(BaseModel):
    """A set of memory retrieval queries."""

    queries: list[MemoryQuery] = Field(description="List of memory retrieval queries")
    max_tokens: int = Field(
        description="Maximum tokens to allocate for memory retrieval"
    )
