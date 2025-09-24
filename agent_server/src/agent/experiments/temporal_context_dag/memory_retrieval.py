"""
Multi-query extraction system for DAG memory retrieval.

Extracts diverse, targeted search queries from current context to enable
sophisticated memory retrieval beyond simple similarity search.
"""

import logging
from typing import List
from enum import Enum

from agent.chain_of_action.trigger import Trigger, format_trigger_for_prompt
from agent.llm import LLM, SupportedModel
from agent.state import State, build_agent_state_description
from agent.structured_llm import direct_structured_llm_call
from pydantic import BaseModel, Field
from agent.chain_of_action.prompts import format_section

from .models import ContextGraph

logger = logging.getLogger(__name__)


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

    query_type: QueryType = Field(description="Type of query for categorization")
    query_text: str = Field(description="The actual search query text")
    reasoning: str = Field(description="Why this query is relevant for current context")
    importance: float = Field(
        description="Importance weight for this query (0.0-1.0)", ge=0.0, le=1.0
    )


class QueryExtractionResult(BaseModel):
    """Result of multi-query extraction."""

    queries: List[MemoryQuery] = Field(
        description="List of diverse memory retrieval queries"
    )
    context_summary: str = Field(
        description="Brief summary of what makes the current context significant for retrieval"
    )


def extract_memory_queries(
    context: ContextGraph,
    state: State,
    llm: LLM,
    model: SupportedModel,
    max_queries: int,
    trigger: Trigger,
) -> QueryExtractionResult:
    """
    Extract diverse memory retrieval queries from current context.

    Uses LLM to analyze the current context and generate targeted queries
    that will help retrieve relevant memories from different perspectives.

    Args:
        context: Current context graph with existing memories
        state: Current agent state
        llm: LLM instance for query generation
        model: Model to use for query extraction
        max_queries: Maximum number of queries to generate

    Returns:
        QueryExtractionResult with diverse queries and context summary
    """
    # Build context description
    context_description = _build_context_description(context)

    # Build prompt for query extraction
    prompt = f"""I'm {state.name}, {state.role}. I need to search my long-term memory for relevant information based on what just happened and my current context.

{build_agent_state_description(state)}

{format_section("MY MEMORIES AND CONTEXT", context_description)}

{format_section("CURRENT SITUATION (WHAT I'M RESPONDING TO RIGHT NOW)", format_trigger_for_prompt(trigger))}

## Task:
Based on this context, I need to generate diverse memory retrieval queries that will help me find relevant information from my past experiences. I should consider different types of information that might be useful:

- **Factual queries**: Specific facts, data, or information that might be relevant
- **Emotional queries**: Past emotional experiences or reactions that might inform current situation
- **Causal queries**: Past causes, consequences, or patterns that might apply now
- **Temporal queries**: Events from specific time periods or sequences that matter
- **Relationship queries**: Information about people, relationships, or social dynamics
- **Decision queries**: Past decisions, strategies, or approaches I've used in similar situations
- **Pattern queries**: Recurring themes, behaviors, or patterns I've observed

I should generate queries that:
1. Are specific enough to find relevant memories
2. Cover different aspects of what might be useful for my current situation
3. Consider both direct relevance and indirect connections
4. Help me avoid repeating past mistakes or build on past successes
5. Are diverse in type and perspective

Generate up to {max_queries} high-quality queries, each with a clear reasoning for why it would be valuable."""

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=QueryExtractionResult,
            model=model,
            llm=llm,
            caller="memory_query_extraction",
        )

        logger.info(
            f"Extracted {len(response.queries)} memory queries from context with "
            f"{len(context.elements)} elements"
        )

        # Log each query for debugging
        for i, query in enumerate(response.queries, 1):
            logger.info(
                f"Query {i}: [{query.query_type}] {query.query_text} (importance: {query.importance:.2f})"
            )
            logger.debug(f"Query {i} reasoning: {query.reasoning}")

        return response

    except Exception as e:
        logger.warning(f"Memory query extraction failed: {e}")
        # Return minimal fallback queries
        return QueryExtractionResult(
            queries=[
                MemoryQuery(
                    query_type=QueryType.FACTUAL,
                    query_text="relevant facts and information",
                    reasoning="Fallback general query for relevant information",
                    importance=0.7,
                )
            ],
            context_summary="Fallback context summary due to extraction failure",
        )


def _build_context_description(context: ContextGraph) -> str:
    """Build a description of the current context for query extraction."""
    from .context_formatting import format_context

    if not context.elements:
        return "No memories currently in active context."

    # Use the proper context formatting system
    return format_context(context)
