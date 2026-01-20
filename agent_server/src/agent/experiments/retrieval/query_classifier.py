"""Query classification for routing to appropriate retrieval strategies."""

import logging

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from .models import QueryClassificationResponse, QueryType

logger = logging.getLogger(__name__)


CLASSIFICATION_PROMPT = """Classify this query into one of these types:

QUERY: {query}

Query Types:
1. FACT - Asking about a stable piece of information (e.g., "What's my dog's name?", "Where do I work?")
2. STATE - Asking about current/changing state (e.g., "What am I wearing?", "How am I feeling?", "Where am I right now?")
3. EPISODIC - Asking about a specific past moment or experience (e.g., "Remember when we talked about X?", "What happened at the party?")
4. RELATIONSHIP - Asking about a person or relationship (e.g., "Who is Sarah?", "Tell me about my friend Mark")
5. PATTERN - Asking about habits or patterns over time (e.g., "When do I usually exercise?", "What do I typically eat for breakfast?")
6. PROACTIVE - Not a query, but context that might trigger relevant memory surfacing

Key distinctions:
- FACT vs STATE: Facts are stable (dog's name), states change over time (what you're wearing)
- FACT vs EPISODIC: Facts are timeless ("What's my allergy?"), episodic is about a moment ("When did I find out about my allergy?")
- STATE vs PATTERN: State is current ("How am I feeling?"), pattern is aggregate ("How have I been feeling lately?")

Respond with the query type and your confidence level."""


def classify_query(
    query: str,
    llm: LLM,
    model: SupportedModel,
) -> tuple[QueryType, float]:
    """Classify a query into a QueryType.

    Returns:
        Tuple of (QueryType, confidence)
    """
    prompt = CLASSIFICATION_PROMPT.format(query=query)

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=QueryClassificationResponse,
            model=model,
            llm=llm,
            caller="classify_query",
        )

        # Map string to enum
        query_type = QueryType(response.query_type.lower())
        return query_type, response.confidence

    except ValueError as e:
        logger.warning(f"Unknown query type returned: {e}. Defaulting to FACT.")
        return QueryType.FACT, 0.5
    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        return QueryType.FACT, 0.0


def classify_queries_batch(
    queries: list[str],
    llm: LLM,
    model: SupportedModel,
) -> list[tuple[QueryType, float]]:
    """Classify multiple queries.

    Returns list of (QueryType, confidence) tuples.
    """
    results: list[tuple[QueryType, float]] = []
    for query in queries:
        result = classify_query(query, llm, model)
        results.append(result)
    return results
