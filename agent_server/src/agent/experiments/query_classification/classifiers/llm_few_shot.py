"""LLM-based few-shot query classifier with examples."""

import logging
import time

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from ..models import (
    ClassificationResult,
    LLMClassificationResponse,
    QueryType,
)

logger = logging.getLogger(__name__)


# Two examples per query type
FEW_SHOT_EXAMPLES = {
    QueryType.CURRENT_STATE: [
        ("What is David wearing?", "Asking for current state of appearance"),
        ("Where does Sarah work?", "Asking for current employment attribute"),
    ],
    QueryType.HISTORY: [
        ("What has David worn this week?", "Asking about changes over time"),
        ("Remember when we talked about cooking?", "Episodic memory of past conversation"),
    ],
    QueryType.ENTITY_OVERVIEW: [
        ("What do you know about Sarah?", "Requesting all info about entity"),
        ("Tell me about my dog", "Entity overview request"),
    ],
    QueryType.TEMPORAL: [
        ("What happened yesterday?", "Time-bounded query with 'yesterday'"),
        ("How was I feeling in December?", "Month-bounded query"),
    ],
    QueryType.CONTINUITY: [
        ("How did the interview go?", "Following up on ongoing situation"),
        ("Any update on that?", "Vague follow-up on recent topic"),
    ],
    QueryType.PROACTIVE_CONTEXT: [
        ("I saw Sarah at the coffee shop", "Statement mentioning entity needing context"),
        ("My mom called me today", "Entity mention requiring background fetch"),
    ],
    QueryType.NO_RETRIEVAL: [
        ("Hello!", "Simple greeting, no retrieval needed"),
        ("What time is it?", "System query, not memory-related"),
    ],
}


def build_few_shot_prompt(query: str) -> str:
    """Build prompt with few-shot examples."""
    examples_section = []

    for query_type, examples in FEW_SHOT_EXAMPLES.items():
        examples_section.append(f"\n{query_type.value}:")
        for ex_query, ex_reasoning in examples:
            examples_section.append(f'  Query: "{ex_query}"')
            examples_section.append(f"  Why: {ex_reasoning}")

    prompt = f"""Classify this query into exactly one of these query types based on the examples.

EXAMPLES:
{"".join(examples_section)}

KEY DISTINCTIONS:
- current_state: Asks for ONE current attribute value ("What IS X?")
- history: Asks about past events or changes over time ("What HAS been?", "Remember when?")
- entity_overview: Asks for ALL info about an entity ("Tell me about X", "Who is X?")
- temporal: Has SPECIFIC time reference (yesterday, last week, in December)
- continuity: Follows up on ONGOING situation ("How did X go?", "Any update?")
- proactive_context: User STATEMENT (not question) mentioning entities
- no_retrieval: Greetings, math, jokes, general help - no memory needed

NOW CLASSIFY:
Query: "{query}"

Respond with the query type, confidence, and brief reasoning."""

    return prompt


class LLMFewShotClassifier:
    """Few-shot LLM classifier with examples for each type."""

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel = SupportedModel.CLAUDE_HAIKU_4_5,
    ):
        self.llm = llm
        self.model = model
        self.name = "llm_few_shot"

    def classify(self, query: str) -> ClassificationResult:
        """Classify a single query."""
        start_time = time.perf_counter()

        prompt = build_few_shot_prompt(query)

        try:
            response = direct_structured_llm_call(
                prompt=prompt,
                response_model=LLMClassificationResponse,
                model=self.model,
                llm=self.llm,
                caller="llm_few_shot_classify",
                temperature=0.0,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Map string to enum
            try:
                query_type = QueryType(response.query_type.lower())
            except ValueError:
                logger.warning(
                    f"Unknown query type: {response.query_type}, defaulting to no_retrieval"
                )
                query_type = QueryType.NO_RETRIEVAL

            return ClassificationResult(
                query=query,
                predicted_type=query_type,
                confidence=response.confidence,
                reasoning=response.reasoning,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Classification failed for query '{query}': {e}")
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ClassificationResult(
                query=query,
                predicted_type=QueryType.NO_RETRIEVAL,
                confidence=0.0,
                reasoning=f"Error: {e}",
                latency_ms=latency_ms,
            )

    def classify_batch(self, queries: list[str]) -> list[ClassificationResult]:
        """Classify multiple queries."""
        return [self.classify(q) for q in queries]
