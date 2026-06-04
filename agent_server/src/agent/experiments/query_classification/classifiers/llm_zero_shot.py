"""LLM-based zero-shot query classifier."""

import logging
import time

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from ..models import (
    ClassificationResult,
    LLMClassificationResponse,
    QueryType,
    QUERY_TYPE_DESCRIPTIONS,
)

logger = logging.getLogger(__name__)


CLASSIFICATION_PROMPT = """Classify this query into exactly one of these query types:

QUERY: {query}

QUERY TYPES:

1. current_state - {current_state_desc}

2. history - {history_desc}

3. entity_overview - {entity_overview_desc}

4. temporal - {temporal_desc}

5. continuity - {continuity_desc}

6. proactive_context - {proactive_context_desc}

7. no_retrieval - {no_retrieval_desc}

KEY DISTINCTIONS:
- current_state vs history: "What IS X?" (current) vs "What HAS X been?" (history)
- history vs temporal: History is open-ended past, temporal has specific time reference
- entity_overview vs current_state: Overview asks for ALL info, state asks for ONE attribute
- proactive_context: User is making a STATEMENT mentioning entities, not asking a question

Classify the query. Be precise about the distinction between types."""


class LLMZeroShotClassifier:
    """Zero-shot LLM classifier for query types."""

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel = SupportedModel.CLAUDE_HAIKU_4_5,
    ):
        self.llm = llm
        self.model = model
        self.name = "llm_zero_shot"

    def classify(self, query: str) -> ClassificationResult:
        """Classify a single query."""
        start_time = time.perf_counter()

        prompt = CLASSIFICATION_PROMPT.format(
            query=query,
            current_state_desc=QUERY_TYPE_DESCRIPTIONS[QueryType.CURRENT_STATE],
            history_desc=QUERY_TYPE_DESCRIPTIONS[QueryType.HISTORY],
            entity_overview_desc=QUERY_TYPE_DESCRIPTIONS[QueryType.ENTITY_OVERVIEW],
            temporal_desc=QUERY_TYPE_DESCRIPTIONS[QueryType.TEMPORAL],
            continuity_desc=QUERY_TYPE_DESCRIPTIONS[QueryType.CONTINUITY],
            proactive_context_desc=QUERY_TYPE_DESCRIPTIONS[QueryType.PROACTIVE_CONTEXT],
            no_retrieval_desc=QUERY_TYPE_DESCRIPTIONS[QueryType.NO_RETRIEVAL],
        )

        try:
            response = direct_structured_llm_call(
                prompt=prompt,
                response_model=LLMClassificationResponse,
                model=self.model,
                llm=self.llm,
                caller="llm_zero_shot_classify",
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
