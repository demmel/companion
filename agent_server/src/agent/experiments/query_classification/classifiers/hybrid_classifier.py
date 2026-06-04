"""Hybrid classifier combining embedding and LLM approaches."""

import logging
import time

from agent.llm import LLM, SupportedModel

from ..models import ClassificationResult, Dataset, QueryType
from .embedding_classifier import EmbeddingClassifier
from .llm_few_shot import LLMFewShotClassifier

logger = logging.getLogger(__name__)


class HybridClassifier:
    """Hybrid classifier: embedding for high-confidence, LLM fallback for low-confidence."""

    def __init__(
        self,
        embedding_classifier: EmbeddingClassifier,
        llm: LLM,
        model: SupportedModel = SupportedModel.CLAUDE_HAIKU_4_5,
        confidence_threshold: float = 0.8,
    ):
        self.embedding_classifier = embedding_classifier
        self.llm_classifier = LLMFewShotClassifier(llm, model)
        self.confidence_threshold = confidence_threshold
        self.name = f"hybrid_thresh_{confidence_threshold}"

        # Stats tracking
        self.embedding_calls = 0
        self.llm_fallback_calls = 0

    def train(self, dataset: Dataset) -> None:
        """Train the embedding classifier."""
        self.embedding_classifier.train(dataset)

    def classify(self, query: str) -> ClassificationResult:
        """Classify using embedding first, LLM fallback if low confidence."""
        start_time = time.perf_counter()

        # First try embedding classifier
        embedding_result = self.embedding_classifier.classify(query)
        self.embedding_calls += 1

        if embedding_result.confidence >= self.confidence_threshold:
            # High confidence - use embedding result
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ClassificationResult(
                query=query,
                predicted_type=embedding_result.predicted_type,
                confidence=embedding_result.confidence,
                reasoning=f"Embedding classifier (confidence {embedding_result.confidence:.2f} >= threshold)",
                latency_ms=latency_ms,
            )
        else:
            # Low confidence - fall back to LLM
            self.llm_fallback_calls += 1
            llm_result = self.llm_classifier.classify(query)

            latency_ms = (time.perf_counter() - start_time) * 1000
            return ClassificationResult(
                query=query,
                predicted_type=llm_result.predicted_type,
                confidence=llm_result.confidence,
                reasoning=(
                    f"LLM fallback (embedding confidence {embedding_result.confidence:.2f} "
                    f"< threshold {self.confidence_threshold}): {llm_result.reasoning}"
                ),
                latency_ms=latency_ms,
            )

    def classify_batch(self, queries: list[str]) -> list[ClassificationResult]:
        """Classify multiple queries."""
        return [self.classify(q) for q in queries]

    def get_stats(self) -> dict[str, float]:
        """Get classifier usage statistics."""
        total = self.embedding_calls
        if total == 0:
            return {"embedding_ratio": 0.0, "llm_ratio": 0.0, "total_calls": 0}

        embedding_only = total - self.llm_fallback_calls
        return {
            "embedding_only_calls": embedding_only,
            "llm_fallback_calls": self.llm_fallback_calls,
            "total_calls": total,
            "embedding_ratio": embedding_only / total,
            "llm_fallback_ratio": self.llm_fallback_calls / total,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.embedding_calls = 0
        self.llm_fallback_calls = 0

    @property
    def is_trained(self) -> bool:
        """Check if classifier is trained."""
        return self.embedding_classifier.is_trained
