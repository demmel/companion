"""
Multi-query similarity scoring system for DAG memory retrieval.

Calculates similarity scores across multiple queries and combines them
to rank memories for retrieval.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from agent.memory.embedding_service import get_embedding_service, EmbeddingService

from .models import MemoryElement, MemoryGraph
from .memory_retrieval import MemoryQuery, QueryType

logger = logging.getLogger(__name__)


@dataclass
class SimilarityScore:
    """Similarity score for a memory against a query."""

    memory_id: str
    query_text: str
    query_type: QueryType
    query_importance: float
    raw_similarity: float
    weighted_score: float


@dataclass
class MemoryRetrievalCandidate:
    """A memory candidate with aggregated similarity scores."""

    memory: MemoryElement
    query_scores: List[SimilarityScore]
    combined_score: float
    max_score: float
    mean_score: float


class SimilarityScorer:
    """Handles multi-query similarity scoring for memory retrieval."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize similarity scorer.

        Args:
            embedding_service: Optional embedding service instance.
                               If None, uses global instance.
        """
        self.embedding_service = embedding_service or get_embedding_service()

    def score_memories_against_queries(
        self,
        memories: List[MemoryElement],
        queries: List[MemoryQuery],
        combination_strategy: str = "weighted_max"
    ) -> List[MemoryRetrievalCandidate]:
        """
        Score all memories against all queries and combine results.

        Args:
            memories: List of memory elements to score
            queries: List of queries to score against
            combination_strategy: How to combine multiple query scores
                                 Options: "weighted_max", "weighted_mean", "max", "mean"

        Returns:
            List of memory candidates ranked by combined score (highest first)
        """
        if not memories or not queries:
            return []

        # Generate embeddings for all queries
        query_texts = [q.query_text for q in queries]
        query_embeddings = self.embedding_service.encode_batch(query_texts)

        # Calculate recency bounds for relative weighting
        memory_timestamps = [m.timestamp for m in memories if m.timestamp]
        newest_time = max(memory_timestamps) if memory_timestamps else None
        oldest_time = min(memory_timestamps) if memory_timestamps else None

        # Score each memory against each query
        candidates = []

        for memory in memories:
            if not memory.embedding_vector:
                logger.warning(f"Memory {memory.id} has no embedding vector, skipping")
                continue

            query_scores = []

            for query, query_embedding in zip(queries, query_embeddings):
                # Calculate similarity
                raw_similarity = self.embedding_service.cosine_similarity(
                    memory.embedding_vector, query_embedding
                )

                # Weight by query importance
                weighted_score = raw_similarity * query.importance

                query_scores.append(SimilarityScore(
                    memory_id=memory.id,
                    query_text=query.query_text,
                    query_type=query.query_type,
                    query_importance=query.importance,
                    raw_similarity=raw_similarity,
                    weighted_score=weighted_score
                ))

            # Combine scores using specified strategy
            combined_score = self._combine_scores(query_scores, combination_strategy)

            # Apply recency weighting - more recent memories get higher scores
            recency_weight = self._calculate_recency_weight(memory, newest_time, oldest_time)
            combined_score *= recency_weight

            max_score = max(score.weighted_score for score in query_scores)
            mean_score = sum(score.weighted_score for score in query_scores) / len(query_scores)

            candidates.append(MemoryRetrievalCandidate(
                memory=memory,
                query_scores=query_scores,
                combined_score=combined_score,
                max_score=max_score,
                mean_score=mean_score
            ))

        # Sort by combined score (highest first)
        candidates.sort(key=lambda c: c.combined_score, reverse=True)

        logger.info(
            f"Scored {len(candidates)} memories against {len(queries)} queries. "
            f"Top score: {candidates[0].combined_score:.3f}" if candidates else "No candidates scored"
        )

        # Log top candidates for debugging
        if candidates:
            logger.info("Top scoring memory candidates:")
            for i, candidate in enumerate(candidates[:3], 1):
                preview = candidate.memory.content[:60] + "..." if len(candidate.memory.content) > 60 else candidate.memory.content
                logger.info(f"  {i}. Score: {candidate.combined_score:.3f} | {preview}")

                # Log individual query scores for top candidate
                if i == 1:
                    logger.debug(f"Top candidate query scores:")
                    for score in candidate.query_scores:
                        logger.debug(f"    {score.query_type}: {score.weighted_score:.3f} ({score.query_text[:40]}...)")
        else:
            logger.warning("No memory candidates found!")

        return candidates

    def _combine_scores(
        self,
        scores: List[SimilarityScore],
        strategy: str
    ) -> float:
        """
        Combine multiple query scores into a single score.

        Args:
            scores: List of similarity scores from different queries
            strategy: Combination strategy

        Returns:
            Combined score
        """
        if not scores:
            return 0.0

        if strategy == "weighted_max":
            # Take the maximum weighted score
            return max(score.weighted_score for score in scores)

        elif strategy == "weighted_mean":
            # Weighted average of all scores
            total_weight = sum(score.query_importance for score in scores)
            if total_weight == 0:
                return 0.0
            weighted_sum = sum(score.raw_similarity * score.query_importance for score in scores)
            return weighted_sum / total_weight

        elif strategy == "max":
            # Simple maximum raw similarity
            return max(score.raw_similarity for score in scores)

        elif strategy == "mean":
            # Simple mean raw similarity
            return sum(score.raw_similarity for score in scores) / len(scores)

        else:
            logger.warning(f"Unknown combination strategy: {strategy}, using weighted_max")
            return max(score.weighted_score for score in scores)

    def _calculate_recency_weight(self, memory: MemoryElement, newest_time, oldest_time) -> float:
        """
        Calculate recency weight for a memory relative to the memory set being scored.

        The newest memory in the set gets weight 1.0, oldest gets weight 0.5,
        others are linearly interpolated.

        Args:
            memory: Memory element to calculate recency weight for
            newest_time: Timestamp of the newest memory in the set
            oldest_time: Timestamp of the oldest memory in the set

        Returns:
            Weight multiplier between 0.5 and 1.0
        """
        # Handle edge cases
        if not memory.timestamp or not newest_time or not oldest_time:
            return 1.0  # No penalty if timestamps are missing

        if newest_time == oldest_time:
            return 1.0  # All memories have same timestamp

        # Linear interpolation between 0.5 (oldest) and 1.0 (newest)
        time_range = (newest_time - oldest_time).total_seconds()
        memory_age = (newest_time - memory.timestamp).total_seconds()

        # age_ratio: 0.0 for newest, 1.0 for oldest
        age_ratio = memory_age / time_range

        # weight: 1.0 for newest (age_ratio=0), 0.5 for oldest (age_ratio=1)
        recency_weight = 1.0 - 0.5 * age_ratio

        return recency_weight


def retrieve_top_candidates(
    memory_graph: MemoryGraph,
    queries: List[MemoryQuery],
    top_k: int = 10,
    min_similarity_threshold: float = 0.3,
    combination_strategy: str = "weighted_max"
) -> List[MemoryRetrievalCandidate]:
    """
    Retrieve top memory candidates based on multi-query similarity scoring.

    Args:
        memory_graph: Complete memory graph to search
        queries: List of queries to score against
        top_k: Maximum number of candidates to return
        min_similarity_threshold: Minimum combined score to include
        combination_strategy: How to combine multiple query scores

    Returns:
        List of top memory candidates, ranked by similarity
    """
    scorer = SimilarityScorer()

    # Get all memories from the graph
    all_memories = list(memory_graph.elements.values())

    # Score all memories
    candidates = scorer.score_memories_against_queries(
        memories=all_memories,
        queries=queries,
        combination_strategy=combination_strategy
    )

    # Filter by threshold and limit to top_k
    filtered_candidates = [
        c for c in candidates
        if c.combined_score >= min_similarity_threshold
    ][:top_k]

    logger.info(
        f"Retrieved {len(filtered_candidates)} memory candidates "
        f"(from {len(candidates)} total, threshold: {min_similarity_threshold})"
    )

    return filtered_candidates


def ensure_memory_has_embedding(memory: MemoryElement) -> MemoryElement:
    """
    Ensure a memory element has an embedding vector, generating one if needed.

    Args:
        memory: Memory element to check/update

    Returns:
        Memory element with embedding vector
    """
    if memory.embedding_vector is None:
        embedding_service = get_embedding_service()

        # Create embedding text from content and evidence
        embedding_text = f"{memory.content}\n{memory.evidence}"
        memory.embedding_vector = embedding_service.encode(embedding_text)

        logger.debug(f"Generated embedding for memory {memory.id}")

    return memory