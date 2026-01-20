"""Temporal retrieval strategies for state queries.

The key insight: for state queries like "What is she wearing?", the correct
answer is the MOST RECENT state, not the most semantically similar memory.

This module implements and compares different temporal retrieval strategies.
"""

import logging
from dataclasses import dataclass

import numpy as np

from agent.embedding_service import EmbeddingService

from .models import Memory, RetrievalResult, TestQuery

logger = logging.getLogger(__name__)


@dataclass
class ScoredMemory:
    """A memory with a retrieval score."""

    memory: Memory
    similarity_score: float
    recency_score: float
    combined_score: float


def compute_similarity_scores(
    query: str,
    memories: list[Memory],
    embedding_service: EmbeddingService,
) -> list[float]:
    """Compute embedding similarity between query and each memory."""
    query_embedding = np.array(embedding_service.encode(query))
    memory_embeddings = [
        np.array(embedding_service.encode(m.content)) for m in memories
    ]

    # Cosine similarity (embeddings are normalized by sentence-transformers)
    scores: list[float] = []
    for mem_emb in memory_embeddings:
        similarity = float(np.dot(query_embedding, mem_emb))
        scores.append(similarity)

    return scores


def compute_recency_scores(memories: list[Memory]) -> list[float]:
    """Compute recency scores (higher = more recent).

    Uses normalized ranking where most recent = 1.0, oldest = 0.0
    """
    if not memories:
        return []

    # Sort by timestamp to get ranking
    sorted_by_time = sorted(memories, key=lambda m: m.timestamp)
    timestamp_to_rank = {m.timestamp: i for i, m in enumerate(sorted_by_time)}

    max_rank = len(memories) - 1
    if max_rank == 0:
        return [1.0]

    scores: list[float] = []
    for memory in memories:
        rank = timestamp_to_rank[memory.timestamp]
        recency = rank / max_rank  # 0.0 = oldest, 1.0 = newest
        scores.append(recency)

    return scores


# ============================================================================
# Strategy A: Naive Similarity (baseline)
# ============================================================================


def retrieve_by_similarity(
    query: str,
    memories: list[Memory],
    embedding_service: EmbeddingService,
    top_k: int = 3,
) -> list[ScoredMemory]:
    """Baseline: retrieve by embedding similarity only."""
    similarity_scores = compute_similarity_scores(query, memories, embedding_service)

    scored = [
        ScoredMemory(
            memory=mem,
            similarity_score=sim,
            recency_score=0.0,
            combined_score=sim,
        )
        for mem, sim in zip(memories, similarity_scores)
    ]

    scored.sort(key=lambda x: x.combined_score, reverse=True)
    return scored[:top_k]


# ============================================================================
# Strategy B: Recency-Weighted Similarity
# ============================================================================


def retrieve_by_recency_weighted(
    query: str,
    memories: list[Memory],
    embedding_service: EmbeddingService,
    top_k: int = 3,
    recency_weight: float = 0.3,
) -> list[ScoredMemory]:
    """Combine similarity with recency weighting.

    combined = (1 - recency_weight) * similarity + recency_weight * recency

    Args:
        recency_weight: How much to weight recency (0.0 = pure similarity, 1.0 = pure recency)
    """
    similarity_scores = compute_similarity_scores(query, memories, embedding_service)
    recency_scores = compute_recency_scores(memories)

    scored = [
        ScoredMemory(
            memory=mem,
            similarity_score=sim,
            recency_score=rec,
            combined_score=(1 - recency_weight) * sim + recency_weight * rec,
        )
        for mem, sim, rec in zip(memories, similarity_scores, recency_scores)
    ]

    scored.sort(key=lambda x: x.combined_score, reverse=True)
    return scored[:top_k]


# ============================================================================
# Strategy C: Most Recent Among Similar
# ============================================================================


def retrieve_most_recent_similar(
    query: str,
    memories: list[Memory],
    embedding_service: EmbeddingService,
    top_k: int = 3,
    similarity_threshold: float = 0.5,
) -> list[ScoredMemory]:
    """First filter by similarity threshold, then rank by recency.

    This ensures we only return relevant memories, but among relevant ones,
    we prefer the most recent.
    """
    similarity_scores = compute_similarity_scores(query, memories, embedding_service)
    recency_scores = compute_recency_scores(memories)

    # Filter by similarity threshold
    scored = [
        ScoredMemory(
            memory=mem,
            similarity_score=sim,
            recency_score=rec,
            combined_score=rec if sim >= similarity_threshold else -1.0,
        )
        for mem, sim, rec in zip(memories, similarity_scores, recency_scores)
    ]

    # Filter out below-threshold memories
    scored = [s for s in scored if s.combined_score >= 0]

    if not scored:
        # Fall back to pure similarity if nothing passes threshold
        return retrieve_by_similarity(query, memories, embedding_service, top_k)

    # Sort by recency (combined_score = recency for passing memories)
    scored.sort(key=lambda x: x.combined_score, reverse=True)
    return scored[:top_k]


# ============================================================================
# Strategy D: Attribute-Based State Tracking
# ============================================================================


def retrieve_by_state_tracking(
    query: str,
    memories: list[Memory],
    embedding_service: EmbeddingService,
    state_attribute: str,
    top_k: int = 3,
) -> list[ScoredMemory]:
    """For known state attributes, return most recent update.

    This requires pre-extracted state changes. If the query is about a known
    state attribute (e.g., appearance, location, mood), we return the most
    recent memory that updated that state.

    For now, this is a simplified version that:
    1. Finds memories that match the attribute semantically
    2. Returns the most recent one
    """
    similarity_scores = compute_similarity_scores(query, memories, embedding_service)

    # Find memories likely about this attribute (above threshold)
    threshold = 0.4
    relevant_memories = [
        (mem, sim) for mem, sim in zip(memories, similarity_scores) if sim >= threshold
    ]

    if not relevant_memories:
        return retrieve_by_similarity(query, memories, embedding_service, top_k)

    # Sort by timestamp (most recent first)
    relevant_memories.sort(key=lambda x: x[0].timestamp, reverse=True)

    recency_scores = compute_recency_scores([m for m, _ in relevant_memories])

    scored = [
        ScoredMemory(
            memory=mem,
            similarity_score=sim,
            recency_score=rec,
            combined_score=rec,  # Pure recency for state queries
        )
        for (mem, sim), rec in zip(relevant_memories, recency_scores)
    ]

    return scored[:top_k]


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_retrieval(
    query: TestQuery,
    retrieved: list[ScoredMemory],
) -> RetrievalResult:
    """Evaluate retrieval results against ground truth."""
    retrieved_ids = [s.memory.memory_id for s in retrieved]
    scores = [s.combined_score for s in retrieved]

    # Find rank of first correct answer
    reciprocal_rank = 0.0
    for i, mem_id in enumerate(retrieved_ids):
        if mem_id in query.expected_memory_ids:
            reciprocal_rank = 1.0 / (i + 1)
            break

    correct = len(set(retrieved_ids) & set(query.expected_memory_ids)) > 0

    return RetrievalResult(
        query=query,
        retrieved_memory_ids=retrieved_ids,
        scores=scores,
        correct=correct,
        reciprocal_rank=reciprocal_rank,
    )


def compare_strategies(
    query: TestQuery,
    memories: list[Memory],
    embedding_service: EmbeddingService,
) -> dict[str, RetrievalResult]:
    """Compare all strategies on a single query."""
    strategies = {
        "similarity": retrieve_by_similarity(
            query.query_text, memories, embedding_service
        ),
        "recency_weighted_0.3": retrieve_by_recency_weighted(
            query.query_text, memories, embedding_service, recency_weight=0.3
        ),
        "recency_weighted_0.5": retrieve_by_recency_weighted(
            query.query_text, memories, embedding_service, recency_weight=0.5
        ),
        "most_recent_similar": retrieve_most_recent_similar(
            query.query_text, memories, embedding_service
        ),
    }

    return {
        name: evaluate_retrieval(query, retrieved)
        for name, retrieved in strategies.items()
    }
