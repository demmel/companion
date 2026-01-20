"""Retrieval comparison functions for memory extraction experiment."""

import logging
from dataclasses import dataclass

from agent.embedding_service import EmbeddingService, get_embedding_service

from .models import ExtractionResult, MemorySample, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    """A test query with ground truth."""

    query_text: str
    expected_memory_id: str
    description: str


def search_raw(
    query: str,
    memories: list[MemorySample],
    embedding_service: EmbeddingService | None = None,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Search raw memory content by embedding similarity.

    Args:
        query: The search query
        memories: List of memory samples to search
        embedding_service: Optional embedding service (creates default if None)
        top_k: Number of top results to return

    Returns:
        List of SearchResult sorted by score descending
    """
    if embedding_service is None:
        embedding_service = get_embedding_service()

    # Embed query
    query_embedding = embedding_service.encode(query)

    # Score each memory
    results: list[SearchResult] = []
    for memory in memories:
        memory_embedding = embedding_service.encode(memory.content)
        score = EmbeddingService.cosine_similarity(query_embedding, memory_embedding)

        results.append(
            SearchResult(
                content=memory.content,
                score=score,
                source_id=memory.memory_id,
                source_type="raw",
            )
        )

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)

    return results[:top_k]


def search_extracted(
    query: str,
    extractions: list[ExtractionResult],
    embedding_service: EmbeddingService | None = None,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Search extracted facts by embedding similarity.

    Args:
        query: The search query
        extractions: List of extraction results to search
        embedding_service: Optional embedding service
        top_k: Number of top results to return

    Returns:
        List of SearchResult sorted by score descending
    """
    if embedding_service is None:
        embedding_service = get_embedding_service()

    # Embed query
    query_embedding = embedding_service.encode(query)

    # Score each extracted fact
    results: list[SearchResult] = []
    for extraction in extractions:
        for fact in extraction.facts:
            fact_embedding = embedding_service.encode(fact.content)
            score = EmbeddingService.cosine_similarity(query_embedding, fact_embedding)

            results.append(
                SearchResult(
                    content=fact.content,
                    score=score,
                    source_id=extraction.memory_id,
                    source_type="extracted",
                )
            )

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)

    return results[:top_k]


def search_hybrid(
    query: str,
    memories: list[MemorySample],
    extractions: list[ExtractionResult],
    embedding_service: EmbeddingService | None = None,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Search both raw memories and extracted facts.

    Args:
        query: The search query
        memories: List of memory samples
        extractions: List of extraction results
        embedding_service: Optional embedding service
        top_k: Number of top results to return

    Returns:
        List of SearchResult sorted by score descending
    """
    if embedding_service is None:
        embedding_service = get_embedding_service()

    # Get results from both sources
    raw_results = search_raw(query, memories, embedding_service, top_k=top_k * 2)
    extracted_results = search_extracted(
        query, extractions, embedding_service, top_k=top_k * 2
    )

    # Combine and sort
    all_results = raw_results + extracted_results
    all_results.sort(key=lambda r: r.score, reverse=True)

    return all_results[:top_k]


def compute_mrr(
    results: list[SearchResult],
    expected_memory_id: str,
) -> float:
    """
    Compute Mean Reciprocal Rank for a single query.

    Args:
        results: List of search results (ranked)
        expected_memory_id: The ID of the expected correct memory

    Returns:
        Reciprocal rank (1/rank if found, 0 if not found)
    """
    for rank, result in enumerate(results, start=1):
        if result.source_id == expected_memory_id:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    test_queries: list[TestQuery],
    memories: list[MemorySample],
    extractions: list[ExtractionResult],
    embedding_service: EmbeddingService | None = None,
) -> dict[str, float]:
    """
    Evaluate retrieval performance across test queries.

    Args:
        test_queries: List of test queries with ground truth
        memories: List of memory samples
        extractions: List of extraction results
        embedding_service: Optional embedding service

    Returns:
        Dictionary with MRR for each search approach
    """
    if embedding_service is None:
        embedding_service = get_embedding_service()

    raw_mrrs: list[float] = []
    extracted_mrrs: list[float] = []
    hybrid_mrrs: list[float] = []

    for query in test_queries:
        # Search raw
        raw_results = search_raw(
            query.query_text, memories, embedding_service, top_k=10
        )
        raw_mrr = compute_mrr(raw_results, query.expected_memory_id)
        raw_mrrs.append(raw_mrr)

        # Search extracted
        extracted_results = search_extracted(
            query.query_text, extractions, embedding_service, top_k=10
        )
        extracted_mrr = compute_mrr(extracted_results, query.expected_memory_id)
        extracted_mrrs.append(extracted_mrr)

        # Search hybrid
        hybrid_results = search_hybrid(
            query.query_text, memories, extractions, embedding_service, top_k=10
        )
        hybrid_mrr = compute_mrr(hybrid_results, query.expected_memory_id)
        hybrid_mrrs.append(hybrid_mrr)

        logger.debug(
            f"Query '{query.query_text[:30]}...': raw={raw_mrr:.2f}, extracted={extracted_mrr:.2f}, hybrid={hybrid_mrr:.2f}"
        )

    return {
        "raw_mrr": sum(raw_mrrs) / len(raw_mrrs) if raw_mrrs else 0.0,
        "extracted_mrr": (
            sum(extracted_mrrs) / len(extracted_mrrs) if extracted_mrrs else 0.0
        ),
        "hybrid_mrr": sum(hybrid_mrrs) / len(hybrid_mrrs) if hybrid_mrrs else 0.0,
        "raw_wins": sum(
            1
            for r, e, h in zip(raw_mrrs, extracted_mrrs, hybrid_mrrs)
            if r >= e and r >= h
        ),
        "extracted_wins": sum(
            1
            for r, e, h in zip(raw_mrrs, extracted_mrrs, hybrid_mrrs)
            if e > r and e >= h
        ),
        "hybrid_wins": sum(
            1
            for r, e, h in zip(raw_mrrs, extracted_mrrs, hybrid_mrrs)
            if h > r and h > e
        ),
    }


def print_retrieval_comparison(
    query: str,
    memories: list[MemorySample],
    extractions: list[ExtractionResult],
    expected_memory_id: str | None = None,
    embedding_service: EmbeddingService | None = None,
    top_k: int = 5,
) -> None:
    """Print a side-by-side comparison of retrieval results."""
    if embedding_service is None:
        embedding_service = get_embedding_service()

    print(f"\n=== Retrieval Comparison ===")
    print(f"Query: {query}")
    print()

    raw_results = search_raw(query, memories, embedding_service, top_k)
    extracted_results = search_extracted(query, extractions, embedding_service, top_k)
    hybrid_results = search_hybrid(
        query, memories, extractions, embedding_service, top_k
    )

    print("RAW RESULTS:")
    for i, r in enumerate(raw_results, 1):
        marker = " <-- EXPECTED" if r.source_id == expected_memory_id else ""
        print(f"  {i}. [{r.score:.3f}] {r.content[:80]}...{marker}")

    print("\nEXTRACTED RESULTS:")
    for i, r in enumerate(extracted_results, 1):
        marker = " <-- EXPECTED" if r.source_id == expected_memory_id else ""
        print(f"  {i}. [{r.score:.3f}] {r.content[:80]}...{marker}")

    print("\nHYBRID RESULTS:")
    for i, r in enumerate(hybrid_results, 1):
        marker = " <-- EXPECTED" if r.source_id == expected_memory_id else ""
        print(f"  {i}. [{r.score:.3f}] ({r.source_type}) {r.content[:60]}...{marker}")

    if expected_memory_id:
        raw_mrr = compute_mrr(raw_results, expected_memory_id)
        extracted_mrr = compute_mrr(extracted_results, expected_memory_id)
        hybrid_mrr = compute_mrr(hybrid_results, expected_memory_id)
        print(
            f"\nMRR: raw={raw_mrr:.2f}, extracted={extracted_mrr:.2f}, hybrid={hybrid_mrr:.2f}"
        )
