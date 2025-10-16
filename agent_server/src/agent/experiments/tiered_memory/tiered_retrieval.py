"""
Multi-tier retrieval system for hierarchical memory access.

Performs semantic similarity search across all 4 tiers and allows the
agent to select appropriate granularity levels for each query.
"""

import logging
from typing import List, Dict, Optional, Set
from pydantic import BaseModel, Field
from enum import Enum

from agent.memory.models import MemoryElement, MemoryGraph
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.embedding_service import get_embedding_service
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from .models import (
    ConversationBoundary,
    SemanticCluster,
    TieredMemoryGraph,
    MemoryTier,
    RetrievalResult,
    TieredRetrievalResults,
)

logger = logging.getLogger(__name__)






def retrieve_from_tier_1(
    query_embedding: List[float],
    memory_graph: MemoryGraph,
    top_k: int = 5,
    min_similarity: float = 0.4,
) -> List[RetrievalResult]:
    """
    Retrieve from tier 1 (atomic memory elements).

    Args:
        query_embedding: Embedding vector for the query
        memory_graph: Memory graph with tier 1 elements
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of RetrievalResult objects
    """
    embedding_service = get_embedding_service()
    results = []

    for memory in memory_graph.elements.values():
        if not memory.embedding_vector:
            continue

        similarity = embedding_service.cosine_similarity(
            query_embedding,
            memory.embedding_vector
        )

        if similarity >= min_similarity:
            results.append(
                RetrievalResult(
                    tier=MemoryTier.ATOMIC,
                    element_id=memory.id,
                    score=similarity,
                    summary=memory.content,
                    drill_down_ids=[],  # Tier 1 has no drill-down
                    metadata={
                        "timestamp": memory.timestamp.isoformat(),
                        "confidence": memory.confidence_level.value,
                        "container_id": memory.container_id,
                    }
                )
            )

    # Sort by score and return top k
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def retrieve_from_tier_2(
    query_embedding: List[float],
    trigger_entries: List[TriggerHistoryEntry],
    top_k: int = 5,
    min_similarity: float = 0.4,
) -> List[RetrievalResult]:
    """
    Retrieve from tier 2 (trigger-response pairs).

    Args:
        query_embedding: Embedding vector for the query
        trigger_entries: List of trigger history entries
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of RetrievalResult objects
    """
    embedding_service = get_embedding_service()
    results = []

    for entry in trigger_entries:
        if not entry.embedding_vector:
            continue

        similarity = embedding_service.cosine_similarity(
            query_embedding,
            entry.embedding_vector
        )

        if similarity >= min_similarity:
            summary = entry.compressed_summary or "No summary available"

            results.append(
                RetrievalResult(
                    tier=MemoryTier.TRIGGER_RESPONSE,
                    element_id=entry.entry_id,
                    score=similarity,
                    summary=summary,
                    drill_down_ids=[],  # Could drill down to tier 1 actions
                    metadata={
                        "timestamp": entry.timestamp.isoformat(),
                        "action_count": len(entry.actions_taken),
                    }
                )
            )

    # Sort by score and return top k
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def retrieve_from_tier_3(
    query_embedding: List[float],
    conversations: List[ConversationBoundary],
    top_k: int = 5,
    min_similarity: float = 0.4,
) -> List[RetrievalResult]:
    """
    Retrieve from tier 3 (conversation boundaries).

    Args:
        query_embedding: Embedding vector for the query
        conversations: List of conversation boundaries
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of RetrievalResult objects
    """
    embedding_service = get_embedding_service()
    results = []

    for conversation in conversations:
        if not conversation.embedding_vector:
            continue

        similarity = embedding_service.cosine_similarity(
            query_embedding,
            conversation.embedding_vector
        )

        if similarity >= min_similarity:
            results.append(
                RetrievalResult(
                    tier=MemoryTier.CONVERSATION,
                    element_id=conversation.id,
                    score=similarity,
                    summary=conversation.summary,
                    drill_down_ids=conversation.trigger_entry_ids,  # Drill down to tier 2
                    metadata={
                        "start_time": conversation.start_timestamp.isoformat(),
                        "end_time": conversation.end_timestamp.isoformat(),
                        "duration_seconds": conversation.duration_seconds,
                        "topic_tags": conversation.topic_tags,
                        "entry_count": len(conversation.trigger_entry_ids),
                    }
                )
            )

    # Sort by score and return top k
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def retrieve_from_tier_4(
    query_embedding: List[float],
    semantic_clusters: List[SemanticCluster],
    top_k: int = 5,
    min_similarity: float = 0.4,
) -> List[RetrievalResult]:
    """
    Retrieve from tier 4 (semantic topic clusters).

    Args:
        query_embedding: Embedding vector for the query
        semantic_clusters: List of semantic clusters
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of RetrievalResult objects
    """
    embedding_service = get_embedding_service()
    results = []

    for cluster in semantic_clusters:
        if not cluster.embedding_vector:
            continue

        similarity = embedding_service.cosine_similarity(
            query_embedding,
            cluster.embedding_vector
        )

        if similarity >= min_similarity:
            # Collect all drill-down IDs
            drill_down_ids = cluster.get_all_element_ids()

            results.append(
                RetrievalResult(
                    tier=MemoryTier.SEMANTIC_CLUSTER,
                    element_id=cluster.id,
                    score=similarity,
                    summary=f"Topic: {cluster.cluster_topic}\n{cluster.summary}",
                    drill_down_ids=drill_down_ids,
                    metadata={
                        "topic": cluster.cluster_topic,
                        "cluster_size": cluster.cluster_size,
                        "created_at": cluster.created_at.isoformat(),
                        "conversation_count": len(cluster.conversation_ids),
                        "trigger_entry_count": len(cluster.trigger_entry_ids),
                        "memory_element_count": len(cluster.memory_element_ids),
                    }
                )
            )

    # Sort by score and return top k
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def retrieve_multi_tier(
    query: str,
    tiered_graph: TieredMemoryGraph,
    memory_graph: MemoryGraph,
    trigger_entries: List[TriggerHistoryEntry],
    min_similarity: float = 0.4,
) -> TieredRetrievalResults:
    """
    Perform multi-tier retrieval across the memory hierarchy.

    Retrieves ALL memories above similarity threshold from all tiers,
    sorted by relevance. Budget-based filtering happens at formatting time.

    Args:
        query: Search query text
        tiered_graph: Tiered memory graph with tier 3 & 4
        memory_graph: Base memory graph with tier 1 & 2
        trigger_entries: List of trigger history entries
        min_similarity: Minimum similarity threshold

    Returns:
        TieredRetrievalResults with all relevant results sorted by similarity score
    """
    logger.info(f"Multi-tier retrieval for query: '{query}'")

    # Generate query embedding
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.encode(query)

    # Retrieve ALL relevant results from all tiers
    all_results = []

    tier1_results = retrieve_from_tier_1(
        query_embedding, memory_graph, top_k=999999, min_similarity=min_similarity
    )
    all_results.extend(tier1_results)
    logger.info(f"Retrieved {len(tier1_results)} results from tier 1")

    tier2_results = retrieve_from_tier_2(
        query_embedding, trigger_entries, top_k=999999, min_similarity=min_similarity
    )
    all_results.extend(tier2_results)
    logger.info(f"Retrieved {len(tier2_results)} results from tier 2")

    conversations = list(tiered_graph.conversations.values())
    tier3_results = retrieve_from_tier_3(
        query_embedding, conversations, top_k=999999, min_similarity=min_similarity
    )
    all_results.extend(tier3_results)
    logger.info(f"Retrieved {len(tier3_results)} results from tier 3")

    clusters = list(tiered_graph.semantic_clusters.values())
    tier4_results = retrieve_from_tier_4(
        query_embedding, clusters, top_k=999999, min_similarity=min_similarity
    )
    all_results.extend(tier4_results)
    logger.info(f"Retrieved {len(tier4_results)} results from tier 4")

    # Sort all results by similarity score (best matches first)
    all_results.sort(key=lambda r: r.score, reverse=True)

    logger.info(
        f"Multi-tier retrieval complete: {len(all_results)} total results"
    )

    tiers_used = [MemoryTier.ATOMIC, MemoryTier.TRIGGER_RESPONSE, MemoryTier.CONVERSATION, MemoryTier.SEMANTIC_CLUSTER]

    return TieredRetrievalResults(
        query=query,
        results=all_results,
        total_results=len(all_results),
        tiers_used=tiers_used,
        retrieval_strategy="similarity_based",
    )


def drill_down_result(
    result: RetrievalResult,
    tiered_graph: TieredMemoryGraph,
    memory_graph: MemoryGraph,
    trigger_entries_dict: Dict[str, TriggerHistoryEntry],
    target_tier: MemoryTier,
) -> List[RetrievalResult]:
    """
    Drill down from a high-level result to lower-tier details.

    Args:
        result: High-level retrieval result
        tiered_graph: Tiered memory graph
        memory_graph: Base memory graph
        trigger_entries_dict: Dict of trigger entries by ID
        target_tier: Target tier to drill down to

    Returns:
        List of RetrievalResult objects at target tier
    """
    logger.info(
        f"Drilling down from {result.tier.value} to {target_tier.value} "
        f"for element {result.element_id[:8]}"
    )

    drill_down_results = []

    # Get the appropriate drill-down IDs based on target tier
    if result.tier == MemoryTier.SEMANTIC_CLUSTER:
        cluster = tiered_graph.semantic_clusters.get(result.element_id)
        if not cluster:
            return []

        if target_tier == MemoryTier.CONVERSATION:
            # Drill down to conversations
            for conv_id in cluster.conversation_ids:
                conv = tiered_graph.conversations.get(conv_id)
                if conv:
                    drill_down_results.append(
                        RetrievalResult(
                            tier=MemoryTier.CONVERSATION,
                            element_id=conv.id,
                            score=result.score * 0.95,  # Slight score decay
                            summary=conv.summary,
                            drill_down_ids=conv.trigger_entry_ids,
                            metadata={
                                "parent_cluster": cluster.cluster_topic,
                                "topic_tags": conv.topic_tags,
                            }
                        )
                    )

        elif target_tier == MemoryTier.TRIGGER_RESPONSE:
            # Drill down to trigger entries
            for entry_id in cluster.trigger_entry_ids:
                entry = trigger_entries_dict.get(entry_id)
                if entry:
                    drill_down_results.append(
                        RetrievalResult(
                            tier=MemoryTier.TRIGGER_RESPONSE,
                            element_id=entry.entry_id,
                            score=result.score * 0.9,
                            summary=entry.compressed_summary or "No summary",
                            drill_down_ids=[],
                            metadata={
                                "parent_cluster": cluster.cluster_topic,
                                "timestamp": entry.timestamp.isoformat(),
                            }
                        )
                    )

    elif result.tier == MemoryTier.CONVERSATION:
        conv = tiered_graph.conversations.get(result.element_id)
        if not conv:
            return []

        if target_tier == MemoryTier.TRIGGER_RESPONSE:
            # Drill down to trigger entries
            for entry_id in conv.trigger_entry_ids:
                entry = trigger_entries_dict.get(entry_id)
                if entry:
                    drill_down_results.append(
                        RetrievalResult(
                            tier=MemoryTier.TRIGGER_RESPONSE,
                            element_id=entry.entry_id,
                            score=result.score * 0.95,
                            summary=entry.compressed_summary or "No summary",
                            drill_down_ids=[],
                            metadata={
                                "parent_conversation": result.element_id,
                                "timestamp": entry.timestamp.isoformat(),
                            }
                        )
                    )

    logger.info(f"Drill-down produced {len(drill_down_results)} results")

    return drill_down_results
