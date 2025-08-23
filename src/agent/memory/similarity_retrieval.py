"""
Similarity-based memory retrieval using embeddings.
"""

import logging
import time
from typing import List, Optional, Tuple
from datetime import datetime

from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry
from agent.memory.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


def retrieve_memories_by_similarity(
    query_text: str,
    trigger_history: TriggerHistory,
    max_results: int = 5,
    min_similarity: float = 0.1,
    time_filter_start: Optional[datetime] = None,
    time_filter_end: Optional[datetime] = None,
) -> List[TriggerHistoryEntry]:
    """
    Retrieve memories based on semantic similarity to query text.
    
    Args:
        query_text: Text to find similar memories for
        trigger_history: History to search through
        max_results: Maximum number of memories to return
        min_similarity: Minimum similarity score (0-1) to include
        time_filter_start: Optional start time filter
        time_filter_end: Optional end time filter
        
    Returns:
        List of trigger history entries ranked by similarity
    """
    start_time = time.time()
    
    # Get old entries (exclude recent ones that are in stream of consciousness)
    old_entries = trigger_history.get_old_entries()
    total_entries = len(old_entries)
    
    if total_entries == 0:
        elapsed = time.time() - start_time
        logger.info(f"Memory similarity search completed in {elapsed:.3f}s - no old entries to search")
        return []
    
    # Apply time filtering if specified
    candidate_entries = []
    for entry in old_entries:
        # Check time bounds
        if time_filter_start and entry.timestamp < time_filter_start:
            continue
        if time_filter_end and entry.timestamp > time_filter_end:
            continue
        
        # Only include entries that have embeddings
        if entry.embedding_vector:
            candidate_entries.append(entry)
    
    if not candidate_entries:
        elapsed = time.time() - start_time
        logger.info(f"Memory similarity search completed in {elapsed:.3f}s - no entries with embeddings found")
        return []
    
    # Generate embedding for query
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.encode(query_text)
    
    # Calculate similarities
    similarities: List[Tuple[TriggerHistoryEntry, float]] = []
    for entry in candidate_entries:
        similarity = embedding_service.cosine_similarity(query_embedding, entry.embedding_vector)
        
        # Only include if above minimum threshold
        if similarity >= min_similarity:
            similarities.append((entry, similarity))
    
    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Take top results
    results = [entry for entry, score in similarities[:max_results]]
    
    elapsed = time.time() - start_time
    logger.info(
        f"Memory similarity search completed in {elapsed:.3f}s - "
        f"searched {total_entries} entries, "
        f"time-filtered to {len(candidate_entries)}, "
        f"found {len(similarities)} above threshold {min_similarity}, "
        f"returned top {len(results)}"
    )
    
    # Log similarity scores for debugging
    if similarities:
        score_info = ", ".join([f"{score:.3f}" for _, score in similarities[:max_results]])
        logger.debug(f"Top similarity scores: [{score_info}]")
    
    return results


