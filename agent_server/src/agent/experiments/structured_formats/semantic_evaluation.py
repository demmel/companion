"""
Semantic similarity-based evaluation for structured outputs.

Uses embeddings to compare extracted facts semantically rather than exact matching.
"""

from typing import Tuple, Any, Dict, List
from pydantic import BaseModel
import logging

from agent.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


def semantic_evaluate(
    extracted: BaseModel, expected: BaseModel, similarity_threshold: float = 0.6
) -> Tuple[float, float, float]:
    """
    Evaluate extraction quality using semantic similarity.

    Converts both extracted and expected to JSON strings and compares them
    using embeddings. This allows credit for synonyms, paraphrases, and
    semantically equivalent extractions.

    Args:
        extracted: Result from LLM extraction
        expected: Ground truth expected result
        similarity_threshold: Minimum cosine similarity to count as a match (0.0-1.0)

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    # Convert both to JSON strings
    extracted_json = extracted.model_dump_json()
    expected_json = expected.model_dump_json()

    # Get embedding service
    embedding_service = get_embedding_service()

    # Calculate semantic similarity
    try:
        ext_emb = embedding_service.encode(extracted_json)
        exp_emb = embedding_service.encode(expected_json)
        similarity = embedding_service.cosine_similarity(ext_emb, exp_emb)
        similarity = max(0.0, similarity)
    except Exception as e:
        logger.error(f"Failed to compute semantic similarity: {e}")
        similarity = 0.0

    # Use similarity as the score
    # If similarity >= threshold, treat as mostly correct
    if similarity >= similarity_threshold:
        # High similarity - good match
        precision = similarity
        recall = similarity
    else:
        # Low similarity - poor match
        precision = similarity
        recall = similarity

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    logger.debug(
        f"Semantic eval: similarity={similarity:.2f}, threshold={similarity_threshold}, "
        f"P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}"
    )

    return precision, recall, f1
