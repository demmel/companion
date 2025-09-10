#!/usr/bin/env python3
"""
kNN Entity Search for Deduplication

Implements proper k-nearest neighbors search for entity similarity matching,
replacing simple pairwise comparisons with efficient similarity search.
"""

from asyncio import Protocol
import numpy as np
from typing import Generic, List, Dict, Optional, Any, TypeVar
from dataclasses import dataclass
import logging

from agent.memory.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class IKNNEntity(Protocol):
    def get_id(self) -> str: ...
    def get_embedding(self) -> np.ndarray: ...
    def get_text(self) -> str: ...


T = TypeVar("T", bound=IKNNEntity)


@dataclass
class EntityMatch(Generic[T]):
    t: T
    similarity: float


@dataclass
class EntitySearchStatistics:
    total_entities: int
    average_embedding_norm: float


class KNNEntitySearch(Generic[T]):
    """k-nearest neighbors search for entity deduplication"""

    def __init__(self):
        self.embedding_service = get_embedding_service()

        # Cache for entity embeddings and metadata
        self.entity_metadata: List[T] = []  # Store node_id, composite_key, entity_type
        self.embedding_index_dirty = True

    def add_entity(self, t: T) -> None:
        """Add an entity to the search index"""

        self.entity_metadata.append(t)
        self.embedding_index_dirty = True

    def find_similar_entities(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        similarity_threshold: float = 0.85,
    ) -> List[EntityMatch[T]]:
        """Find k most similar entities using cosine similarity"""

        if not self.entity_metadata:
            return []

        # Convert to numpy array for vectorized operations
        embeddings_matrix = np.vstack([t.get_embedding() for t in self.entity_metadata])

        # Calculate cosine similarities
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        # Normalize query embedding
        normalized_query = query_embedding / query_norm

        # Normalize all embeddings
        embedding_norms = np.linalg.norm(embeddings_matrix, axis=1)
        
        # Embedding diagnostics
        zero_embeddings = np.sum(embedding_norms == 0)
        if zero_embeddings > 0:
            logger.warning(f"Found {zero_embeddings} zero embeddings in entity index")
        
        logger.info(f"🔢 Entity embedding norms: min={embedding_norms.min():.3f}, max={embedding_norms.max():.3f}, mean={embedding_norms.mean():.3f}")
        
        # Avoid division by zero
        non_zero_mask = embedding_norms != 0
        if not np.any(non_zero_mask):
            logger.warning("All entity embeddings are zero, cannot compute similarities")
            return []

        # Only calculate similarities for non-zero embeddings
        valid_embeddings = embeddings_matrix[non_zero_mask]
        valid_norms = embedding_norms[non_zero_mask]

        normalized_embeddings = valid_embeddings / valid_norms[:, np.newaxis]

        # Calculate cosine similarities (dot product of normalized vectors)
        similarities = np.dot(normalized_embeddings, normalized_query)

        # Similarity diagnostics
        if len(similarities) > 0:
            logger.info(f"🔢 Similarity scores: min={similarities.min():.3f}, max={similarities.max():.3f}, mean={similarities.mean():.3f}")
            
            # Show top 3 similarities for debugging
            sorted_sims = np.sort(similarities)[::-1]
            top_3 = sorted_sims[:min(3, len(sorted_sims))]
            logger.info(f"🔢 Top 3 similarities: {[f'{s:.3f}' for s in top_3]}")

        # Map back to original indices
        valid_indices = np.where(non_zero_mask)[0]

        if len(similarities) == 0:
            return []

        # Get top-k most similar
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Filter by similarity threshold and create EntityMatch objects
        results = []
        for idx in top_k_indices:
            similarity = float(similarities[idx])
            if similarity >= similarity_threshold:
                original_idx = valid_indices[idx]
                metadata = self.entity_metadata[original_idx]
                results.append(EntityMatch(t=metadata, similarity=similarity))

        return results

    def find_best_match(
        self,
        query_text: str,
    ) -> Optional[EntityMatch[T]]:
        """Find the best matching entity for deduplication"""

        try:
            query_embedding = np.array(self.embedding_service.encode(query_text))
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")
            return None
        
        # Embedding quality diagnostics
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            logger.warning(f"Generated zero embedding for query: '{query_text}'")
            return None
        
        logger.info(f"🔢 Query embedding norm: {query_norm:.3f} for '{query_text[:50]}...'")
        
        # Check for empty index
        if not self.entity_metadata:
            logger.info(f"🔍 Empty entity index, cannot find matches")
            return None

        # Find similar entities
        matches = self.find_similar_entities(
            query_embedding,
            k=3,
        )

        if not matches:
            return None

        # Return the best match (highest similarity)
        best_match = matches[0]

        logger.debug(
            f"Found {len(matches)} similar entities for '{query_text}', "
            f"best match: '{best_match.t.get_id()}' (similarity: {best_match.similarity:.3f})"
        )

        return best_match

    def find_duplicate_entity(
        self,
        query_text: str,
        similarity_threshold: float = 0.85,
        auto_accept_threshold: float = 0.95,
        auto_reject_threshold: float = 0.3,
    ) -> Optional[str]:
        """Find duplicate entity using kNN search, returns composite key if found"""

        best_match = self.find_best_match(query_text)

        if not best_match:
            logger.info(f"🔍 No entities found in index for query: '{query_text}' (index size: {len(self.entity_metadata)})")
            return None
        
        # Always log the best match found for diagnostics
        logger.info(f"🔍 Best match for '{query_text}': similarity={best_match.similarity:.3f}, entity='{best_match.t.get_text()}'")

        # Auto-accept for very high similarity
        if best_match.similarity >= auto_accept_threshold:
            logger.info(
                f"Auto-accepting entity match (similarity: {best_match.similarity:.3f}): "
                f"'{query_text}' -> '{best_match.t.get_text()}'"
            )
            return best_match.t.get_id()

        # Auto-reject for very low similarity
        if best_match.similarity < auto_reject_threshold:
            logger.info(
                f"Auto-rejecting entity match (similarity: {best_match.similarity:.3f}): "
                f"'{query_text}' vs '{best_match.t.get_text()}'"
            )
            return None

        # For medium similarities, could add LLM validation here if needed
        # For now, use threshold-based decision
        if best_match.similarity >= similarity_threshold:
            logger.info(
                f"Accepting entity match (similarity: {best_match.similarity:.3f}): "
                f"'{query_text}' -> '{best_match.t.get_text()}'"
            )
            return best_match.t.get_id()

        # Log when we reject a potentially good match
        logger.info(
            f"🚫 Rejecting entity match below threshold (similarity: {best_match.similarity:.3f} < {similarity_threshold}): "
            f"'{query_text}' vs '{best_match.t.get_text()}'"
        )
        return None

    def remove_entity(self, id: str) -> None:
        """Remove an entity from the search index"""

        indices_to_remove = []
        for i, metadata in enumerate(self.entity_metadata):
            if metadata.get_id() == id:
                indices_to_remove.append(i)

        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del self.entity_metadata[i]

        self.embedding_index_dirty = True

    def get_entity_statistics(self) -> EntitySearchStatistics:
        """Get statistics about the entity search index"""

        if not self.entity_metadata:
            return EntitySearchStatistics(
                total_entities=0,
                average_embedding_norm=0.0,
            )

        # Calculate average embedding norm
        if self.entity_metadata:
            norms = [np.linalg.norm(e.get_embedding()) for e in self.entity_metadata]
            avg_norm = np.mean(norms)
        else:
            avg_norm = 0.0

        return EntitySearchStatistics(
            total_entities=len(self.entity_metadata),
            average_embedding_norm=float(avg_norm),
        )

    def clear_index(self) -> None:
        """Clear all entities from the search index"""
        self.entity_metadata.clear()
        self.embedding_index_dirty = True
