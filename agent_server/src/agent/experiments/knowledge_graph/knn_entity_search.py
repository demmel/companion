#!/usr/bin/env python3
"""
kNN Entity Search for Deduplication

Implements proper k-nearest neighbors search for entity similarity matching,
replacing simple pairwise comparisons with efficient similarity search.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

from agent.experiments.knowledge_graph.knowledge_graph_prototype import GraphNode
from agent.memory.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    """Result of entity similarity search"""

    node_id: str
    node: GraphNode
    similarity: float
    composite_key: str


class KNNEntitySearch:
    """k-nearest neighbors search for entity deduplication"""

    def __init__(self, similarity_threshold: float = 0.7):
        self.embedding_service = get_embedding_service()
        self.similarity_threshold = similarity_threshold

        # Cache for entity embeddings and metadata
        self.entity_embeddings: List[np.ndarray] = []
        self.entity_metadata: List[Dict[str, Any]] = (
            []
        )  # Store node_id, composite_key, node
        self.embedding_index_dirty = True

    def add_entity(self, node: GraphNode, composite_key: str) -> None:
        """Add an entity to the search index"""

        if node.embedding is not None:
            self.entity_embeddings.append(np.array(node.embedding))
            self.entity_metadata.append(
                {
                    "node_id": node.id,
                    "composite_key": composite_key,
                    "node": node,
                    "entity_type": node.node_type.value,
                    "name": node.name,
                }
            )
            self.embedding_index_dirty = True

    def find_similar_entities(
        self,
        query_embedding: np.ndarray,
        entity_type: str,
        k: int = 5,
        type_filter: bool = True,
    ) -> List[EntityMatch]:
        """Find k most similar entities using cosine similarity"""

        if not self.entity_embeddings:
            return []

        # Convert to numpy array for vectorized operations
        embeddings_matrix = np.vstack(self.entity_embeddings)

        # Calculate cosine similarities
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        # Normalize query embedding
        normalized_query = query_embedding / query_norm

        # Normalize all embeddings
        embedding_norms = np.linalg.norm(embeddings_matrix, axis=1)
        # Avoid division by zero
        non_zero_mask = embedding_norms != 0
        if not np.any(non_zero_mask):
            return []

        # Only calculate similarities for non-zero embeddings
        valid_embeddings = embeddings_matrix[non_zero_mask]
        valid_norms = embedding_norms[non_zero_mask]

        normalized_embeddings = valid_embeddings / valid_norms[:, np.newaxis]

        # Calculate cosine similarities (dot product of normalized vectors)
        similarities = np.dot(normalized_embeddings, normalized_query)

        # Map back to original indices
        valid_indices = np.where(non_zero_mask)[0]

        # Filter by entity type if requested
        if type_filter:
            type_mask = np.array(
                [
                    self.entity_metadata[idx]["entity_type"] == entity_type
                    for idx in valid_indices
                ]
            )
            if np.any(type_mask):
                similarities = similarities[type_mask]
                valid_indices = valid_indices[type_mask]

        if len(similarities) == 0:
            return []

        # Get top-k most similar
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Filter by similarity threshold
        results = []
        for idx in top_k_indices:
            similarity = float(similarities[idx])
            if similarity >= self.similarity_threshold:
                original_idx = valid_indices[idx]
                metadata = self.entity_metadata[original_idx]

                results.append(
                    EntityMatch(
                        node_id=metadata["node_id"],
                        node=metadata["node"],
                        similarity=similarity,
                        composite_key=metadata["composite_key"],
                    )
                )

        return results

    def find_best_match(
        self, entity_name: str, entity_type: str, entity_description: str
    ) -> Optional[EntityMatch]:
        """Find the best matching entity for deduplication"""

        # Generate embedding for query entity
        query_text = f"[{entity_type}] {entity_name}: {entity_description}"
        try:
            query_embedding = np.array(self.embedding_service.encode(query_text))
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")
            return None

        # Find similar entities
        matches = self.find_similar_entities(
            query_embedding, entity_type, k=3, type_filter=True  # Get top 3 matches
        )

        if not matches:
            return None

        # Return the best match (highest similarity)
        best_match = matches[0]

        logger.debug(
            f"Found {len(matches)} similar entities for '{entity_name}', "
            f"best match: '{best_match.node.name}' (similarity: {best_match.similarity:.3f})"
        )

        return best_match

    def update_entity_embedding(self, node_id: str, new_embedding: np.ndarray) -> None:
        """Update the embedding for an existing entity"""

        for i, metadata in enumerate(self.entity_metadata):
            if metadata["node_id"] == node_id:
                self.entity_embeddings[i] = new_embedding
                break

    def remove_entity(self, node_id: str) -> None:
        """Remove an entity from the search index"""

        indices_to_remove = []
        for i, metadata in enumerate(self.entity_metadata):
            if metadata["node_id"] == node_id:
                indices_to_remove.append(i)

        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del self.entity_embeddings[i]
            del self.entity_metadata[i]

        self.embedding_index_dirty = True

    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about the entity search index"""

        if not self.entity_metadata:
            return {
                "total_entities": 0,
                "entities_by_type": {},
                "average_embedding_norm": 0.0,
            }

        # Count entities by type
        type_counts = {}
        for metadata in self.entity_metadata:
            entity_type = metadata["entity_type"]
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

        # Calculate average embedding norm
        if self.entity_embeddings:
            norms = [np.linalg.norm(emb) for emb in self.entity_embeddings]
            avg_norm = np.mean(norms)
        else:
            avg_norm = 0.0

        return {
            "total_entities": len(self.entity_metadata),
            "entities_by_type": type_counts,
            "average_embedding_norm": float(avg_norm),
            "similarity_threshold": self.similarity_threshold,
        }

    def clear_index(self) -> None:
        """Clear all entities from the search index"""
        self.entity_embeddings.clear()
        self.entity_metadata.clear()
        self.embedding_index_dirty = True


class KNNEntityDeduplicator:
    """Entity deduplication using kNN search"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.knn_search = KNNEntitySearch(similarity_threshold)
        self.auto_accept_threshold = 0.95
        self.auto_reject_threshold = 0.3

    def add_entity_to_index(self, node: GraphNode, composite_key: str) -> None:
        """Add entity to the kNN search index"""
        self.knn_search.add_entity(node, composite_key)

    def find_duplicate_entity(
        self,
        entity_name: str,
        entity_type: str,
        entity_description: str,
        entity_evidence: str,
    ) -> Optional[str]:
        """Find duplicate entity using kNN search, returns composite key if found"""

        best_match = self.knn_search.find_best_match(
            entity_name, entity_type, entity_description
        )

        if not best_match:
            return None

        # Auto-accept for very high similarity
        if best_match.similarity >= self.auto_accept_threshold:
            logger.info(
                f"Auto-accepting entity match (similarity: {best_match.similarity:.3f}): "
                f"'{entity_name}' -> '{best_match.node.name}'"
            )
            return best_match.composite_key

        # Auto-reject for very low similarity
        if best_match.similarity < self.auto_reject_threshold:
            logger.debug(
                f"Auto-rejecting entity match (similarity: {best_match.similarity:.3f}): "
                f"'{entity_name}' vs '{best_match.node.name}'"
            )
            return None

        # For medium similarities, could add LLM validation here if needed
        # For now, use threshold-based decision
        if best_match.similarity >= self.knn_search.similarity_threshold:
            logger.info(
                f"Accepting entity match (similarity: {best_match.similarity:.3f}): "
                f"'{entity_name}' -> '{best_match.node.name}'"
            )
            return best_match.composite_key

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return self.knn_search.get_entity_statistics()
