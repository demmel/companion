"""
Clustering implementations using sklearn.

Each function takes memory embeddings and returns ClusteringResult.
"""

import logging
import re
import uuid
from collections import Counter
from typing import List, Tuple, Optional, Dict
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, spectral_clustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import scipy.sparse as sp

from agent.memory.dag.models import MemoryElement
from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.storage import ITriggerHistory
from agent.chain_of_action.trigger import UserInputTrigger, WakeupTrigger, BirthTrigger
from agent.chain_of_action.action.action_types import ActionType

from .models import (
    TopicCluster,
    ClusteringResult,
    ClusteringMethod,
    SoftClusterAssignment,
    OptimalKResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Action-Type Parsing and Residual Projection (v2 additions)
# =============================================================================


def parse_action_type(content: str) -> str:
    """
    Extract action type from memory content prefix.

    Patterns recognized:
    - "[?] I thought about..." -> "thought"
    - "[?] I responded to..." -> "responded"
    - "[?] I updated my appearance..." -> "updated_appearance"
    - "[?] My mood changed..." -> "mood_changed"
    - "[?] I update_environment..." -> "updated_environment"
    - "[?] I add_priority..." -> "add_priority"
    - "[?] I remove_priority..." -> "remove_priority"
    - "David said to me:..." -> "user_message"
    - "I continue to exist..." -> "existence"
    - Otherwise -> "other"
    """
    content_lower = content.lower()

    # Check for specific patterns
    if content.startswith("[?] I thought about"):
        return "thought"
    if content.startswith("[?] I responded to"):
        return "responded"
    if content.startswith("[?] I updated my appearance"):
        return "updated_appearance"
    if content.startswith("[?] My mood changed"):
        return "mood_changed"
    if content.startswith("[?] I update_environment"):
        return "updated_environment"
    if content.startswith("[?] I add_priority"):
        return "add_priority"
    if content.startswith("[?] I remove_priority"):
        return "remove_priority"
    if content.startswith("David said to me"):
        return "user_message"
    if "continue to exist" in content_lower:
        return "existence"
    if content.startswith("[?]"):
        # Try to extract action from [?] I <action>
        match = re.match(r"\[\?\] I (\w+)", content)
        if match:
            return match.group(1).lower()
        return "unknown_action"

    return "other"


def build_action_type_mapping(
    trigger_history: ITriggerHistory,
) -> Dict[Tuple[str, int], str]:
    """
    Build a mapping from (container_id, sequence_in_container) to action type.

    For sequence_in_container == 0: This is the trigger itself
        - UserMessageTrigger -> "user_message"
        - ExistenceTrigger -> "existence"
        - Other triggers -> "trigger_other"

    For sequence_in_container >= 1: This is an action
        - Look up actions_taken[sequence - 1].type.value

    Returns:
        Dict mapping (entry_id, sequence) -> action_type string
    """
    mapping: Dict[Tuple[str, int], str] = {}

    for entry in trigger_history.iter_entries(reverse=False, start=0):
        entry_id = entry.entry_id

        # Map sequence 0 to trigger type
        if isinstance(entry.trigger, UserInputTrigger):
            mapping[(entry_id, 0)] = "user_message"
        elif isinstance(entry.trigger, WakeupTrigger):
            mapping[(entry_id, 0)] = "existence"
        elif isinstance(entry.trigger, BirthTrigger):
            mapping[(entry_id, 0)] = "birth"
        else:
            mapping[(entry_id, 0)] = "trigger_other"

        # Map sequences 1+ to action types
        for idx, action in enumerate(entry.actions_taken):
            sequence = idx + 1
            mapping[(entry_id, sequence)] = action.type.value

    return mapping


def get_action_types_from_trigger_history(
    memories: List[MemoryElement],
    trigger_history: ITriggerHistory,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Get action types for memories using the actual action types from trigger history.

    This is more accurate than parsing from text content.

    Args:
        memories: List of MemoryElement with container_id and sequence_in_container
        trigger_history: ITriggerHistory containing the actual action data

    Returns:
        action_types: List of action type strings (same order as memories)
        type_counts: Dict mapping action type to count
    """
    mapping = build_action_type_mapping(trigger_history)

    action_types: List[str] = []
    for m in memories:
        key = (m.container_id, m.sequence_in_container)
        action_type = mapping.get(key, "unknown")
        if action_type == "unknown":
            # Fall back to text parsing if not found in mapping
            action_type = parse_action_type(m.content)
        action_types.append(action_type)

    type_counts = Counter(action_types)

    logger.info(
        f"Found {len(type_counts)} unique action types from trigger history: {dict(type_counts)}"
    )

    return action_types, dict(type_counts)


def get_action_types_for_memories(
    memories: List[MemoryElement],
) -> Tuple[List[str], Dict[str, int]]:
    """
    Parse action types for all memories from text content.

    Note: This is the legacy text-parsing approach. Prefer get_action_types_from_trigger_history
    when trigger_history is available.

    Returns:
        action_types: List of action type strings (same order as memories)
        type_counts: Dict mapping action type to count
    """
    action_types = [parse_action_type(m.content) for m in memories]
    type_counts = Counter(action_types)

    logger.info(f"Found {len(type_counts)} unique action types: {dict(type_counts)}")

    return action_types, dict(type_counts)


def compute_action_type_centroids(
    embeddings: np.ndarray,
    action_types: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute centroid embedding for each action type.

    Args:
        embeddings: (N, D) array of embeddings
        action_types: List of N action type strings

    Returns:
        Dict mapping action type -> centroid embedding (D,)
    """
    centroids: Dict[str, np.ndarray] = {}
    unique_types = set(action_types)

    for action_type in unique_types:
        indices = [i for i, t in enumerate(action_types) if t == action_type]
        if indices:
            type_embeddings = embeddings[indices]
            centroid = np.mean(type_embeddings, axis=0)
            centroids[action_type] = centroid
            logger.debug(
                f"Action type '{action_type}': {len(indices)} memories, "
                f"centroid norm={np.linalg.norm(centroid):.4f}"
            )

    logger.info(f"Computed centroids for {len(centroids)} action types")

    return centroids


def project_orthogonal_to_action_types(
    embeddings: np.ndarray,
    centroids: Dict[str, np.ndarray],
    n_components_to_remove: int = 5,
) -> np.ndarray:
    """
    Project embeddings into space orthogonal to action-type signal.

    Uses SVD to identify the principal directions of the action-type centroids,
    then projects all embeddings orthogonal to those directions.

    Args:
        embeddings: (N, D) array of embeddings
        centroids: Dict mapping action type -> centroid embedding
        n_components_to_remove: Number of action-type principal components to remove

    Returns:
        (N, D) array of residual embeddings
    """
    # Stack centroids into matrix
    centroid_matrix = np.array(list(centroids.values()))  # (num_types, D)

    # Center the centroids
    centroid_mean = np.mean(centroid_matrix, axis=0)
    centered_centroids = centroid_matrix - centroid_mean

    # SVD to find principal directions of action-type variance
    # U: (num_types, k), S: (k,), Vt: (k, D)
    n_components = min(
        n_components_to_remove, len(centroids) - 1, centroid_matrix.shape[1]
    )
    if n_components < 1:
        logger.warning(
            "Not enough action types for projection, returning original embeddings"
        )
        return embeddings

    U, S, Vt = np.linalg.svd(centered_centroids, full_matrices=False)

    # Take top n_components directions to remove
    action_type_directions = Vt[:n_components]  # (n_components, D)

    # Project embeddings orthogonal to these directions
    # For each embedding, subtract its projection onto the action-type subspace
    residual_embeddings = embeddings.copy()

    for direction in action_type_directions:
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        # Project each embedding onto this direction and subtract
        projections = np.outer(embeddings @ direction, direction)
        residual_embeddings = residual_embeddings - projections

    # Log variance explained by removed components
    total_variance = np.sum(S**2)
    removed_variance = np.sum(S[:n_components] ** 2)
    variance_ratio = removed_variance / total_variance if total_variance > 0 else 0

    logger.info(
        f"Removed {n_components} action-type components, "
        f"explaining {variance_ratio*100:.1f}% of centroid variance"
    )

    return residual_embeddings


def calculate_action_type_entropy(
    cluster_memory_ids: List[str],
    memory_id_to_action_type: Dict[str, str],
) -> float:
    """
    Calculate Shannon entropy of action types within a cluster.

    Higher entropy = more diverse action types = better (not clustering by action type).
    Normalized to [0, 1] range.

    Args:
        cluster_memory_ids: List of memory IDs in the cluster
        memory_id_to_action_type: Dict mapping memory ID -> action type

    Returns:
        Normalized entropy (0 = all same type, 1 = uniformly distributed)
    """
    action_types = [
        memory_id_to_action_type.get(mid, "unknown") for mid in cluster_memory_ids
    ]

    if not action_types:
        return 0.0

    counts = Counter(action_types)
    total = len(action_types)
    num_types = len(counts)

    if num_types <= 1:
        return 0.0

    # Calculate Shannon entropy
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    # Normalize by max possible entropy (uniform distribution)
    max_entropy = np.log2(num_types)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return float(normalized_entropy)


def cluster_with_residual_embeddings(
    memories: List[MemoryElement],
    k: int,
    n_components_to_remove: int = 5,
    random_state: int = 42,
) -> Tuple[ClusteringResult, Dict[str, float]]:
    """
    Cluster using action-type residual embeddings.

    Returns both the clustering result and action-type entropy per cluster.
    """
    embeddings, memory_ids, valid_memories = prepare_embeddings(memories)

    # Get action types
    action_types, type_counts = get_action_types_for_memories(valid_memories)

    # Build memory_id -> action_type mapping
    memory_id_to_action_type = {
        mid: atype for mid, atype in zip(memory_ids, action_types)
    }

    # Compute action-type centroids
    centroids = compute_action_type_centroids(embeddings, action_types)

    # Project to residual space
    residual_embeddings = project_orthogonal_to_action_types(
        embeddings, centroids, n_components_to_remove
    )

    # Cluster in residual space
    logger.info(f"Running K-Means on residual embeddings with k={k}")
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(residual_embeddings)

    # Build clusters (using residual embeddings for centroids)
    clusters, unclustered = labels_to_clusters(labels, memory_ids, residual_embeddings)

    # Calculate metrics on residual embeddings
    sil_score = (
        silhouette_score(residual_embeddings, labels) if len(set(labels)) > 1 else 0.0
    )
    db_score = (
        davies_bouldin_score(residual_embeddings, labels)
        if len(set(labels)) > 1
        else float("inf")
    )
    ch_score = (
        calinski_harabasz_score(residual_embeddings, labels)
        if len(set(labels)) > 1
        else 0.0
    )

    # Calculate action-type entropy for each cluster
    cluster_entropies: Dict[str, float] = {}
    for cluster in clusters:
        entropy = calculate_action_type_entropy(
            cluster.memory_ids, memory_id_to_action_type
        )
        cluster_entropies[cluster.id] = entropy

    avg_entropy = np.mean(list(cluster_entropies.values()))
    logger.info(
        f"Residual clustering: silhouette={sil_score:.4f}, "
        f"avg action-type entropy={avg_entropy:.4f}"
    )

    result = ClusteringResult(
        clusters=clusters,
        unclustered=unclustered,
        silhouette_score=sil_score,
        davies_bouldin_score=db_score,
        calinski_harabasz_score=ch_score,
        method=ClusteringMethod.KMEANS,
        parameters={
            "k": k,
            "random_state": random_state,
            "residual_projection": True,
            "n_components_removed": n_components_to_remove,
        },
    )

    return result, cluster_entropies


# =============================================================================
# Cross-Action-Type Graph Clustering (v3 additions)
# =============================================================================


def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 15,
    metric: str = "cosine",
) -> sp.csr_matrix:
    """
    Build a KNN similarity graph from embeddings.

    Args:
        embeddings: (N, D) array of embeddings
        k: Number of nearest neighbors
        metric: Distance metric ('cosine' or 'euclidean')

    Returns:
        Sparse adjacency matrix (N, N) with similarity weights.
        Self-loops are excluded. Matrix is symmetrized.
    """
    n_samples = embeddings.shape[0]

    # Find k+1 nearest neighbors (including self)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_samples), metric=metric)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Build sparse adjacency matrix
    # For cosine distance, similarity = 1 - distance
    # Exclude self-connections (first neighbor is self with distance 0)
    rows = []
    cols = []
    data = []

    for i in range(n_samples):
        for j_idx in range(1, len(indices[i])):  # Skip self (index 0)
            j = indices[i][j_idx]
            dist = distances[i][j_idx]

            # Convert distance to similarity
            if metric == "cosine":
                similarity = 1.0 - dist
            else:
                # For euclidean, use inverse distance
                similarity = 1.0 / (1.0 + dist)

            rows.append(i)
            cols.append(j)
            data.append(max(0.0, similarity))  # Ensure non-negative

    # Create sparse matrix
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

    # Symmetrize: A = (A + A.T) / 2
    adjacency = (adjacency + adjacency.T) / 2

    logger.info(f"Built KNN graph: {n_samples} nodes, {adjacency.nnz} edges, k={k}")

    return adjacency


def apply_cross_action_type_weighting(
    adjacency: sp.csr_matrix,
    action_types: List[str],
    same_type_weight: float = 0.1,
) -> sp.csr_matrix:
    """
    Downweight edges between memories of the same action type.

    This forces clustering to prefer cross-action-type connections,
    encouraging topics that span multiple action types.

    Args:
        adjacency: KNN graph adjacency matrix (N, N)
        action_types: Action type for each memory (length N)
        same_type_weight: Multiplier for same-type edges (0.1 = 10% strength)

    Returns:
        Modified adjacency matrix with downweighted same-type edges
    """
    # Convert to lil_matrix for efficient modification
    modified = adjacency.tolil()

    n_samples = adjacency.shape[0]
    same_type_edges = 0
    cross_type_edges = 0

    # Iterate over non-zero entries
    rows, cols = adjacency.nonzero()
    for i, j in zip(rows, cols):
        if i < j:  # Only process upper triangle (matrix is symmetric)
            if action_types[i] == action_types[j]:
                # Same action type - downweight
                modified[i, j] *= same_type_weight
                modified[j, i] *= same_type_weight
                same_type_edges += 1
            else:
                cross_type_edges += 1

    logger.info(
        f"Applied cross-action-type weighting: "
        f"{same_type_edges} same-type edges (weight={same_type_weight}), "
        f"{cross_type_edges} cross-type edges (weight=1.0)"
    )

    return modified.tocsr()


def cluster_cross_action_type(
    memories: List[MemoryElement],
    n_clusters: int = 12,
    k_neighbors: int = 15,
    same_type_weight: float = 0.1,
    random_state: int = 42,
    trigger_history: Optional[ITriggerHistory] = None,
) -> Tuple[ClusteringResult, Dict[str, float], Dict[str, Dict[str, int]]]:
    """
    Cluster memories using cross-action-type graph structure.

    This method builds a KNN graph, downweights same-action-type edges,
    and applies spectral clustering. The result encourages topics that
    span multiple action types rather than being dominated by one type.

    Args:
        memories: List of MemoryElement objects
        n_clusters: Number of clusters to find
        k_neighbors: Number of nearest neighbors for KNN graph
        same_type_weight: Multiplier for same-action-type edges (0.1 = 10%)
        random_state: Random seed for reproducibility
        trigger_history: Optional ITriggerHistory for accurate action type lookup

    Returns:
        result: ClusteringResult with cluster assignments
        cluster_entropies: Dict mapping cluster_id -> action-type entropy
        cluster_action_distributions: Dict mapping cluster_id -> {action_type: count}
    """
    embeddings, memory_ids, valid_memories = prepare_embeddings(memories)

    # Get action types for all memories - use trigger history if available
    if trigger_history is not None:
        action_types, type_counts = get_action_types_from_trigger_history(
            valid_memories, trigger_history
        )
    else:
        action_types, type_counts = get_action_types_for_memories(valid_memories)

    # Build memory_id -> action_type mapping
    memory_id_to_action_type = {
        mid: atype for mid, atype in zip(memory_ids, action_types)
    }

    logger.info(
        f"Running cross-action-type clustering: n_clusters={n_clusters}, "
        f"k={k_neighbors}, same_type_weight={same_type_weight}"
    )

    # Build KNN graph
    adjacency = build_knn_graph(embeddings, k=k_neighbors, metric="cosine")

    # Apply cross-action-type weighting
    filtered_adjacency = apply_cross_action_type_weighting(
        adjacency, action_types, same_type_weight
    )

    # Spectral clustering on the modified graph
    # spectral_clustering expects an affinity matrix
    labels = spectral_clustering(
        filtered_adjacency,
        n_clusters=n_clusters,
        random_state=random_state,
        assign_labels="kmeans",
    )

    # Build clusters
    clusters, unclustered = labels_to_clusters(labels, memory_ids, embeddings)

    # Calculate metrics on original embeddings
    sil_score = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0.0
    db_score = (
        davies_bouldin_score(embeddings, labels)
        if len(set(labels)) > 1
        else float("inf")
    )
    ch_score = (
        calinski_harabasz_score(embeddings, labels) if len(set(labels)) > 1 else 0.0
    )

    # Calculate action-type entropy and distribution for each cluster
    cluster_entropies: Dict[str, float] = {}
    cluster_action_distributions: Dict[str, Dict[str, int]] = {}

    for cluster in clusters:
        entropy = calculate_action_type_entropy(
            cluster.memory_ids, memory_id_to_action_type
        )
        cluster_entropies[cluster.id] = entropy

        # Count action types in this cluster
        action_counts: Dict[str, int] = {}
        for mid in cluster.memory_ids:
            atype = memory_id_to_action_type.get(mid, "unknown")
            action_counts[atype] = action_counts.get(atype, 0) + 1
        cluster_action_distributions[cluster.id] = action_counts

    avg_entropy = np.mean(list(cluster_entropies.values()))
    num_types_per_cluster = [
        len(dist) for dist in cluster_action_distributions.values()
    ]
    avg_types_per_cluster = np.mean(num_types_per_cluster)

    logger.info(
        f"Cross-action-type clustering: silhouette={sil_score:.4f}, "
        f"avg entropy={avg_entropy:.4f}, avg action types per cluster={avg_types_per_cluster:.1f}"
    )

    result = ClusteringResult(
        clusters=clusters,
        unclustered=unclustered,
        silhouette_score=sil_score,
        davies_bouldin_score=db_score,
        calinski_harabasz_score=ch_score,
        method=ClusteringMethod.KMEANS,  # Using spectral, but no enum for it
        parameters={
            "n_clusters": n_clusters,
            "k_neighbors": k_neighbors,
            "same_type_weight": same_type_weight,
            "random_state": random_state,
            "method": "cross_action_type_spectral",
        },
    )

    return result, cluster_entropies, cluster_action_distributions


# =============================================================================
# Cross-Action-Type Only Clustering (v4)
# =============================================================================


def build_cross_action_affinity_matrix(
    embeddings: np.ndarray,
    action_types: List[str],
    similarity_threshold: float = 0.0,
) -> sp.csr_matrix:
    """
    Build affinity matrix where only cross-action-type pairs have non-zero affinity.

    For all pairs (i, j):
    - If action_type[i] != action_type[j]: affinity = cosine_similarity(i, j)
    - If action_type[i] == action_type[j]: affinity = 0

    Args:
        embeddings: (N, D) array of embeddings
        action_types: Action type for each memory (length N)
        similarity_threshold: Minimum similarity to include an edge (default 0.0)

    Returns:
        Sparse affinity matrix (N, N) with only cross-action-type edges
    """
    n_samples = embeddings.shape[0]

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms

    # Build sparse matrix with only cross-action-type edges
    rows = []
    cols = []
    data = []

    # Group indices by action type for efficient processing
    type_to_indices: Dict[str, List[int]] = {}
    for i, atype in enumerate(action_types):
        if atype not in type_to_indices:
            type_to_indices[atype] = []
        type_to_indices[atype].append(i)

    unique_types = list(type_to_indices.keys())
    logger.info(
        f"Building cross-action affinity matrix: {n_samples} memories, {len(unique_types)} action types"
    )

    # For each pair of different action types, compute similarities
    total_edges = 0
    for t1_idx, type1 in enumerate(unique_types):
        indices1 = type_to_indices[type1]
        emb1 = normalized[indices1]

        for type2 in unique_types[t1_idx + 1 :]:  # Only pairs where type1 < type2
            indices2 = type_to_indices[type2]
            emb2 = normalized[indices2]

            # Compute cosine similarity between all pairs
            similarities = emb1 @ emb2.T  # (len1, len2)

            # Add edges above threshold
            for i_local, i_global in enumerate(indices1):
                for j_local, j_global in enumerate(indices2):
                    sim = similarities[i_local, j_local]
                    if sim > similarity_threshold:
                        # Add both directions for symmetric matrix
                        rows.extend([i_global, j_global])
                        cols.extend([j_global, i_global])
                        data.extend([sim, sim])
                        total_edges += 1

    # Create sparse matrix
    affinity = sp.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

    logger.info(
        f"Cross-action affinity matrix: {total_edges} edges (symmetric: {affinity.nnz})"
    )

    return affinity


def build_cross_action_knn_graph(
    embeddings: np.ndarray,
    action_types: List[str],
    k: int = 15,
) -> sp.csr_matrix:
    """
    Build KNN graph where each node connects to K nearest neighbors
    from OTHER action types only.

    This creates a sparse graph with controlled density, allowing Louvain
    to find more granular community structure.

    Args:
        embeddings: (N, D) array of embeddings
        action_types: Action type for each memory (length N)
        k: Number of nearest neighbors per node

    Returns:
        Sparse affinity matrix (N, N) with KNN edges across action types
    """
    n_samples = embeddings.shape[0]

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Group indices by action type
    type_to_indices: Dict[str, List[int]] = {}
    for i, atype in enumerate(action_types):
        if atype not in type_to_indices:
            type_to_indices[atype] = []
        type_to_indices[atype].append(i)

    logger.info(
        f"Building cross-action KNN graph: {n_samples} memories, {len(type_to_indices)} action types, k={k}"
    )

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    # For each memory, find K nearest neighbors from OTHER action types
    for i in range(n_samples):
        my_type = action_types[i]

        # Collect all indices of different action types
        other_indices: List[int] = []
        for atype, indices in type_to_indices.items():
            if atype != my_type:
                other_indices.extend(indices)

        if len(other_indices) == 0:
            continue

        # Compute similarities to all other-type memories
        other_indices_arr = np.array(other_indices)
        other_embeddings = normalized[other_indices_arr]
        similarities = normalized[i] @ other_embeddings.T

        # Get top K (or all if fewer than K)
        actual_k = min(k, len(other_indices))
        top_k_local = np.argsort(similarities)[-actual_k:]

        for local_idx in top_k_local:
            global_idx = other_indices[local_idx]
            sim = similarities[local_idx]
            if sim > 0:
                rows.append(i)
                cols.append(global_idx)
                data.append(float(sim))

    # Build sparse matrix
    affinity = sp.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

    # Symmetrize: A = (A + A.T) / 2
    affinity = (affinity + affinity.T) / 2

    logger.info(
        f"Cross-action KNN graph: {len(data)} directed edges, {affinity.nnz} after symmetrization"
    )

    return affinity


def cluster_cross_action_only(
    memories: List[MemoryElement],
    trigger_history: ITriggerHistory,
    method: str = "louvain",
    n_clusters: int = 12,
    k_neighbors: int = 15,
    resolution: float = 1.0,
) -> Tuple[ClusteringResult, Dict[str, float], Dict[str, Dict[str, int]]]:
    """
    Cluster memories using only cross-action-type similarity.

    This method ensures that clustering is based purely on semantic similarity
    across different action types, preventing action-type structure from
    dominating the clusters.

    Uses KNN graph where each memory connects to K nearest neighbors from
    OTHER action types only. This creates a sparse graph with controlled
    density for better community detection.

    Args:
        memories: List of MemoryElement objects
        trigger_history: ITriggerHistory for action type lookup
        method: Clustering method - "louvain" or "spectral"
        n_clusters: Number of clusters (only used for spectral)
        k_neighbors: Number of cross-action-type nearest neighbors per memory
        resolution: Resolution parameter for Louvain (higher = more clusters)

    Returns:
        result: ClusteringResult with cluster assignments
        cluster_entropies: Dict mapping cluster_id -> action-type entropy
        cluster_action_distributions: Dict mapping cluster_id -> {action_type: count}
    """
    embeddings, memory_ids, valid_memories = prepare_embeddings(memories)

    # Get action types from trigger history
    action_types, type_counts = get_action_types_from_trigger_history(
        valid_memories, trigger_history
    )

    logger.info(f"Found {len(type_counts)} unique action types: {type_counts}")

    # Build cross-action-type KNN graph
    affinity = build_cross_action_knn_graph(embeddings, action_types, k=k_neighbors)

    # Check connectivity
    n_components = sp.csgraph.connected_components(affinity, directed=False)[0]
    logger.info(f"Affinity graph has {n_components} connected components")

    # Apply clustering
    if method == "louvain":
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities

            # Convert sparse matrix to networkx graph
            G = nx.from_scipy_sparse_array(affinity)

            # Run Louvain community detection
            communities = louvain_communities(G, weight="weight", resolution=resolution)

            # Convert to labels
            labels = np.full(len(memory_ids), -1, dtype=int)
            for cluster_idx, community in enumerate(communities):
                for node in community:
                    labels[node] = cluster_idx

            n_found = len(communities)
            logger.info(f"Louvain found {n_found} communities")

        except ImportError:
            logger.warning(
                "networkx not available, falling back to spectral clustering"
            )
            method = "spectral"

    if method == "spectral":
        # Use spectral clustering on the affinity matrix
        labels = spectral_clustering(
            affinity,
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
            assign_labels="kmeans",
        )
        logger.info(f"Spectral clustering created {n_clusters} clusters")

    # Build memory_id -> action_type mapping
    memory_id_to_action_type = {
        mid: atype for mid, atype in zip(memory_ids, action_types)
    }

    # Build clusters
    clusters, unclustered = labels_to_clusters(labels, memory_ids, embeddings)

    # Calculate metrics
    valid_labels = labels[labels >= 0]
    valid_embeddings = embeddings[labels >= 0]

    if len(set(valid_labels)) > 1 and len(valid_labels) > 0:
        sil_score = silhouette_score(valid_embeddings, valid_labels)
        db_score = davies_bouldin_score(valid_embeddings, valid_labels)
        ch_score = calinski_harabasz_score(valid_embeddings, valid_labels)
    else:
        sil_score = 0.0
        db_score = float("inf")
        ch_score = 0.0

    # Calculate action-type entropy and distribution for each cluster
    cluster_entropies: Dict[str, float] = {}
    cluster_action_distributions: Dict[str, Dict[str, int]] = {}

    for cluster in clusters:
        entropy = calculate_action_type_entropy(
            cluster.memory_ids, memory_id_to_action_type
        )
        cluster_entropies[cluster.id] = entropy

        # Count action types in this cluster
        action_counts: Dict[str, int] = {}
        for mid in cluster.memory_ids:
            atype = memory_id_to_action_type.get(mid, "unknown")
            action_counts[atype] = action_counts.get(atype, 0) + 1
        cluster_action_distributions[cluster.id] = action_counts

    # Log summary
    avg_entropy = (
        np.mean(list(cluster_entropies.values())) if cluster_entropies else 0.0
    )
    avg_types_per_cluster = (
        np.mean([len(dist) for dist in cluster_action_distributions.values()])
        if cluster_action_distributions
        else 0.0
    )

    logger.info(
        f"Cross-action-only clustering ({method}): "
        f"silhouette={sil_score:.4f}, avg entropy={avg_entropy:.4f}, "
        f"avg action types per cluster={avg_types_per_cluster:.1f}"
    )

    result = ClusteringResult(
        clusters=clusters,
        unclustered=unclustered,
        silhouette_score=sil_score,
        davies_bouldin_score=db_score,
        calinski_harabasz_score=ch_score,
        method=ClusteringMethod.KMEANS,  # No enum for this method
        parameters={
            "method": f"cross_action_only_{method}",
            "n_clusters": n_clusters if method == "spectral" else len(clusters),
            "k_neighbors": k_neighbors,
            "resolution": resolution if method == "louvain" else None,
        },
    )

    return result, cluster_entropies, cluster_action_distributions


# =============================================================================
# Original Functions
# =============================================================================


def prepare_embeddings(
    memories: List[MemoryElement],
) -> Tuple[np.ndarray, List[str], List[MemoryElement]]:
    """
    Extract embeddings from memories, filtering those without embeddings.

    Returns:
        embeddings: numpy array of shape (n_samples, n_features)
        memory_ids: list of memory IDs corresponding to rows
        valid_memories: list of MemoryElement objects with embeddings
    """
    valid_memories = [m for m in memories if m.embedding_vector is not None]

    if len(valid_memories) < len(memories):
        logger.warning(
            f"Filtered {len(memories) - len(valid_memories)} memories without embeddings"
        )

    if not valid_memories:
        raise ValueError("No memories with embeddings found")

    embeddings = np.array([m.embedding_vector for m in valid_memories])
    memory_ids = [m.id for m in valid_memories]

    logger.info(
        f"Prepared {len(embeddings)} embeddings of dimension {embeddings.shape[1]}"
    )

    return embeddings, memory_ids, valid_memories


def calculate_cluster_centroid(
    embeddings: np.ndarray, indices: np.ndarray
) -> List[float]:
    """Calculate centroid of cluster from embedding indices."""
    cluster_embeddings = embeddings[indices]
    centroid = np.mean(cluster_embeddings, axis=0)
    return centroid.tolist()


def calculate_intra_cluster_distance(
    embeddings: np.ndarray, labels: np.ndarray, cluster_id: int
) -> float:
    """
    Calculate average distance within a cluster (coherence).

    Lower distance = tighter cluster = higher coherence.
    Returns coherence as 1 - normalized_distance.
    """
    indices = np.where(labels == cluster_id)[0]
    if len(indices) < 2:
        return 1.0  # Single element is perfectly coherent

    cluster_embeddings = embeddings[indices]
    centroid = np.mean(cluster_embeddings, axis=0)

    # Calculate average cosine distance to centroid
    # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
    # Cosine distance: 1 - cosine_similarity
    distances = []
    for emb in cluster_embeddings:
        norm_emb = np.linalg.norm(emb)
        norm_centroid = np.linalg.norm(centroid)
        if norm_emb > 0 and norm_centroid > 0:
            similarity = np.dot(emb, centroid) / (norm_emb * norm_centroid)
            distances.append(1 - similarity)
        else:
            distances.append(1.0)

    avg_distance = np.mean(distances)
    # Convert to coherence score (higher is better)
    coherence = 1 - avg_distance
    return max(0.0, min(1.0, coherence))


def labels_to_clusters(
    labels: np.ndarray,
    memory_ids: List[str],
    embeddings: np.ndarray,
) -> Tuple[List[TopicCluster], List[str]]:
    """
    Convert sklearn cluster labels to TopicCluster objects.

    Returns:
        clusters: List of TopicCluster objects (without LLM-generated names yet)
        unclustered: List of memory IDs not assigned to any cluster (label=-1)
    """
    clusters = []
    unclustered = []

    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            # Unclustered points (from DBSCAN)
            indices = np.where(labels == label)[0]
            unclustered.extend([memory_ids[i] for i in indices])
            continue

        indices = np.where(labels == label)[0]
        cluster_memory_ids = [memory_ids[i] for i in indices]

        centroid = calculate_cluster_centroid(embeddings, indices)
        coherence = calculate_intra_cluster_distance(embeddings, labels, label)

        cluster = TopicCluster(
            id=str(uuid.uuid4()),
            name="",  # Will be filled by topic naming
            description="",  # Will be filled by topic naming
            memory_ids=cluster_memory_ids,
            centroid=centroid,
            coherence_score=coherence,
            keywords=[],  # Will be filled by topic naming
        )
        clusters.append(cluster)

    # Sort clusters by size (largest first)
    clusters.sort(key=lambda c: len(c.memory_ids), reverse=True)

    logger.info(
        f"Created {len(clusters)} clusters, {len(unclustered)} unclustered memories"
    )

    return clusters, unclustered


def cluster_kmeans(
    memories: List[MemoryElement], k: int, random_state: int = 42
) -> ClusteringResult:
    """
    Cluster memories using K-Means algorithm.

    Args:
        memories: List of MemoryElement objects
        k: Number of clusters
        random_state: Random seed for reproducibility
    """
    embeddings, memory_ids, valid_memories = prepare_embeddings(memories)

    if len(embeddings) < k:
        raise ValueError(f"Not enough memories ({len(embeddings)}) for {k} clusters")

    logger.info(f"Running K-Means with k={k}")

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters, unclustered = labels_to_clusters(labels, memory_ids, embeddings)

    # Calculate metrics
    sil_score = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0.0
    db_score = (
        davies_bouldin_score(embeddings, labels)
        if len(set(labels)) > 1
        else float("inf")
    )
    ch_score = (
        calinski_harabasz_score(embeddings, labels) if len(set(labels)) > 1 else 0.0
    )

    logger.info(
        f"K-Means results: silhouette={sil_score:.4f}, "
        f"davies_bouldin={db_score:.4f}, calinski_harabasz={ch_score:.4f}"
    )

    return ClusteringResult(
        clusters=clusters,
        unclustered=unclustered,
        silhouette_score=sil_score,
        davies_bouldin_score=db_score,
        calinski_harabasz_score=ch_score,
        method=ClusteringMethod.KMEANS,
        parameters={"k": k, "random_state": random_state},
    )


def cluster_hierarchical(
    memories: List[MemoryElement],
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    linkage: str = "ward",
) -> ClusteringResult:
    """
    Cluster memories using agglomerative hierarchical clustering.

    Args:
        memories: List of MemoryElement objects
        n_clusters: Number of clusters (mutually exclusive with distance_threshold)
        distance_threshold: Distance threshold for clustering
        linkage: Linkage method ('ward', 'complete', 'average', 'single')
    """
    embeddings, memory_ids, valid_memories = prepare_embeddings(memories)

    if n_clusters is None and distance_threshold is None:
        raise ValueError("Must specify either n_clusters or distance_threshold")

    logger.info(
        f"Running Hierarchical clustering with n_clusters={n_clusters}, "
        f"distance_threshold={distance_threshold}, linkage={linkage}"
    )

    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage=linkage,
    )
    labels = agg.fit_predict(embeddings)

    clusters, unclustered = labels_to_clusters(labels, memory_ids, embeddings)

    # Calculate metrics
    n_labels = len(set(labels))
    sil_score = silhouette_score(embeddings, labels) if n_labels > 1 else 0.0
    db_score = (
        davies_bouldin_score(embeddings, labels) if n_labels > 1 else float("inf")
    )
    ch_score = calinski_harabasz_score(embeddings, labels) if n_labels > 1 else 0.0

    logger.info(
        f"Hierarchical results: silhouette={sil_score:.4f}, "
        f"davies_bouldin={db_score:.4f}, calinski_harabasz={ch_score:.4f}"
    )

    return ClusteringResult(
        clusters=clusters,
        unclustered=unclustered,
        silhouette_score=sil_score,
        davies_bouldin_score=db_score,
        calinski_harabasz_score=ch_score,
        method=ClusteringMethod.HIERARCHICAL,
        parameters={
            "n_clusters": n_clusters,
            "distance_threshold": distance_threshold,
            "linkage": linkage,
        },
    )


def cluster_dbscan(
    memories: List[MemoryElement], eps: float = 0.5, min_samples: int = 3
) -> ClusteringResult:
    """
    Cluster memories using DBSCAN density-based clustering.

    Args:
        memories: List of MemoryElement objects
        eps: Maximum distance between samples in same neighborhood
        min_samples: Minimum samples in neighborhood to form core point
    """
    embeddings, memory_ids, valid_memories = prepare_embeddings(memories)

    logger.info(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(embeddings)

    clusters, unclustered = labels_to_clusters(labels, memory_ids, embeddings)

    # Calculate metrics (excluding noise points)
    non_noise_mask = labels != -1
    if np.sum(non_noise_mask) > 1 and len(set(labels[non_noise_mask])) > 1:
        sil_score = silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask])
        db_score = davies_bouldin_score(
            embeddings[non_noise_mask], labels[non_noise_mask]
        )
        ch_score = calinski_harabasz_score(
            embeddings[non_noise_mask], labels[non_noise_mask]
        )
    else:
        sil_score = 0.0
        db_score = float("inf")
        ch_score = 0.0

    logger.info(
        f"DBSCAN results: {len(clusters)} clusters found, "
        f"{len(unclustered)} noise points, silhouette={sil_score:.4f}"
    )

    return ClusteringResult(
        clusters=clusters,
        unclustered=unclustered,
        silhouette_score=sil_score,
        davies_bouldin_score=db_score,
        calinski_harabasz_score=ch_score,
        method=ClusteringMethod.DBSCAN,
        parameters={"eps": eps, "min_samples": min_samples},
    )


def cluster_gmm(
    memories: List[MemoryElement], n_components: int, random_state: int = 42
) -> ClusteringResult:
    """
    Cluster memories using Gaussian Mixture Model (soft clustering).

    Returns ClusteringResult with soft_assignments populated.
    """
    embeddings, memory_ids, valid_memories = prepare_embeddings(memories)

    if len(embeddings) < n_components:
        raise ValueError(
            f"Not enough memories ({len(embeddings)}) for {n_components} components"
        )

    logger.info(f"Running GMM with n_components={n_components}")

    gmm = GaussianMixture(
        n_components=n_components, random_state=random_state, covariance_type="full"
    )
    gmm.fit(embeddings)

    # Get hard labels for metrics
    labels = gmm.predict(embeddings)

    # Get soft probabilities
    probabilities = gmm.predict_proba(embeddings)

    clusters, unclustered = labels_to_clusters(labels, memory_ids, embeddings)

    # Create soft assignments
    soft_assignments = []
    for i, mem_id in enumerate(memory_ids):
        cluster_probs = {
            clusters[j].id: float(probabilities[i][j]) for j in range(n_components)
        }
        soft_assignments.append(
            SoftClusterAssignment(memory_id=mem_id, cluster_probabilities=cluster_probs)
        )

    # Calculate metrics using hard labels
    sil_score = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0.0
    db_score = (
        davies_bouldin_score(embeddings, labels)
        if len(set(labels)) > 1
        else float("inf")
    )
    ch_score = (
        calinski_harabasz_score(embeddings, labels) if len(set(labels)) > 1 else 0.0
    )

    logger.info(
        f"GMM results: silhouette={sil_score:.4f}, "
        f"davies_bouldin={db_score:.4f}, calinski_harabasz={ch_score:.4f}"
    )

    return ClusteringResult(
        clusters=clusters,
        unclustered=unclustered,
        silhouette_score=sil_score,
        davies_bouldin_score=db_score,
        calinski_harabasz_score=ch_score,
        method=ClusteringMethod.GMM,
        parameters={"n_components": n_components, "random_state": random_state},
        soft_assignments=soft_assignments,
    )


def find_optimal_k(
    memories: List[MemoryElement], k_range: List[int], random_state: int = 42
) -> OptimalKResult:
    """
    Find optimal K using silhouette score analysis.

    Args:
        memories: List of MemoryElement objects
        k_range: List of K values to test (e.g., [3, 5, 8, 10, 12, 15, 20])
        random_state: Random seed for reproducibility

    Returns:
        OptimalKResult with silhouette scores and recommendation
    """
    embeddings, memory_ids, valid_memories = prepare_embeddings(memories)

    silhouette_scores = []

    logger.info(f"Finding optimal K in range {k_range}")

    for k in k_range:
        if k >= len(embeddings):
            logger.warning(f"Skipping k={k}, not enough samples ({len(embeddings)})")
            silhouette_scores.append(0.0)
            continue

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
        logger.info(f"K={k}: silhouette={score:.4f}")

    # Find optimal K (highest silhouette)
    optimal_idx = int(np.argmax(silhouette_scores))
    optimal_k = k_range[optimal_idx]

    # Detect elbow point (rate of change analysis)
    elbow_k = _detect_elbow(k_range, silhouette_scores)

    analysis = _generate_k_analysis(k_range, silhouette_scores, optimal_k, elbow_k)

    return OptimalKResult(
        k_values=k_range,
        silhouette_scores=silhouette_scores,
        optimal_k=optimal_k,
        elbow_k=elbow_k,
        analysis=analysis,
    )


def _detect_elbow(k_values: List[int], scores: List[float]) -> Optional[int]:
    """Detect elbow point in silhouette curve using second derivative."""
    if len(scores) < 3:
        return None

    # Calculate first derivative (rate of change)
    first_deriv = np.diff(scores)

    # Calculate second derivative (rate of rate of change)
    second_deriv = np.diff(first_deriv)

    # Find point where second derivative changes sign significantly
    # or where the improvement rate drops substantially
    for i in range(len(second_deriv)):
        # If score improvement becomes negative and second derivative is positive
        # (curve is flattening after peak)
        if first_deriv[i + 1] < 0 and second_deriv[i] > 0:
            return k_values[i + 1]

    return None


def _generate_k_analysis(
    k_values: List[int],
    scores: List[float],
    optimal_k: int,
    elbow_k: Optional[int],
) -> str:
    """Generate human-readable analysis of K selection."""
    lines = []
    lines.append("K-Value Analysis:")
    lines.append("-" * 40)

    for k, score in zip(k_values, scores):
        indicator = ""
        if k == optimal_k:
            indicator = " <- OPTIMAL (highest silhouette)"
        elif k == elbow_k:
            indicator = " <- ELBOW POINT"
        lines.append(f"  K={k:2d}: silhouette={score:.4f}{indicator}")

    lines.append("-" * 40)
    lines.append(f"Recommended K: {optimal_k}")
    lines.append(f"Best silhouette score: {max(scores):.4f}")

    if elbow_k and elbow_k != optimal_k:
        lines.append(f"Elbow point at K={elbow_k} (alternative choice)")

    return "\n".join(lines)
