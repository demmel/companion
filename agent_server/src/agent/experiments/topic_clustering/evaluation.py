"""
Evaluation metrics and manual review helpers for cluster quality.
"""

import logging
import random
from typing import List, Dict, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from agent.embedding_service import get_embedding_service
from agent.llm import LLM, SupportedModel
from agent.memory.dag.models import MemoryGraph, MemoryElement
from agent.state import State
from agent.structured_llm import direct_structured_llm_call

from .models import (
    TopicCluster,
    ClusteringResult,
    ClusterCoherenceReview,
    SoftClusterAssignment,
)

logger = logging.getLogger(__name__)


def calculate_cluster_coherence(
    cluster: TopicCluster, memory_graph: MemoryGraph
) -> float:
    """
    Calculate coherence score for a single cluster.

    Coherence = average cosine similarity within cluster.
    Higher is better (tighter cluster).
    """
    embeddings = []
    for mem_id in cluster.memory_ids:
        if mem_id in memory_graph.elements:
            mem = memory_graph.elements[mem_id]
            if mem.embedding_vector is not None:
                embeddings.append(mem.embedding_vector)

    if len(embeddings) < 2:
        return 1.0  # Single element is perfectly coherent

    embeddings_array = np.array(embeddings)
    centroid = np.mean(embeddings_array, axis=0)

    # Calculate average cosine similarity to centroid
    similarities = []
    for emb in embeddings_array:
        norm_emb = np.linalg.norm(emb)
        norm_centroid = np.linalg.norm(centroid)
        if norm_emb > 0 and norm_centroid > 0:
            similarity = np.dot(emb, centroid) / (norm_emb * norm_centroid)
            similarities.append(similarity)
        else:
            similarities.append(0.0)

    return float(np.mean(similarities))


def calculate_all_coherence_scores(
    clustering_result: ClusteringResult, memory_graph: MemoryGraph
) -> Dict[str, float]:
    """Calculate coherence for all clusters."""
    scores = {}
    for cluster in clustering_result.clusters:
        score = calculate_cluster_coherence(cluster, memory_graph)
        scores[cluster.id] = score
        logger.debug(f"Cluster '{cluster.name}' coherence: {score:.4f}")
    return scores


def find_cluster_outliers(
    cluster: TopicCluster, memory_graph: MemoryGraph, threshold: float = 0.3
) -> List[str]:
    """
    Find memories that don't fit well in the cluster.

    Outlier = memory whose cosine distance to centroid > threshold
    (or equivalently, similarity < 1 - threshold)
    """
    outliers = []
    centroid = np.array(cluster.centroid)
    norm_centroid = np.linalg.norm(centroid)

    if norm_centroid == 0:
        return []

    similarity_threshold = 1 - threshold

    for mem_id in cluster.memory_ids:
        if mem_id in memory_graph.elements:
            mem = memory_graph.elements[mem_id]
            if mem.embedding_vector is not None:
                emb = np.array(mem.embedding_vector)
                norm_emb = np.linalg.norm(emb)
                if norm_emb > 0:
                    similarity = np.dot(emb, centroid) / (norm_emb * norm_centroid)
                    if similarity < similarity_threshold:
                        outliers.append(mem_id)

    logger.info(
        f"Found {len(outliers)} outliers in cluster '{cluster.name}' "
        f"(threshold={threshold})"
    )

    return outliers


def calculate_inter_cluster_separation(clustering_result: ClusteringResult) -> float:
    """
    Calculate average distance between cluster centroids.

    Higher = better separated clusters.
    """
    if len(clustering_result.clusters) < 2:
        return 0.0

    centroids = [np.array(c.centroid) for c in clustering_result.clusters]

    distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            # Cosine distance
            norm_i = np.linalg.norm(centroids[i])
            norm_j = np.linalg.norm(centroids[j])
            if norm_i > 0 and norm_j > 0:
                similarity = np.dot(centroids[i], centroids[j]) / (norm_i * norm_j)
                distance = 1 - similarity
                distances.append(distance)

    avg_separation = float(np.mean(distances)) if distances else 0.0
    logger.info(f"Inter-cluster separation: {avg_separation:.4f}")

    return avg_separation


def generate_coherence_reviews(
    clustering_result: ClusteringResult,
    memory_graph: MemoryGraph,
    sample_size: int = 10,
) -> List[ClusterCoherenceReview]:
    """
    Generate review templates for manual cluster evaluation.

    Creates ClusterCoherenceReview objects with sample memories
    for each cluster to facilitate manual review.
    """
    reviews = []

    for cluster in clustering_result.clusters:
        # Get sample memories
        sample_contents = []
        for mem_id in cluster.memory_ids[:sample_size]:
            if mem_id in memory_graph.elements:
                mem = memory_graph.elements[mem_id]
                content = (
                    mem.content[:200] + "..." if len(mem.content) > 200 else mem.content
                )
                sample_contents.append(content)

        review = ClusterCoherenceReview(
            cluster_id=cluster.id,
            cluster_name=cluster.name or f"Cluster {cluster.id[:8]}",
            sample_memories=sample_contents,
            coherence_rating=None,  # To be filled during manual review
            outliers_identified=[],
            notes="",
        )
        reviews.append(review)

    logger.info(f"Generated {len(reviews)} coherence review templates")

    return reviews


def analyze_topic_overlap(
    clustering_result: ClusteringResult, probability_threshold: float = 0.3
) -> Dict[str, object]:
    """
    Analyze topic overlap patterns from GMM soft clustering.

    Returns:
        - multi_topic_count: Number of memories in multiple clusters
        - multi_topic_percentage: Percentage of total
        - overlap_pairs: Which cluster pairs overlap most
        - example_multi_topic_memories: Sample memories with probabilities
    """
    if clustering_result.soft_assignments is None:
        raise ValueError(
            "Soft assignments required for overlap analysis. Use GMM clustering."
        )

    multi_topic = [
        sa
        for sa in clustering_result.soft_assignments
        if sa.is_multi_topic(probability_threshold)
    ]

    # Analyze which cluster pairs overlap
    overlap_counts: Dict[Tuple[str, str], int] = {}
    for sa in multi_topic:
        above_threshold = [
            cid
            for cid, prob in sa.cluster_probabilities.items()
            if prob >= probability_threshold
        ]
        # Count each pair
        for i, c1 in enumerate(above_threshold):
            for c2 in above_threshold[i + 1 :]:
                pair = tuple(sorted([c1, c2]))
                overlap_counts[pair] = overlap_counts.get(pair, 0) + 1

    # Sort pairs by overlap count
    sorted_pairs = sorted(overlap_counts.items(), key=lambda x: x[1], reverse=True)

    # Get cluster name mapping
    cluster_names = {c.id: c.name for c in clustering_result.clusters}

    # Format overlap pairs with names
    formatted_pairs = []
    for (c1, c2), count in sorted_pairs[:10]:
        name1 = cluster_names.get(c1, c1[:8])
        name2 = cluster_names.get(c2, c2[:8])
        formatted_pairs.append({"cluster1": name1, "cluster2": name2, "count": count})

    # Format example multi-topic memories
    example_memories = []
    for sa in multi_topic[:5]:
        probs_sorted = sorted(
            sa.cluster_probabilities.items(), key=lambda x: x[1], reverse=True
        )
        top_probs = [
            {"cluster": cluster_names.get(cid, cid[:8]), "probability": prob}
            for cid, prob in probs_sorted[:3]
        ]
        example_memories.append(
            {"memory_id": sa.memory_id, "top_probabilities": top_probs}
        )

    total = len(clustering_result.soft_assignments)
    percentage = (len(multi_topic) / total * 100) if total > 0 else 0

    logger.info(
        f"Topic overlap analysis: {len(multi_topic)}/{total} ({percentage:.1f}%) "
        f"memories belong to multiple topics"
    )

    return {
        "multi_topic_count": len(multi_topic),
        "multi_topic_percentage": percentage,
        "overlap_pairs": formatted_pairs,
        "example_multi_topic_memories": example_memories,
    }


def test_query_against_summary(
    query: str, summary: str, threshold: float = 0.5
) -> Dict[str, object]:
    """
    Test if a query would match a cluster summary.

    Returns cosine similarity and match determination.
    """
    embedding_service = get_embedding_service()

    query_emb = embedding_service.encode(query)
    summary_emb = embedding_service.encode(summary)

    similarity = embedding_service.cosine_similarity(query_emb, summary_emb)

    return {
        "query": query,
        "similarity": similarity,
        "would_match": similarity >= threshold,
    }


def calculate_cluster_statistics(
    clustering_result: ClusteringResult, memory_graph: MemoryGraph
) -> Dict[str, object]:
    """
    Calculate comprehensive statistics for a clustering result.
    """
    cluster_sizes = [len(c.memory_ids) for c in clustering_result.clusters]
    coherence_scores = calculate_all_coherence_scores(clustering_result, memory_graph)

    stats = {
        "num_clusters": len(clustering_result.clusters),
        "num_unclustered": len(clustering_result.unclustered),
        "total_memories": sum(cluster_sizes) + len(clustering_result.unclustered),
        "cluster_sizes": {
            "min": min(cluster_sizes) if cluster_sizes else 0,
            "max": max(cluster_sizes) if cluster_sizes else 0,
            "mean": float(np.mean(cluster_sizes)) if cluster_sizes else 0,
            "std": float(np.std(cluster_sizes)) if cluster_sizes else 0,
        },
        "coherence_scores": {
            "min": min(coherence_scores.values()) if coherence_scores else 0,
            "max": max(coherence_scores.values()) if coherence_scores else 0,
            "mean": (
                float(np.mean(list(coherence_scores.values())))
                if coherence_scores
                else 0
            ),
        },
        "silhouette_score": clustering_result.silhouette_score,
        "davies_bouldin_score": clustering_result.davies_bouldin_score,
        "calinski_harabasz_score": clustering_result.calinski_harabasz_score,
        "inter_cluster_separation": calculate_inter_cluster_separation(
            clustering_result
        ),
    }

    return stats


# =============================================================================
# V2 Evaluation Functions
# =============================================================================


class ThemeIdentificationResponse(BaseModel):
    """LLM response for identifying shared theme in memories."""

    theme: str = Field(description="The main topic/theme these memories share")
    confidence: float = Field(
        description="Confidence in the identified theme (0-1)", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this theme was identified"
    )


class ThemeComparisonResponse(BaseModel):
    """LLM response for comparing two themes."""

    are_same_theme: bool = Field(
        description="Whether the two themes represent the same underlying topic"
    )
    similarity_score: float = Field(
        description="How similar the themes are (0-1)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Explanation of the comparison")


class MemoryClassificationResponse(BaseModel):
    """LLM response for classifying which memories belong to a topic."""

    memory_indices_in_topic: List[int] = Field(
        description="Indices of memories that belong to the given topic (0-indexed)"
    )
    reasoning: str = Field(description="Brief explanation of classification decisions")


def blind_coherence_evaluation(
    cluster: TopicCluster,
    memory_graph: MemoryGraph,
    state: State,
    llm: LLM,
    model: SupportedModel,
    num_samples: int = 3,
    sample_size: int = 5,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Evaluate cluster coherence using blind theme identification.

    Method:
    1. Randomly sample num_samples sets of sample_size memories from the cluster
    2. For each set, ask LLM to identify the shared theme (without knowing they're clustered)
    3. Compare themes across sets - if cluster is coherent, themes should match

    Returns:
        - themes: List of identified themes for each sample
        - agreement_score: How often themes matched (0-1)
        - avg_confidence: Average confidence across samples
    """
    rng = random.Random(random_seed)

    if len(cluster.memory_ids) < sample_size * num_samples:
        # Not enough memories for full sampling, use smaller samples
        sample_size = max(3, len(cluster.memory_ids) // num_samples)

    # Sample memory sets
    all_memory_ids = list(cluster.memory_ids)
    samples: List[List[str]] = []

    for _ in range(num_samples):
        if len(all_memory_ids) >= sample_size:
            sample = rng.sample(all_memory_ids, sample_size)
            samples.append(sample)

    if not samples:
        return {
            "themes": [],
            "agreement_score": 0.0,
            "avg_confidence": 0.0,
            "error": "Not enough memories for sampling",
        }

    # Identify theme for each sample
    themes: List[ThemeIdentificationResponse] = []
    for sample_ids in samples:
        # Get memory contents
        contents = []
        for mid in sample_ids:
            if mid in memory_graph.elements:
                mem = memory_graph.elements[mid]
                # Truncate long content
                content = mem.content[:500] if len(mem.content) > 500 else mem.content
                contents.append(content)

        if not contents:
            continue

        prompt = f"""You are analyzing a set of memories from an AI companion named {state.name}.

Below are {len(contents)} memories. Your task is to identify the main topic or theme they share.

MEMORIES:
{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(contents))}

Identify the main theme these memories have in common. Be specific but concise."""

        try:
            response = direct_structured_llm_call(
                prompt=prompt,
                response_model=ThemeIdentificationResponse,
                model=model,
                llm=llm,
                caller="blind_coherence_evaluation",
            )
            themes.append(response)
        except Exception as e:
            logger.warning(f"Theme identification failed: {e}")

    if len(themes) < 2:
        return {
            "themes": [t.theme for t in themes],
            "agreement_score": 1.0 if themes else 0.0,
            "avg_confidence": themes[0].confidence if themes else 0.0,
            "error": "Not enough successful theme identifications",
        }

    # Compare themes pairwise
    agreements = 0
    comparisons = 0

    for i in range(len(themes)):
        for j in range(i + 1, len(themes)):
            prompt = f"""Compare these two themes identified from different memory samples:

Theme 1: {themes[i].theme}
Theme 2: {themes[j].theme}

Determine if these represent the same underlying topic/theme, or different topics."""

            try:
                response = direct_structured_llm_call(
                    prompt=prompt,
                    response_model=ThemeComparisonResponse,
                    model=model,
                    llm=llm,
                    caller="blind_coherence_comparison",
                )
                comparisons += 1
                if response.are_same_theme or response.similarity_score >= 0.7:
                    agreements += 1
            except Exception as e:
                logger.warning(f"Theme comparison failed: {e}")

    agreement_score = agreements / comparisons if comparisons > 0 else 0.0
    avg_confidence = np.mean([t.confidence for t in themes])

    return {
        "themes": [t.theme for t in themes],
        "agreement_score": float(agreement_score),
        "avg_confidence": float(avg_confidence),
        "num_samples": len(themes),
        "num_comparisons": comparisons,
        "num_agreements": agreements,
    }


def cluster_predictability_test(
    memories: List[MemoryElement],
    embeddings: np.ndarray,
    memory_ids: List[str],
    k: int,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Test if cluster assignments are predictable from held-out data.

    Method:
    1. Split data into train (80%) and test (20%)
    2. Cluster training data
    3. Assign test data to nearest cluster centroid
    4. Measure how well test assignments match what full clustering would produce

    Returns:
        - train_size: Number of training samples
        - test_size: Number of test samples
        - assignment_accuracy: How often test assignment matches full clustering
    """
    n = len(embeddings)
    test_size = int(n * test_fraction)
    train_size = n - test_size

    if train_size < k or test_size < 1:
        return {
            "error": "Not enough samples for split",
            "train_size": train_size,
            "test_size": test_size,
        }

    # Random split
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_embeddings = embeddings[train_indices]
    test_embeddings = embeddings[test_indices]

    # Cluster training data
    train_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    train_labels = train_kmeans.fit_predict(train_embeddings)
    train_centroids = train_kmeans.cluster_centers_

    # Assign test data to nearest centroid
    test_assignments = []
    for emb in test_embeddings:
        distances = [np.linalg.norm(emb - c) for c in train_centroids]
        test_assignments.append(int(np.argmin(distances)))

    # For comparison: cluster ALL data and see what test items get
    full_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    full_labels = full_kmeans.fit_predict(embeddings)
    full_test_labels = full_labels[test_indices]

    # Map train cluster labels to full cluster labels (they may not align perfectly)
    # Use Hungarian algorithm or simple majority voting
    from collections import Counter

    label_mapping: Dict[int, int] = {}
    for train_label in range(k):
        # Find which full label most train points with this label have
        train_mask = train_labels == train_label
        corresponding_full = full_labels[train_indices][train_mask]
        if len(corresponding_full) > 0:
            label_mapping[train_label] = Counter(corresponding_full).most_common(1)[0][
                0
            ]
        else:
            label_mapping[train_label] = train_label

    # Map test assignments to full label space
    mapped_assignments = [label_mapping.get(a, a) for a in test_assignments]

    # Calculate accuracy
    correct = sum(1 for a, b in zip(mapped_assignments, full_test_labels) if a == b)
    accuracy = correct / len(test_indices)

    return {
        "train_size": train_size,
        "test_size": test_size,
        "assignment_accuracy": float(accuracy),
        "correct_assignments": correct,
    }


def validate_cluster_name(
    name: str,
    cluster: TopicCluster,
    other_clusters: List[TopicCluster],
    memory_graph: MemoryGraph,
    state: State,
    llm: LLM,
    model: SupportedModel,
    num_in_cluster: int = 5,
    num_from_others: int = 5,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Validate a cluster name by testing if LLM can identify cluster members.

    Method:
    1. Sample memories from the target cluster
    2. Sample memories from other clusters
    3. Present all memories (shuffled) with the topic name
    4. Ask LLM to identify which memories belong to the topic
    5. Calculate precision and recall

    Returns:
        - precision: Of memories LLM said belong, what fraction actually do
        - recall: Of memories that belong, what fraction did LLM identify
        - f1_score: Harmonic mean of precision and recall
    """
    rng = random.Random(random_seed)

    # Sample from target cluster
    in_cluster_ids = rng.sample(
        cluster.memory_ids, min(num_in_cluster, len(cluster.memory_ids))
    )

    # Sample from other clusters
    other_ids: List[str] = []
    all_other_ids = [mid for c in other_clusters for mid in c.memory_ids]
    if all_other_ids:
        other_ids = rng.sample(all_other_ids, min(num_from_others, len(all_other_ids)))

    # Combine and shuffle
    all_ids = in_cluster_ids + other_ids
    ground_truth = [True] * len(in_cluster_ids) + [False] * len(other_ids)

    # Shuffle together
    combined = list(zip(all_ids, ground_truth))
    rng.shuffle(combined)
    shuffled_ids, shuffled_truth = zip(*combined) if combined else ([], [])

    # Get memory contents
    contents = []
    for mid in shuffled_ids:
        if mid in memory_graph.elements:
            mem = memory_graph.elements[mid]
            content = mem.content[:400] if len(mem.content) > 400 else mem.content
            contents.append(content)

    if not contents:
        return {
            "error": "No valid memories found",
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
        }

    prompt = f"""You are analyzing memories from an AI companion named {state.name}.

TOPIC: {name}

Below are {len(contents)} memories. Identify which ones belong to the topic "{name}".

MEMORIES:
{chr(10).join(f'[{i}] {c}' for i, c in enumerate(contents))}

Return the indices (0-indexed) of memories that belong to this topic."""

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=MemoryClassificationResponse,
            model=model,
            llm=llm,
            caller="validate_cluster_name",
        )
        predicted_in_topic = set(response.memory_indices_in_topic)
    except Exception as e:
        logger.warning(f"Name validation failed: {e}")
        return {"error": str(e), "precision": 0, "recall": 0, "f1_score": 0}

    # Calculate precision/recall
    actual_in_topic = {i for i, t in enumerate(shuffled_truth) if t}

    true_positives = len(predicted_in_topic & actual_in_topic)
    false_positives = len(predicted_in_topic - actual_in_topic)
    false_negatives = len(actual_in_topic - predicted_in_topic)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "num_in_cluster": len(in_cluster_ids),
        "num_from_others": len(other_ids),
    }
