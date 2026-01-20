"""
Topic clustering experiment for grouping semantically similar memories.

This experiment compares multiple clustering algorithms (K-Means, Hierarchical, DBSCAN, GMM)
to group memories by topic, generates LLM-based topic names, and evaluates cluster quality.

V1 Usage (original, has methodological issues):
    uv run python -m agent.experiments.topic_clustering.run_experiments --conversation <prefix>

V2 Usage (improved methodology):
    uv run python -m agent.experiments.topic_clustering.run_experiments_v2 --conversation <prefix>

V3 Usage (cross-action-type graph clustering - still has issues):
    uv run python -m agent.experiments.topic_clustering.run_experiments_v3 --conversation <prefix>

V4 Usage (cross-action-only similarity - no intra-action-type comparisons):
    uv run python -m agent.experiments.topic_clustering.run_experiments_v4 --conversation <prefix>
"""

from .models import (
    TopicCluster,
    ClusteringResult,
    ClusteringMethod,
    TopicNamingApproach,
    SoftClusterAssignment,
    OptimalKResult,
    TopicNamingResult,
    ClusterCoherenceReview,
    AlgorithmComparisonResult,
)

from .clustering import (
    cluster_kmeans,
    cluster_hierarchical,
    cluster_dbscan,
    cluster_gmm,
    find_optimal_k,
    prepare_embeddings,
    # V2 additions
    parse_action_type,
    get_action_types_for_memories,
    build_action_type_mapping,
    get_action_types_from_trigger_history,
    compute_action_type_centroids,
    project_orthogonal_to_action_types,
    calculate_action_type_entropy,
    cluster_with_residual_embeddings,
    # V3 additions
    build_knn_graph,
    apply_cross_action_type_weighting,
    cluster_cross_action_type,
    # V4 additions
    build_cross_action_affinity_matrix,
    cluster_cross_action_only,
)

from .topic_naming import (
    generate_topic_name_simple,
    generate_topic_name_structured,
    generate_topic_name_contrastive,
    generate_topic_names_all_approaches,
    name_all_clusters,
)

from .topic_summary import (
    generate_topic_summary,
    evaluate_summary_quality,
    summarize_all_clusters,
    TopicSummaryResponse,
    SummaryQualityEvaluation,
)

from .evaluation import (
    calculate_cluster_coherence,
    calculate_all_coherence_scores,
    find_cluster_outliers,
    calculate_inter_cluster_separation,
    generate_coherence_reviews,
    analyze_topic_overlap,
    test_query_against_summary,
    calculate_cluster_statistics,
    # V2 additions
    blind_coherence_evaluation,
    cluster_predictability_test,
    validate_cluster_name,
)

__all__ = [
    # Models
    "TopicCluster",
    "ClusteringResult",
    "ClusteringMethod",
    "TopicNamingApproach",
    "SoftClusterAssignment",
    "OptimalKResult",
    "TopicNamingResult",
    "ClusterCoherenceReview",
    "AlgorithmComparisonResult",
    # Clustering
    "cluster_kmeans",
    "cluster_hierarchical",
    "cluster_dbscan",
    "cluster_gmm",
    "find_optimal_k",
    "prepare_embeddings",
    # V2 clustering
    "parse_action_type",
    "get_action_types_for_memories",
    "build_action_type_mapping",
    "get_action_types_from_trigger_history",
    "compute_action_type_centroids",
    "project_orthogonal_to_action_types",
    "calculate_action_type_entropy",
    "cluster_with_residual_embeddings",
    # V3 clustering
    "build_knn_graph",
    "apply_cross_action_type_weighting",
    "cluster_cross_action_type",
    # V4 clustering
    "build_cross_action_affinity_matrix",
    "cluster_cross_action_only",
    # Topic naming
    "generate_topic_name_simple",
    "generate_topic_name_structured",
    "generate_topic_name_contrastive",
    "generate_topic_names_all_approaches",
    "name_all_clusters",
    # Summary
    "generate_topic_summary",
    "evaluate_summary_quality",
    "summarize_all_clusters",
    "TopicSummaryResponse",
    "SummaryQualityEvaluation",
    # Evaluation
    "calculate_cluster_coherence",
    "calculate_all_coherence_scores",
    "find_cluster_outliers",
    "calculate_inter_cluster_separation",
    "generate_coherence_reviews",
    "analyze_topic_overlap",
    "test_query_against_summary",
    "calculate_cluster_statistics",
    # V2 evaluation
    "blind_coherence_evaluation",
    "cluster_predictability_test",
    "validate_cluster_name",
]
