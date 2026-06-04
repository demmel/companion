"""
Main experiment runner for topic clustering experiments.

Run all experiments:
    uv run python -m agent.experiments.topic_clustering.run_experiments --conversation <prefix>

Run specific experiment:
    uv run python -m agent.experiments.topic_clustering.run_experiments --conversation <prefix> --experiment 1
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

from agent.conversation_persistence import ConversationPersistence
from agent.llm import create_llm, SupportedModel
from agent.memory.dag.models import MemoryElement
from agent.memory.dag.dag_memory_manager import DagMemoryManager
from agent.ui_output import ui_print

from .models import (
    ClusteringResult,
    ClusteringMethod,
    TopicNamingApproach,
    TopicCluster,
)
from .clustering import (
    cluster_kmeans,
    cluster_hierarchical,
    cluster_dbscan,
    cluster_gmm,
    find_optimal_k,
)
from .topic_naming import (
    name_all_clusters,
    generate_topic_names_all_approaches,
)
from .topic_summary import (
    summarize_all_clusters,
    evaluate_summary_quality,
)
from .evaluation import (
    calculate_all_coherence_scores,
    generate_coherence_reviews,
    analyze_topic_overlap,
    calculate_cluster_statistics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TopicClusteringExperiment:
    """Main experiment orchestrator."""

    def __init__(
        self,
        conversation_prefix: str,
        conversations_dir: str = "conversations",
        output_dir: str = str(Path(__file__).parent / "output" / "results"),
    ):
        self.conversation_prefix = conversation_prefix
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load agent data
        logger.info(f"Loading agent data from {conversation_prefix}")
        persistence = ConversationPersistence(conversations_dir=conversations_dir)
        agent_data = persistence.load_agent_data(
            conversation_prefix, use_individual_formatting=True
        )

        self.state = agent_data.state

        # Get memory manager (IMemory interface, but actually DagMemoryManager)
        memory = agent_data.memory
        if not isinstance(memory, DagMemoryManager):
            raise ValueError(f"Expected DagMemoryManager, got {type(memory)}")
        self.dag_manager = memory
        self.memory_graph = self.dag_manager.get_memory_graph()

        # Get memories with embeddings
        self.memories: List[MemoryElement] = [
            mem
            for mem in self.memory_graph.elements.values()
            if mem.embedding_vector is not None
        ]

        total = len(self.memory_graph.elements)
        with_emb = len(self.memories)
        logger.info(f"Loaded {total} memories, {with_emb} with embeddings")

        if with_emb < total:
            logger.warning(
                f"{total - with_emb} memories lack embeddings and will be excluded"
            )

        # LLM setup
        self.llm = create_llm()
        self.model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    def run_experiment_1_algorithm_comparison(self) -> Dict[str, object]:
        """
        Experiment 1: Algorithm Comparison

        Compare K-Means, Hierarchical, and DBSCAN on:
        - Silhouette score
        - Number of clusters found
        - Cluster size distribution
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 1: Algorithm Comparison")
        ui_print("=" * 80)

        results: Dict[str, object] = {}

        # K-Means with various K values
        for k in [5, 8, 10, 15]:
            ui_print(f"\nRunning K-Means with K={k}...")
            try:
                result = cluster_kmeans(self.memories, k=k)
                named_result = self._name_clusters(result)
                results[f"kmeans_k{k}"] = self._serialize_result(named_result)
            except Exception as e:
                logger.error(f"K-Means K={k} failed: {e}")
                results[f"kmeans_k{k}"] = {"error": str(e)}

        # Hierarchical with various cluster counts
        for n in [5, 8, 10]:
            ui_print(f"\nRunning Hierarchical with n_clusters={n}...")
            try:
                result = cluster_hierarchical(self.memories, n_clusters=n)
                named_result = self._name_clusters(result)
                results[f"hierarchical_n{n}"] = self._serialize_result(named_result)
            except Exception as e:
                logger.error(f"Hierarchical n={n} failed: {e}")
                results[f"hierarchical_n{n}"] = {"error": str(e)}

        # DBSCAN with various eps values
        for eps in [0.2, 0.3, 0.4, 0.5]:
            ui_print(f"\nRunning DBSCAN with eps={eps}...")
            try:
                result = cluster_dbscan(self.memories, eps=eps, min_samples=3)
                if result.clusters:
                    named_result = self._name_clusters(result)
                    results[f"dbscan_eps{eps}"] = self._serialize_result(named_result)
                else:
                    results[f"dbscan_eps{eps}"] = {
                        "error": "No clusters found",
                        "unclustered": len(result.unclustered),
                    }
            except Exception as e:
                logger.error(f"DBSCAN eps={eps} failed: {e}")
                results[f"dbscan_eps{eps}"] = {"error": str(e)}

        # Save results
        self._save_results("experiment_1_algorithm_comparison", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Algorithm Comparison Summary:")
        for name, data in results.items():
            if isinstance(data, dict) and "error" not in data:
                ui_print(
                    f"  {name}: silhouette={data.get('silhouette_score', 0):.4f}, "
                    f"clusters={data.get('num_clusters', 0)}"
                )

        return results

    def run_experiment_2_optimal_k(self) -> Dict[str, object]:
        """
        Experiment 2: Optimal K Discovery

        Find optimal K using silhouette analysis.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 2: Optimal K Discovery")
        ui_print("=" * 80)

        k_range = [3, 5, 8, 10, 12, 15, 20]
        result = find_optimal_k(self.memories, k_range)

        results: Dict[str, object] = {
            "k_values": result.k_values,
            "silhouette_scores": result.silhouette_scores,
            "optimal_k": result.optimal_k,
            "elbow_k": result.elbow_k,
            "analysis": result.analysis,
        }

        self._save_results("experiment_2_optimal_k", results)

        ui_print("\n" + result.analysis)

        return results

    def run_experiment_3_coherence_review(
        self, method: ClusteringMethod = ClusteringMethod.KMEANS, k: int = 8
    ) -> Dict[str, object]:
        """
        Experiment 3: Cluster Coherence Review

        Generate review templates for manual evaluation.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 3: Cluster Coherence Review")
        ui_print("=" * 80)

        # Run clustering with best parameters
        ui_print(f"\nClustering with {method.value}, k={k}...")
        if method == ClusteringMethod.KMEANS:
            result = cluster_kmeans(self.memories, k=k)
        elif method == ClusteringMethod.HIERARCHICAL:
            result = cluster_hierarchical(self.memories, n_clusters=k)
        else:
            raise ValueError(f"Unsupported method for coherence review: {method}")

        # Name clusters
        ui_print("Naming clusters...")
        named_result = self._name_clusters(result)

        # Calculate coherence scores
        ui_print("Calculating coherence scores...")
        coherence_scores = calculate_all_coherence_scores(
            named_result, self.memory_graph
        )

        # Generate review templates
        ui_print("Generating review templates...")
        reviews = generate_coherence_reviews(
            named_result, self.memory_graph, sample_size=10
        )

        results: Dict[str, object] = {
            "clustering": self._serialize_result(named_result),
            "coherence_scores": coherence_scores,
            "reviews": [
                {
                    "cluster_id": r.cluster_id,
                    "cluster_name": r.cluster_name,
                    "sample_memories": r.sample_memories,
                    "coherence_rating": r.coherence_rating,
                    "notes": r.notes,
                }
                for r in reviews
            ],
        }

        self._save_results("experiment_3_coherence_review", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Coherence Review Summary:")
        for cluster in named_result.clusters:
            score = coherence_scores.get(cluster.id, 0)
            ui_print(
                f"  '{cluster.name}': coherence={score:.4f}, size={len(cluster.memory_ids)}"
            )

        return results

    def run_experiment_4_topic_naming(self, k: int = 8) -> Dict[str, object]:
        """
        Experiment 4: Topic Naming Quality

        Compare simple, structured, and contrastive naming approaches.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 4: Topic Naming Quality")
        ui_print("=" * 80)

        # Run clustering
        ui_print(f"\nClustering with k={k}...")
        result = cluster_kmeans(self.memories, k=k)

        # Generate names with all approaches for each cluster
        naming_results = []
        for cluster in tqdm(result.clusters, desc="Naming clusters", file=sys.stdout):
            ui_print(
                f"\nNaming cluster {len(naming_results) + 1}/{len(result.clusters)}..."
            )
            naming_result = generate_topic_names_all_approaches(
                cluster=cluster,
                memory_graph=self.memory_graph,
                all_clusters=result.clusters,
                state=self.state,
                llm=self.llm,
                model=self.model,
            )
            naming_results.append(
                {
                    "cluster_id": naming_result.cluster_id,
                    "simple_name": naming_result.names_by_approach.get(
                        TopicNamingApproach.SIMPLE, ""
                    ),
                    "structured_name": naming_result.names_by_approach.get(
                        TopicNamingApproach.STRUCTURED, ""
                    ),
                    "contrastive_name": naming_result.names_by_approach.get(
                        TopicNamingApproach.CONTRASTIVE, ""
                    ),
                    "best_approach": naming_result.best_approach.value,
                    "evaluation_notes": naming_result.evaluation_notes,
                }
            )

        results: Dict[str, object] = {
            "k": k,
            "naming_results": naming_results,
        }

        self._save_results("experiment_4_topic_naming", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Topic Naming Comparison:")
        for nr in naming_results:
            ui_print(f"\n  Cluster: {nr['cluster_id'][:8]}...")
            ui_print(f"    Simple:      {nr['simple_name']}")
            ui_print(f"    Structured:  {nr['structured_name']}")
            ui_print(f"    Contrastive: {nr['contrastive_name']}")

        return results

    def run_experiment_5_summary_quality(self, k: int = 8) -> Dict[str, object]:
        """
        Experiment 5: Summary Quality

        Generate and evaluate cluster summaries.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 5: Summary Quality")
        ui_print("=" * 80)

        # Run clustering and name clusters
        ui_print(f"\nClustering with k={k}...")
        result = cluster_kmeans(self.memories, k=k)
        ui_print("Naming clusters...")
        named_result = self._name_clusters(result)

        # Generate summaries
        ui_print("Generating summaries...")
        summaries = summarize_all_clusters(
            clusters=named_result.clusters,
            memory_graph=self.memory_graph,
            state=self.state,
            llm=self.llm,
            model=self.model,
        )

        # Evaluate summaries
        ui_print("Evaluating summaries...")
        evaluations = []
        for cluster, summary in zip(named_result.clusters, summaries):
            evaluation = evaluate_summary_quality(
                summary=summary,
                cluster=cluster,
                memory_graph=self.memory_graph,
                state=self.state,
                llm=self.llm,
                model=self.model,
            )
            evaluations.append(
                {
                    "cluster_id": cluster.id,
                    "cluster_name": cluster.name,
                    "summary": summary.summary,
                    "key_events": summary.key_events,
                    "themes": summary.themes,
                    "searchable_terms": summary.searchable_terms,
                    "evaluation": {
                        "completeness": evaluation.completeness_score,
                        "accuracy": evaluation.accuracy_score,
                        "coherence": evaluation.coherence_score,
                        "searchability": evaluation.searchability_score,
                        "overall": evaluation.overall_score,
                        "issues": evaluation.issues,
                    },
                }
            )

        results: Dict[str, object] = {
            "k": k,
            "summary_evaluations": evaluations,
        }

        self._save_results("experiment_5_summary_quality", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Summary Quality Scores:")
        for ev in evaluations:
            ui_print(
                f"  '{ev['cluster_name']}': overall={ev['evaluation']['overall']:.2f}"
            )

        return results

    def run_experiment_6_topic_overlap(
        self, n_components: int = 8
    ) -> Dict[str, object]:
        """
        Experiment 6: Topic Overlap Analysis

        Use GMM soft clustering to analyze topic overlap.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 6: Topic Overlap Analysis")
        ui_print("=" * 80)

        # Run GMM clustering
        ui_print(f"\nRunning GMM with n_components={n_components}...")
        result = cluster_gmm(self.memories, n_components=n_components)
        ui_print("Naming clusters...")
        named_result = self._name_clusters(result)

        # Analyze overlap
        ui_print("Analyzing topic overlap...")
        overlap_analysis = analyze_topic_overlap(
            named_result, probability_threshold=0.3
        )

        results: Dict[str, object] = {
            "n_components": n_components,
            "clustering": self._serialize_result(named_result),
            "overlap_analysis": {
                "multi_topic_count": overlap_analysis["multi_topic_count"],
                "multi_topic_percentage": overlap_analysis["multi_topic_percentage"],
                "top_overlapping_pairs": overlap_analysis["overlap_pairs"],
                "example_memories": overlap_analysis["example_multi_topic_memories"],
            },
        }

        self._save_results("experiment_6_topic_overlap", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Topic Overlap Analysis:")
        ui_print(
            f"  Multi-topic memories: {overlap_analysis['multi_topic_count']} "
            f"({overlap_analysis['multi_topic_percentage']:.1f}%)"
        )
        ui_print("  Top overlapping pairs:")
        for pair in overlap_analysis["overlap_pairs"][:5]:
            ui_print(f"    {pair['cluster1']} <-> {pair['cluster2']}: {pair['count']}")

        return results

    def run_all_experiments(self) -> Dict[str, object]:
        """Run all experiments in sequence."""
        all_results: Dict[str, object] = {}

        all_results["experiment_1"] = self.run_experiment_1_algorithm_comparison()
        all_results["experiment_2"] = self.run_experiment_2_optimal_k()

        # Use optimal K from experiment 2 for subsequent experiments
        optimal_k = all_results["experiment_2"]["optimal_k"]
        ui_print(f"\n*** Using optimal K={optimal_k} for remaining experiments ***\n")

        all_results["experiment_3"] = self.run_experiment_3_coherence_review(
            k=optimal_k
        )
        all_results["experiment_4"] = self.run_experiment_4_topic_naming(k=optimal_k)
        all_results["experiment_5"] = self.run_experiment_5_summary_quality(k=optimal_k)
        all_results["experiment_6"] = self.run_experiment_6_topic_overlap(
            n_components=optimal_k
        )

        # Save consolidated results
        self._save_results(
            "all_experiments_summary",
            {
                "conversation_prefix": self.conversation_prefix,
                "total_memories": len(self.memory_graph.elements),
                "memories_with_embeddings": len(self.memories),
                "optimal_k": optimal_k,
                "timestamp": datetime.now().isoformat(),
            },
        )

        ui_print("\n" + "=" * 80)
        ui_print("ALL EXPERIMENTS COMPLETE")
        ui_print(f"Results saved to: {self.output_dir}")
        ui_print("=" * 80)

        return all_results

    def _name_clusters(self, result: ClusteringResult) -> ClusteringResult:
        """Name all clusters in a clustering result using structured approach."""
        named_clusters = name_all_clusters(
            clusters=result.clusters,
            memory_graph=self.memory_graph,
            state=self.state,
            llm=self.llm,
            model=self.model,
            approach=TopicNamingApproach.STRUCTURED,
        )

        return ClusteringResult(
            clusters=named_clusters,
            unclustered=result.unclustered,
            silhouette_score=result.silhouette_score,
            davies_bouldin_score=result.davies_bouldin_score,
            calinski_harabasz_score=result.calinski_harabasz_score,
            method=result.method,
            parameters=result.parameters,
            soft_assignments=result.soft_assignments,
        )

    def _serialize_result(self, result: ClusteringResult) -> Dict[str, object]:
        """Serialize ClusteringResult for JSON output."""
        return {
            "method": result.method.value,
            "parameters": result.parameters,
            "silhouette_score": result.silhouette_score,
            "davies_bouldin_score": result.davies_bouldin_score,
            "calinski_harabasz_score": result.calinski_harabasz_score,
            "num_clusters": len(result.clusters),
            "num_unclustered": len(result.unclustered),
            "cluster_sizes": [len(c.memory_ids) for c in result.clusters],
            "clusters": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "size": len(c.memory_ids),
                    "coherence_score": c.coherence_score,
                    "keywords": c.keywords,
                }
                for c in result.clusters
            ],
        }

    def _save_results(self, name: str, results: Dict[str, object]) -> None:
        """Save results to JSON file."""
        filepath = self.output_dir / f"{name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run topic clustering experiments")
    parser.add_argument(
        "--conversation", type=str, required=True, help="Conversation ID prefix"
    )
    parser.add_argument(
        "--conversations-dir",
        type=str,
        default="conversations",
        help="Directory containing conversation files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "output" / "results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Run specific experiment (1-6). If not specified, runs all.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of clusters for experiments that need K",
    )

    args = parser.parse_args()

    experiment = TopicClusteringExperiment(
        conversation_prefix=args.conversation,
        conversations_dir=args.conversations_dir,
        output_dir=args.output_dir,
    )

    if args.experiment is None:
        experiment.run_all_experiments()
    else:
        if args.experiment == 1:
            experiment.run_experiment_1_algorithm_comparison()
        elif args.experiment == 2:
            experiment.run_experiment_2_optimal_k()
        elif args.experiment == 3:
            experiment.run_experiment_3_coherence_review(k=args.k)
        elif args.experiment == 4:
            experiment.run_experiment_4_topic_naming(k=args.k)
        elif args.experiment == 5:
            experiment.run_experiment_5_summary_quality(k=args.k)
        elif args.experiment == 6:
            experiment.run_experiment_6_topic_overlap(n_components=args.k)


if __name__ == "__main__":
    main()
