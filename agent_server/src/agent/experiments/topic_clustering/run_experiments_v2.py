"""
V2 Experiment Runner for Topic Clustering.

This version addresses the methodological issues found in v1:
1. Action-type contamination in embeddings
2. Lack of semantic validation
3. No proper baseline characterization

Run all experiments:
    uv run python -m agent.experiments.topic_clustering.run_experiments_v2 --conversation <prefix>

Run specific experiment:
    uv run python -m agent.experiments.topic_clustering.run_experiments_v2 --conversation <prefix> --experiment 1
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from collections import Counter
import numpy as np
from tqdm import tqdm

from agent.conversation_persistence import ConversationPersistence
from agent.llm import create_llm, SupportedModel
from agent.memory.dag.models import MemoryElement
from agent.memory.dag.dag_memory_manager import DagMemoryManager
from agent.ui_output import ui_print

from .models import ClusteringMethod, TopicNamingApproach
from .clustering import (
    cluster_kmeans,
    prepare_embeddings,
    parse_action_type,
    get_action_types_for_memories,
    compute_action_type_centroids,
    project_orthogonal_to_action_types,
    calculate_action_type_entropy,
    cluster_with_residual_embeddings,
)
from .topic_naming import name_all_clusters
from .evaluation import (
    blind_coherence_evaluation,
    cluster_predictability_test,
    validate_cluster_name,
    calculate_cluster_statistics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TopicClusteringExperimentV2:
    """V2 Experiment orchestrator with improved methodology."""

    def __init__(
        self,
        conversation_prefix: str,
        conversations_dir: str = "conversations",
        output_dir: str = "./topic_clustering_results",
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

        # Get memory manager
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

        # Prepare embeddings once
        self.embeddings, self.memory_ids, self.valid_memories = prepare_embeddings(
            self.memories
        )

        # Parse action types once
        self.action_types, self.action_type_counts = get_action_types_for_memories(
            self.valid_memories
        )
        self.memory_id_to_action_type = {
            mid: atype for mid, atype in zip(self.memory_ids, self.action_types)
        }

        # LLM setup
        self.llm = create_llm()
        self.model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    def run_experiment_1_baseline_characterization(
        self, k: int = 12
    ) -> Dict[str, object]:
        """
        Experiment 1: Baseline Characterization

        Quantify how much action type explains current clustering.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 1: Baseline Characterization")
        ui_print("=" * 80)

        ui_print(f"\nClustering raw embeddings with K={k}...")
        result = cluster_kmeans(self.memories, k=k)

        # Calculate action-type entropy for each cluster
        cluster_entropies: Dict[str, Dict[str, object]] = {}

        for cluster in result.clusters:
            entropy = calculate_action_type_entropy(
                cluster.memory_ids, self.memory_id_to_action_type
            )

            # Get action type distribution for this cluster
            action_counts = Counter(
                self.memory_id_to_action_type.get(mid, "unknown")
                for mid in cluster.memory_ids
            )
            total = len(cluster.memory_ids)
            action_dist = {
                k: {"count": v, "percentage": round(v / total * 100, 1)}
                for k, v in action_counts.most_common(5)
            }

            cluster_entropies[cluster.id] = {
                "size": len(cluster.memory_ids),
                "entropy": round(entropy, 4),
                "dominant_action": (
                    action_counts.most_common(1)[0][0] if action_counts else "unknown"
                ),
                "dominant_percentage": (
                    round(action_counts.most_common(1)[0][1] / total * 100, 1)
                    if action_counts
                    else 0
                ),
                "action_distribution": action_dist,
            }

        # Summary statistics
        entropies = [e["entropy"] for e in cluster_entropies.values()]
        avg_entropy = float(np.mean(entropies))
        min_entropy = float(np.min(entropies))
        max_entropy = float(np.max(entropies))

        results: Dict[str, object] = {
            "k": k,
            "num_memories": len(self.memories),
            "num_action_types": len(self.action_type_counts),
            "action_type_distribution": self.action_type_counts,
            "cluster_analysis": cluster_entropies,
            "entropy_summary": {
                "mean": round(avg_entropy, 4),
                "min": round(min_entropy, 4),
                "max": round(max_entropy, 4),
            },
            "interpretation": self._interpret_baseline_entropy(avg_entropy),
        }

        self._save_results("v2_experiment_1_baseline", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Baseline Characterization Results:")
        ui_print(f"  Average action-type entropy: {avg_entropy:.4f}")
        ui_print(f"  (0 = all same type, 1 = uniformly distributed)")
        ui_print(f"  Entropy range: [{min_entropy:.4f}, {max_entropy:.4f}]")
        ui_print(f"\n  Interpretation: {results['interpretation']}")

        return results

    def _interpret_baseline_entropy(self, avg_entropy: float) -> str:
        """Interpret baseline entropy score."""
        if avg_entropy < 0.3:
            return "LOW entropy - clusters are strongly dominated by action type. Clustering is capturing action-type structure, not semantic topics."
        elif avg_entropy < 0.6:
            return "MODERATE entropy - some action-type influence but also semantic structure. Mixed signal."
        else:
            return "HIGH entropy - clusters are diverse in action types. Clustering may be capturing semantic topics."

    def run_experiment_2_residual_clustering(
        self, k: int = 12, n_components_to_remove: int = 5
    ) -> Dict[str, object]:
        """
        Experiment 2: Action-Residual Clustering

        Project out action-type signal and cluster in residual space.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 2: Action-Residual Clustering")
        ui_print("=" * 80)

        ui_print(f"\nComputing action-type centroids...")
        centroids = compute_action_type_centroids(self.embeddings, self.action_types)

        ui_print(
            f"Projecting out top {n_components_to_remove} action-type components..."
        )
        residual_embeddings = project_orthogonal_to_action_types(
            self.embeddings, centroids, n_components_to_remove
        )

        ui_print(f"Clustering residual embeddings with K={k}...")
        result, cluster_entropies = cluster_with_residual_embeddings(
            self.memories, k=k, n_components_to_remove=n_components_to_remove
        )

        # Build detailed cluster analysis
        cluster_analysis: Dict[str, Dict[str, object]] = {}
        for cluster in result.clusters:
            action_counts = Counter(
                self.memory_id_to_action_type.get(mid, "unknown")
                for mid in cluster.memory_ids
            )
            total = len(cluster.memory_ids)
            action_dist = {
                k: {"count": v, "percentage": round(v / total * 100, 1)}
                for k, v in action_counts.most_common(5)
            }

            cluster_analysis[cluster.id] = {
                "size": len(cluster.memory_ids),
                "entropy": round(cluster_entropies[cluster.id], 4),
                "dominant_action": (
                    action_counts.most_common(1)[0][0] if action_counts else "unknown"
                ),
                "action_distribution": action_dist,
            }

        # Summary
        entropies = list(cluster_entropies.values())
        avg_entropy = float(np.mean(entropies))

        results: Dict[str, object] = {
            "k": k,
            "n_components_removed": n_components_to_remove,
            "silhouette_score": result.silhouette_score,
            "cluster_analysis": cluster_analysis,
            "entropy_summary": {
                "mean": round(avg_entropy, 4),
                "min": round(float(np.min(entropies)), 4),
                "max": round(float(np.max(entropies)), 4),
            },
        }

        self._save_results("v2_experiment_2_residual", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Residual Clustering Results:")
        ui_print(f"  Silhouette score: {result.silhouette_score:.4f}")
        ui_print(f"  Average action-type entropy: {avg_entropy:.4f}")
        ui_print(f"  (Higher entropy = more action-type diversity = better)")

        return results

    def run_experiment_3_semantic_coherence(self, k: int = 12) -> Dict[str, object]:
        """
        Experiment 3: Semantic Coherence Evaluation

        Test if clusters are semantically coherent using blind theme identification.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 3: Semantic Coherence Evaluation")
        ui_print("=" * 80)

        ui_print(f"\nClustering with residual embeddings (K={k})...")
        result, _ = cluster_with_residual_embeddings(self.memories, k=k)

        ui_print("Naming clusters for reference...")
        named_clusters = name_all_clusters(
            clusters=result.clusters,
            memory_graph=self.memory_graph,
            state=self.state,
            llm=self.llm,
            model=self.model,
            approach=TopicNamingApproach.STRUCTURED,
        )

        ui_print("Running blind coherence evaluation on each cluster...")
        coherence_results: List[Dict[str, object]] = []

        for cluster in tqdm(
            named_clusters, desc="Evaluating coherence", file=sys.stdout
        ):
            eval_result = blind_coherence_evaluation(
                cluster=cluster,
                memory_graph=self.memory_graph,
                state=self.state,
                llm=self.llm,
                model=self.model,
                num_samples=3,
                sample_size=5,
            )

            coherence_results.append(
                {
                    "cluster_id": cluster.id[:8],
                    "cluster_name": cluster.name,
                    "cluster_size": len(cluster.memory_ids),
                    "themes_identified": eval_result.get("themes", []),
                    "agreement_score": eval_result.get("agreement_score", 0),
                    "avg_confidence": eval_result.get("avg_confidence", 0),
                }
            )

        # Summary
        agreement_scores = [r["agreement_score"] for r in coherence_results]
        avg_agreement = float(np.mean(agreement_scores)) if agreement_scores else 0

        results: Dict[str, object] = {
            "k": k,
            "cluster_evaluations": coherence_results,
            "summary": {
                "avg_agreement_score": round(avg_agreement, 4),
                "min_agreement": (
                    round(float(np.min(agreement_scores)), 4) if agreement_scores else 0
                ),
                "max_agreement": (
                    round(float(np.max(agreement_scores)), 4) if agreement_scores else 0
                ),
                "clusters_above_0.7": sum(1 for s in agreement_scores if s >= 0.7),
            },
            "interpretation": self._interpret_coherence(avg_agreement),
        }

        self._save_results("v2_experiment_3_coherence", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Semantic Coherence Results:")
        ui_print(f"  Average agreement score: {avg_agreement:.4f}")
        ui_print(
            f"  Clusters with agreement >= 0.7: {results['summary']['clusters_above_0.7']}/{k}"
        )
        ui_print(f"\n  Interpretation: {results['interpretation']}")

        return results

    def _interpret_coherence(self, avg_agreement: float) -> str:
        """Interpret coherence score."""
        if avg_agreement < 0.5:
            return "LOW coherence - independent samples yield different themes. Clusters may not represent meaningful topics."
        elif avg_agreement < 0.7:
            return "MODERATE coherence - some agreement on themes. Clusters have partial semantic structure."
        else:
            return "HIGH coherence - independent samples agree on themes. Clusters likely represent meaningful topics."

    def run_experiment_4_predictability(self, k: int = 12) -> Dict[str, object]:
        """
        Experiment 4: Cluster Predictability (Held-Out Test)

        Test if cluster assignments generalize to held-out data.
        Compare baseline vs residual embeddings.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 4: Cluster Predictability")
        ui_print("=" * 80)

        # Test on raw embeddings
        ui_print("\nTesting predictability on raw embeddings...")
        raw_result = cluster_predictability_test(
            memories=self.valid_memories,
            embeddings=self.embeddings,
            memory_ids=self.memory_ids,
            k=k,
            test_fraction=0.2,
        )

        # Test on residual embeddings
        ui_print("Testing predictability on residual embeddings...")
        centroids = compute_action_type_centroids(self.embeddings, self.action_types)
        residual_embeddings = project_orthogonal_to_action_types(
            self.embeddings, centroids, n_components_to_remove=5
        )

        residual_result = cluster_predictability_test(
            memories=self.valid_memories,
            embeddings=residual_embeddings,
            memory_ids=self.memory_ids,
            k=k,
            test_fraction=0.2,
        )

        results: Dict[str, object] = {
            "k": k,
            "raw_embeddings": {
                "train_size": raw_result.get("train_size", 0),
                "test_size": raw_result.get("test_size", 0),
                "accuracy": round(raw_result.get("assignment_accuracy", 0), 4),
            },
            "residual_embeddings": {
                "train_size": residual_result.get("train_size", 0),
                "test_size": residual_result.get("test_size", 0),
                "accuracy": round(residual_result.get("assignment_accuracy", 0), 4),
            },
            "accuracy_ratio": (
                round(
                    residual_result.get("assignment_accuracy", 0)
                    / raw_result.get("assignment_accuracy", 1),
                    4,
                )
                if raw_result.get("assignment_accuracy", 0) > 0
                else 0
            ),
            "interpretation": self._interpret_predictability(
                raw_result.get("assignment_accuracy", 0),
                residual_result.get("assignment_accuracy", 0),
            ),
        }

        self._save_results("v2_experiment_4_predictability", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Predictability Results:")
        ui_print(
            f"  Raw embedding accuracy: {results['raw_embeddings']['accuracy']:.4f}"
        )
        ui_print(
            f"  Residual embedding accuracy: {results['residual_embeddings']['accuracy']:.4f}"
        )
        ui_print(f"  Accuracy ratio (residual/raw): {results['accuracy_ratio']:.4f}")
        ui_print(f"\n  Interpretation: {results['interpretation']}")

        return results

    def _interpret_predictability(self, raw_acc: float, residual_acc: float) -> str:
        """Interpret predictability results."""
        ratio = residual_acc / raw_acc if raw_acc > 0 else 0

        if ratio > 0.9:
            return "Residual clusters are nearly as predictable as raw. Semantic signal persists after removing action-type variance."
        elif ratio > 0.7:
            return "Some loss in predictability, but residual clusters still learnable. Mixed action-type and semantic signal."
        else:
            return "Large predictability drop in residual space. Raw clustering may have been dominated by action-type structure."

    def run_experiment_5_naming_validation(self, k: int = 12) -> Dict[str, object]:
        """
        Experiment 5: Naming and Interpretability

        Generate cluster names and validate them using LLM classification.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 5: Naming and Interpretability")
        ui_print("=" * 80)

        ui_print(f"\nClustering with residual embeddings (K={k})...")
        result, _ = cluster_with_residual_embeddings(self.memories, k=k)

        ui_print("Naming clusters...")
        named_clusters = name_all_clusters(
            clusters=result.clusters,
            memory_graph=self.memory_graph,
            state=self.state,
            llm=self.llm,
            model=self.model,
            approach=TopicNamingApproach.STRUCTURED,
        )

        ui_print("Validating cluster names...")
        validation_results: List[Dict[str, object]] = []

        for i, cluster in enumerate(
            tqdm(named_clusters, desc="Validating names", file=sys.stdout)
        ):
            other_clusters = [c for j, c in enumerate(named_clusters) if j != i]

            val_result = validate_cluster_name(
                name=cluster.name,
                cluster=cluster,
                other_clusters=other_clusters,
                memory_graph=self.memory_graph,
                state=self.state,
                llm=self.llm,
                model=self.model,
            )

            validation_results.append(
                {
                    "cluster_id": cluster.id[:8],
                    "cluster_name": cluster.name,
                    "cluster_size": len(cluster.memory_ids),
                    "precision": val_result.get("precision", 0),
                    "recall": val_result.get("recall", 0),
                    "f1_score": val_result.get("f1_score", 0),
                }
            )

        # Summary
        precisions = [r["precision"] for r in validation_results]
        recalls = [r["recall"] for r in validation_results]
        f1_scores = [r["f1_score"] for r in validation_results]

        results: Dict[str, object] = {
            "k": k,
            "cluster_validations": validation_results,
            "summary": {
                "avg_precision": round(float(np.mean(precisions)), 4),
                "avg_recall": round(float(np.mean(recalls)), 4),
                "avg_f1": round(float(np.mean(f1_scores)), 4),
                "names_with_f1_above_0.7": sum(1 for f in f1_scores if f >= 0.7),
            },
            "interpretation": self._interpret_naming(float(np.mean(f1_scores))),
        }

        self._save_results("v2_experiment_5_naming", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Naming Validation Results:")
        ui_print(f"  Average precision: {results['summary']['avg_precision']:.4f}")
        ui_print(f"  Average recall: {results['summary']['avg_recall']:.4f}")
        ui_print(f"  Average F1: {results['summary']['avg_f1']:.4f}")
        ui_print(
            f"  Names with F1 >= 0.7: {results['summary']['names_with_f1_above_0.7']}/{k}"
        )
        ui_print(f"\n  Interpretation: {results['interpretation']}")

        return results

    def _interpret_naming(self, avg_f1: float) -> str:
        """Interpret naming validation results."""
        if avg_f1 < 0.5:
            return "LOW name validity - LLM cannot reliably identify cluster members from names. Names may not capture cluster semantics."
        elif avg_f1 < 0.7:
            return "MODERATE name validity - Names partially capture cluster content. Some improvement possible."
        else:
            return "HIGH name validity - Names effectively describe cluster content. LLM can identify members."

    def run_all_experiments(self, k: int = 12) -> Dict[str, object]:
        """Run all v2 experiments in sequence."""
        all_results: Dict[str, object] = {}

        all_results["experiment_1"] = self.run_experiment_1_baseline_characterization(
            k=k
        )
        all_results["experiment_2"] = self.run_experiment_2_residual_clustering(k=k)
        all_results["experiment_3"] = self.run_experiment_3_semantic_coherence(k=k)
        all_results["experiment_4"] = self.run_experiment_4_predictability(k=k)
        all_results["experiment_5"] = self.run_experiment_5_naming_validation(k=k)

        # Save consolidated summary
        self._save_results(
            "v2_all_experiments_summary",
            {
                "conversation_prefix": self.conversation_prefix,
                "total_memories": len(self.memory_graph.elements),
                "memories_with_embeddings": len(self.memories),
                "k": k,
                "timestamp": datetime.now().isoformat(),
                "key_findings": {
                    "baseline_entropy": all_results["experiment_1"]["entropy_summary"][
                        "mean"
                    ],
                    "residual_entropy": all_results["experiment_2"]["entropy_summary"][
                        "mean"
                    ],
                    "coherence_agreement": all_results["experiment_3"]["summary"][
                        "avg_agreement_score"
                    ],
                    "predictability_ratio": all_results["experiment_4"][
                        "accuracy_ratio"
                    ],
                    "naming_f1": all_results["experiment_5"]["summary"]["avg_f1"],
                },
            },
        )

        ui_print("\n" + "=" * 80)
        ui_print("ALL V2 EXPERIMENTS COMPLETE")
        ui_print(f"Results saved to: {self.output_dir}")
        ui_print("=" * 80)

        return all_results

    def _save_results(self, name: str, results: Dict[str, object]) -> None:
        """Save results to JSON file."""
        filepath = self.output_dir / f"{name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run v2 topic clustering experiments")
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
        default="src/agent/experiments/topic_clustering/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific experiment (1-5). If not specified, runs all.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=12,
        help="Number of clusters",
    )

    args = parser.parse_args()

    experiment = TopicClusteringExperimentV2(
        conversation_prefix=args.conversation,
        conversations_dir=args.conversations_dir,
        output_dir=args.output_dir,
    )

    if args.experiment is None:
        experiment.run_all_experiments(k=args.k)
    else:
        if args.experiment == 1:
            experiment.run_experiment_1_baseline_characterization(k=args.k)
        elif args.experiment == 2:
            experiment.run_experiment_2_residual_clustering(k=args.k)
        elif args.experiment == 3:
            experiment.run_experiment_3_semantic_coherence(k=args.k)
        elif args.experiment == 4:
            experiment.run_experiment_4_predictability(k=args.k)
        elif args.experiment == 5:
            experiment.run_experiment_5_naming_validation(k=args.k)


if __name__ == "__main__":
    main()
