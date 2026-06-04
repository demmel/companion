"""
V3 Experiment Runner for Topic Clustering - Cross-Action-Type Graph Clustering.

This version addresses the remaining issues from v2:
1. Clusters still dominated by action type despite residual projection
2. Need to find topics that span across action types

Approach: Build KNN graph, downweight same-action-type edges, use spectral clustering.

Run all experiments:
    uv run python -m agent.experiments.topic_clustering.run_experiments_v3 --conversation <prefix>

Run specific experiment:
    uv run python -m agent.experiments.topic_clustering.run_experiments_v3 --conversation <prefix> --experiment 1
"""

import argparse
import json
import logging
import random
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

from .models import TopicNamingApproach
from .clustering import (
    cluster_kmeans,
    prepare_embeddings,
    get_action_types_from_trigger_history,
    calculate_action_type_entropy,
    cluster_cross_action_type,
)
from .topic_naming import name_all_clusters
from .evaluation import blind_coherence_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TopicClusteringExperimentV3:
    """V3 Experiment orchestrator with cross-action-type graph clustering."""

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
        self.trigger_history = agent_data.trigger_history

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

        # Get action types from trigger history (accurate lookup)
        self.action_types, self.action_type_counts = (
            get_action_types_from_trigger_history(
                self.valid_memories, self.trigger_history
            )
        )
        self.memory_id_to_action_type = {
            mid: atype for mid, atype in zip(self.memory_ids, self.action_types)
        }

        # LLM setup
        self.llm = create_llm()
        self.model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    def run_experiment_1_baseline_vs_cross_action(
        self, n_clusters: int = 12, k_neighbors: int = 15, same_type_weight: float = 0.1
    ) -> Dict[str, object]:
        """
        Experiment 1: Baseline vs Cross-Action Clustering

        Compare K-Means on raw embeddings vs graph-based cross-action clustering.
        Measure action-type entropy for both approaches.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 1: Baseline vs Cross-Action-Type Clustering")
        ui_print("=" * 80)

        # Baseline: K-Means on raw embeddings
        ui_print(f"\nRunning baseline K-Means (K={n_clusters})...")
        baseline_result = cluster_kmeans(self.memories, k=n_clusters)

        baseline_entropies: Dict[str, float] = {}
        baseline_action_dists: Dict[str, Dict[str, int]] = {}
        for cluster in baseline_result.clusters:
            entropy = calculate_action_type_entropy(
                cluster.memory_ids, self.memory_id_to_action_type
            )
            baseline_entropies[cluster.id] = entropy

            action_counts: Dict[str, int] = {}
            for mid in cluster.memory_ids:
                atype = self.memory_id_to_action_type.get(mid, "unknown")
                action_counts[atype] = action_counts.get(atype, 0) + 1
            baseline_action_dists[cluster.id] = action_counts

        baseline_avg_entropy = float(np.mean(list(baseline_entropies.values())))
        baseline_types_per_cluster = [len(d) for d in baseline_action_dists.values()]

        # Cross-action-type clustering
        ui_print(
            f"\nRunning cross-action-type clustering (K={n_clusters}, k_neighbors={k_neighbors}, weight={same_type_weight})..."
        )
        cross_result, cross_entropies, cross_action_dists = cluster_cross_action_type(
            self.memories,
            n_clusters=n_clusters,
            k_neighbors=k_neighbors,
            same_type_weight=same_type_weight,
            trigger_history=self.trigger_history,
        )

        cross_avg_entropy = float(np.mean(list(cross_entropies.values())))
        cross_types_per_cluster = [len(d) for d in cross_action_dists.values()]

        # Build cluster analysis for both
        baseline_analysis: List[Dict[str, object]] = []
        for cluster in baseline_result.clusters:
            total = len(cluster.memory_ids)
            dist = baseline_action_dists[cluster.id]
            baseline_analysis.append(
                {
                    "cluster_id": cluster.id[:8],
                    "size": total,
                    "entropy": round(baseline_entropies[cluster.id], 4),
                    "num_action_types": len(dist),
                    "action_distribution": {
                        k: {"count": v, "percentage": round(v / total * 100, 1)}
                        for k, v in sorted(dist.items(), key=lambda x: -x[1])[:5]
                    },
                }
            )

        cross_analysis: List[Dict[str, object]] = []
        for cluster in cross_result.clusters:
            total = len(cluster.memory_ids)
            dist = cross_action_dists[cluster.id]
            cross_analysis.append(
                {
                    "cluster_id": cluster.id[:8],
                    "size": total,
                    "entropy": round(cross_entropies[cluster.id], 4),
                    "num_action_types": len(dist),
                    "action_distribution": {
                        k: {"count": v, "percentage": round(v / total * 100, 1)}
                        for k, v in sorted(dist.items(), key=lambda x: -x[1])[:5]
                    },
                }
            )

        results: Dict[str, object] = {
            "n_clusters": n_clusters,
            "k_neighbors": k_neighbors,
            "same_type_weight": same_type_weight,
            "num_memories": len(self.memories),
            "action_type_counts": self.action_type_counts,
            "baseline": {
                "method": "kmeans_raw",
                "avg_entropy": round(baseline_avg_entropy, 4),
                "avg_action_types_per_cluster": round(
                    float(np.mean(baseline_types_per_cluster)), 2
                ),
                "silhouette_score": round(baseline_result.silhouette_score, 4),
                "clusters": baseline_analysis,
            },
            "cross_action": {
                "method": "cross_action_type_spectral",
                "avg_entropy": round(cross_avg_entropy, 4),
                "avg_action_types_per_cluster": round(
                    float(np.mean(cross_types_per_cluster)), 2
                ),
                "silhouette_score": round(cross_result.silhouette_score, 4),
                "clusters": cross_analysis,
            },
            "improvement": {
                "entropy_increase": round(cross_avg_entropy - baseline_avg_entropy, 4),
                "entropy_ratio": (
                    round(cross_avg_entropy / baseline_avg_entropy, 2)
                    if baseline_avg_entropy > 0
                    else 0
                ),
                "types_increase": round(
                    float(np.mean(cross_types_per_cluster))
                    - float(np.mean(baseline_types_per_cluster)),
                    2,
                ),
            },
        }

        self._save_results("v3_experiment_1_comparison", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Baseline vs Cross-Action-Type Results:")
        ui_print(f"\n  BASELINE (K-Means):")
        ui_print(f"    Avg entropy: {baseline_avg_entropy:.4f}")
        ui_print(
            f"    Avg action types per cluster: {np.mean(baseline_types_per_cluster):.1f}"
        )
        ui_print(f"\n  CROSS-ACTION-TYPE:")
        ui_print(f"    Avg entropy: {cross_avg_entropy:.4f}")
        ui_print(
            f"    Avg action types per cluster: {np.mean(cross_types_per_cluster):.1f}"
        )
        ui_print(f"\n  IMPROVEMENT:")
        ui_print(
            f"    Entropy increase: {results['improvement']['entropy_increase']:.4f}"
        )
        ui_print(f"    Entropy ratio: {results['improvement']['entropy_ratio']:.2f}x")

        return results

    def run_experiment_2_weight_sensitivity(
        self, n_clusters: int = 12, k_neighbors: int = 15
    ) -> Dict[str, object]:
        """
        Experiment 2: Weight Sensitivity Analysis

        Test different same_type_weight values to find optimal balance.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 2: Weight Sensitivity Analysis")
        ui_print("=" * 80)

        weight_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
        weight_results: List[Dict[str, object]] = []

        for weight in tqdm(weight_values, desc="Testing weights", file=sys.stdout):
            ui_print(f"\n  Testing same_type_weight={weight}...")

            result, entropies, action_dists = cluster_cross_action_type(
                self.memories,
                n_clusters=n_clusters,
                k_neighbors=k_neighbors,
                same_type_weight=weight,
                trigger_history=self.trigger_history,
            )

            avg_entropy = float(np.mean(list(entropies.values())))
            types_per_cluster = [len(d) for d in action_dists.values()]

            weight_results.append(
                {
                    "weight": weight,
                    "avg_entropy": round(avg_entropy, 4),
                    "min_entropy": round(float(np.min(list(entropies.values()))), 4),
                    "max_entropy": round(float(np.max(list(entropies.values()))), 4),
                    "avg_action_types": round(float(np.mean(types_per_cluster)), 2),
                    "silhouette_score": round(result.silhouette_score, 4),
                }
            )

        # Find optimal weight (highest entropy while maintaining reasonable silhouette)
        # Prioritize entropy but penalize very low silhouette
        scores = []
        for r in weight_results:
            # Combined score: entropy * (silhouette penalty)
            sil_penalty = 1.0 if r["silhouette_score"] > 0.05 else 0.5
            scores.append(r["avg_entropy"] * sil_penalty)

        optimal_idx = int(np.argmax(scores))
        optimal_weight = weight_values[optimal_idx]

        results: Dict[str, object] = {
            "n_clusters": n_clusters,
            "k_neighbors": k_neighbors,
            "weight_analysis": weight_results,
            "optimal_weight": optimal_weight,
            "recommendation": self._interpret_weight_sensitivity(
                weight_results, optimal_weight
            ),
        }

        self._save_results("v3_experiment_2_weight_sensitivity", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Weight Sensitivity Results:")
        for r in weight_results:
            marker = " <- OPTIMAL" if r["weight"] == optimal_weight else ""
            ui_print(
                f"  weight={r['weight']:.2f}: entropy={r['avg_entropy']:.4f}, "
                f"silhouette={r['silhouette_score']:.4f}{marker}"
            )
        ui_print(f"\n  Recommendation: {results['recommendation']}")

        return results

    def _interpret_weight_sensitivity(
        self, results: List[Dict[str, object]], optimal: float
    ) -> str:
        """Interpret weight sensitivity results."""
        if optimal == 0.0:
            return "Completely removing same-type edges works best. Strong action-type separation in raw embeddings."
        elif optimal <= 0.1:
            return f"Low weight ({optimal}) is optimal. Heavy downweighting needed to overcome action-type dominance."
        elif optimal <= 0.5:
            return f"Moderate weight ({optimal}) is optimal. Some same-type structure is informative."
        else:
            return "High weight works best. Action-type signal may already be weak, or graph structure is more important."

    def run_experiment_3_coherence(
        self, n_clusters: int = 12, k_neighbors: int = 15, same_type_weight: float = 0.1
    ) -> Dict[str, object]:
        """
        Experiment 3: Semantic Coherence Comparison

        Run blind coherence evaluation on cross-action-type clusters.
        Compare to v2 results.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 3: Semantic Coherence of Cross-Action Clusters")
        ui_print("=" * 80)

        ui_print(f"\nRunning cross-action-type clustering...")
        result, entropies, action_dists = cluster_cross_action_type(
            self.memories,
            n_clusters=n_clusters,
            k_neighbors=k_neighbors,
            same_type_weight=same_type_weight,
            trigger_history=self.trigger_history,
        )

        ui_print("Naming clusters for reference...")
        named_clusters = name_all_clusters(
            clusters=result.clusters,
            memory_graph=self.memory_graph,
            state=self.state,
            llm=self.llm,
            model=self.model,
            approach=TopicNamingApproach.STRUCTURED,
        )

        ui_print("Running blind coherence evaluation...")
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

            # Get action distribution for this cluster
            action_dist = action_dists.get(cluster.id, {})
            total = len(cluster.memory_ids)

            coherence_results.append(
                {
                    "cluster_id": cluster.id[:8],
                    "cluster_name": cluster.name,
                    "cluster_size": total,
                    "entropy": round(entropies.get(cluster.id, 0), 4),
                    "num_action_types": len(action_dist),
                    "themes_identified": eval_result.get("themes", []),
                    "agreement_score": eval_result.get("agreement_score", 0),
                    "action_type_breakdown": (
                        {
                            k: round(v / total * 100, 1)
                            for k, v in sorted(
                                action_dist.items(), key=lambda x: -x[1]
                            )[:5]
                        }
                        if action_dist
                        else {}
                    ),
                }
            )

        # Summary
        agreement_scores = [r["agreement_score"] for r in coherence_results]
        avg_agreement = float(np.mean(agreement_scores)) if agreement_scores else 0

        results: Dict[str, object] = {
            "n_clusters": n_clusters,
            "k_neighbors": k_neighbors,
            "same_type_weight": same_type_weight,
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
                "avg_entropy": round(float(np.mean(list(entropies.values()))), 4),
            },
            "interpretation": self._interpret_coherence(avg_agreement),
        }

        self._save_results("v3_experiment_3_coherence", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Semantic Coherence Results:")
        ui_print(f"  Average agreement score: {avg_agreement:.4f}")
        ui_print(
            f"  Clusters with agreement >= 0.7: {results['summary']['clusters_above_0.7']}/{n_clusters}"
        )
        ui_print(f"  Average entropy: {results['summary']['avg_entropy']:.4f}")
        ui_print(f"\n  Interpretation: {results['interpretation']}")

        return results

    def _interpret_coherence(self, avg_agreement: float) -> str:
        """Interpret coherence score."""
        if avg_agreement < 0.5:
            return "LOW coherence - cross-action clusters may be mixing unrelated memories."
        elif avg_agreement < 0.7:
            return "MODERATE coherence - some semantic structure, but cross-type mixing may dilute themes."
        else:
            return "HIGH coherence - cross-action clusters successfully find themes that span action types."

    def run_experiment_4_topic_inspection(
        self,
        n_clusters: int = 12,
        k_neighbors: int = 15,
        same_type_weight: float = 0.1,
        samples_per_type: int = 2,
    ) -> Dict[str, object]:
        """
        Experiment 4: Cross-Action Topic Inspection

        For each cluster, sample memories from each action type to verify
        they share semantic themes.
        """
        ui_print("\n" + "=" * 80)
        ui_print("EXPERIMENT 4: Cross-Action Topic Inspection")
        ui_print("=" * 80)

        ui_print(f"\nRunning cross-action-type clustering...")
        result, entropies, action_dists = cluster_cross_action_type(
            self.memories,
            n_clusters=n_clusters,
            k_neighbors=k_neighbors,
            same_type_weight=same_type_weight,
            trigger_history=self.trigger_history,
        )

        ui_print("Naming clusters...")
        named_clusters = name_all_clusters(
            clusters=result.clusters,
            memory_graph=self.memory_graph,
            state=self.state,
            llm=self.llm,
            model=self.model,
            approach=TopicNamingApproach.STRUCTURED,
        )

        # Build memory_id -> memory mapping
        memory_by_id = {m.id: m for m in self.valid_memories}

        cluster_inspections: List[Dict[str, object]] = []

        for cluster in named_clusters:
            # Group cluster memories by action type
            memories_by_type: Dict[str, List[str]] = {}
            for mid in cluster.memory_ids:
                atype = self.memory_id_to_action_type.get(mid, "unknown")
                if atype not in memories_by_type:
                    memories_by_type[atype] = []
                memories_by_type[atype].append(mid)

            # Sample from each action type
            type_samples: Dict[str, List[Dict[str, str]]] = {}
            for atype, mids in memories_by_type.items():
                sampled_ids = random.sample(mids, min(samples_per_type, len(mids)))
                type_samples[atype] = [
                    {
                        "id": mid[:8],
                        "content": (
                            memory_by_id[mid].content[:200] + "..."
                            if len(memory_by_id[mid].content) > 200
                            else memory_by_id[mid].content
                        ),
                    }
                    for mid in sampled_ids
                    if mid in memory_by_id
                ]

            cluster_inspections.append(
                {
                    "cluster_id": cluster.id[:8],
                    "cluster_name": cluster.name,
                    "size": len(cluster.memory_ids),
                    "entropy": round(entropies.get(cluster.id, 0), 4),
                    "action_types_present": list(memories_by_type.keys()),
                    "num_action_types": len(memories_by_type),
                    "action_type_counts": {
                        k: len(v) for k, v in memories_by_type.items()
                    },
                    "samples_by_type": type_samples,
                }
            )

        # Summary statistics
        num_types_list = [c["num_action_types"] for c in cluster_inspections]

        results: Dict[str, object] = {
            "n_clusters": n_clusters,
            "k_neighbors": k_neighbors,
            "same_type_weight": same_type_weight,
            "samples_per_type": samples_per_type,
            "cluster_inspections": cluster_inspections,
            "summary": {
                "avg_action_types_per_cluster": round(
                    float(np.mean(num_types_list)), 2
                ),
                "min_action_types": int(np.min(num_types_list)),
                "max_action_types": int(np.max(num_types_list)),
                "clusters_with_3plus_types": sum(1 for n in num_types_list if n >= 3),
            },
        }

        self._save_results("v3_experiment_4_inspection", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Cross-Action Topic Inspection Results:")
        ui_print(
            f"  Avg action types per cluster: {results['summary']['avg_action_types_per_cluster']:.1f}"
        )
        ui_print(
            f"  Clusters with 3+ action types: {results['summary']['clusters_with_3plus_types']}/{n_clusters}"
        )
        ui_print("\n  Cluster breakdown:")
        for c in cluster_inspections[:5]:  # Show top 5
            ui_print(
                f"    '{c['cluster_name'][:40]}': {c['num_action_types']} types, entropy={c['entropy']:.2f}"
            )

        return results

    def run_all_experiments(
        self, n_clusters: int = 12, k_neighbors: int = 15, same_type_weight: float = 0.1
    ) -> Dict[str, object]:
        """Run all v3 experiments in sequence."""
        all_results: Dict[str, object] = {}

        all_results["experiment_1"] = self.run_experiment_1_baseline_vs_cross_action(
            n_clusters=n_clusters,
            k_neighbors=k_neighbors,
            same_type_weight=same_type_weight,
        )
        all_results["experiment_2"] = self.run_experiment_2_weight_sensitivity(
            n_clusters=n_clusters, k_neighbors=k_neighbors
        )
        all_results["experiment_3"] = self.run_experiment_3_coherence(
            n_clusters=n_clusters,
            k_neighbors=k_neighbors,
            same_type_weight=same_type_weight,
        )
        all_results["experiment_4"] = self.run_experiment_4_topic_inspection(
            n_clusters=n_clusters,
            k_neighbors=k_neighbors,
            same_type_weight=same_type_weight,
        )

        # Save consolidated summary
        self._save_results(
            "v3_all_experiments_summary",
            {
                "conversation_prefix": self.conversation_prefix,
                "total_memories": len(self.memory_graph.elements),
                "memories_with_embeddings": len(self.memories),
                "n_clusters": n_clusters,
                "k_neighbors": k_neighbors,
                "same_type_weight": same_type_weight,
                "timestamp": datetime.now().isoformat(),
                "key_findings": {
                    "baseline_entropy": all_results["experiment_1"]["baseline"][
                        "avg_entropy"
                    ],
                    "cross_action_entropy": all_results["experiment_1"]["cross_action"][
                        "avg_entropy"
                    ],
                    "entropy_improvement": all_results["experiment_1"]["improvement"][
                        "entropy_ratio"
                    ],
                    "optimal_weight": all_results["experiment_2"]["optimal_weight"],
                    "coherence_agreement": all_results["experiment_3"]["summary"][
                        "avg_agreement_score"
                    ],
                    "avg_types_per_cluster": all_results["experiment_4"]["summary"][
                        "avg_action_types_per_cluster"
                    ],
                },
            },
        )

        ui_print("\n" + "=" * 80)
        ui_print("ALL V3 EXPERIMENTS COMPLETE")
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
    parser = argparse.ArgumentParser(description="Run v3 topic clustering experiments")
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
        choices=[1, 2, 3, 4],
        help="Run specific experiment (1-4). If not specified, runs all.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=12,
        help="Number of clusters",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for KNN graph",
    )
    parser.add_argument(
        "--same-type-weight",
        type=float,
        default=0.1,
        help="Weight multiplier for same-action-type edges (0.0-1.0)",
    )

    args = parser.parse_args()

    experiment = TopicClusteringExperimentV3(
        conversation_prefix=args.conversation,
        conversations_dir=args.conversations_dir,
        output_dir=args.output_dir,
    )

    if args.experiment is None:
        experiment.run_all_experiments(
            n_clusters=args.n_clusters,
            k_neighbors=args.k_neighbors,
            same_type_weight=args.same_type_weight,
        )
    else:
        if args.experiment == 1:
            experiment.run_experiment_1_baseline_vs_cross_action(
                n_clusters=args.n_clusters,
                k_neighbors=args.k_neighbors,
                same_type_weight=args.same_type_weight,
            )
        elif args.experiment == 2:
            experiment.run_experiment_2_weight_sensitivity(
                n_clusters=args.n_clusters,
                k_neighbors=args.k_neighbors,
            )
        elif args.experiment == 3:
            experiment.run_experiment_3_coherence(
                n_clusters=args.n_clusters,
                k_neighbors=args.k_neighbors,
                same_type_weight=args.same_type_weight,
            )
        elif args.experiment == 4:
            experiment.run_experiment_4_topic_inspection(
                n_clusters=args.n_clusters,
                k_neighbors=args.k_neighbors,
                same_type_weight=args.same_type_weight,
            )


if __name__ == "__main__":
    main()
