"""
V4 Experiment Runner for Topic Clustering - Cross-Action-Only Similarity.

This version addresses the fundamental issue from v3:
- v3 built a KNN graph using cosine similarity, then tried to downweight same-type edges
- But the graph was already dominated by same-type connections
- v4 ONLY computes similarity between different action types, never within

Approach: Build affinity matrix with 0 for same-type pairs, use Louvain community detection.

Run experiments:
    uv run python -m agent.experiments.topic_clustering.run_experiments_v4 --conversation <prefix>

Run with specific parameters:
    uv run python -m agent.experiments.topic_clustering.run_experiments_v4 --conversation <prefix> --method louvain --k 15
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from agent.conversation_persistence import ConversationPersistence
from agent.llm import create_llm, SupportedModel
from agent.memory.dag.models import MemoryElement
from agent.memory.dag.dag_memory_manager import DagMemoryManager
from agent.ui_output import ui_print

from .models import TopicNamingApproach
from .clustering import (
    prepare_embeddings,
    get_action_types_from_trigger_history,
    calculate_action_type_entropy,
    cluster_cross_action_only,
)
from .topic_naming import name_all_clusters
from .evaluation import blind_coherence_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TopicClusteringExperimentV4:
    """V4 Experiment orchestrator with cross-action-only similarity clustering."""

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

        logger.info(
            f"Found {len(self.action_type_counts)} unique action types: {self.action_type_counts}"
        )

        # LLM setup
        self.llm = create_llm()
        self.model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    def _save_results(self, name: str, results: Dict[str, object]) -> None:
        """Save experiment results to JSON file."""
        filepath = self.output_dir / f"{name}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {filepath}")

    def run_clustering(
        self,
        method: str = "louvain",
        k_neighbors: int = 15,
        resolution: float = 1.0,
        n_clusters: int = 12,
    ) -> Dict[str, object]:
        """
        Run cross-action-only clustering.

        Args:
            method: "louvain" or "spectral"
            k_neighbors: Number of cross-action-type nearest neighbors per memory
            resolution: Louvain resolution parameter (higher = more clusters)
            n_clusters: Number of clusters for spectral method

        Returns:
            Clustering results with cluster assignments and metrics
        """
        ui_print("\n" + "=" * 80)
        ui_print("V4: Cross-Action-Only KNN Clustering")
        ui_print("=" * 80)
        ui_print(f"\nMethod: {method}")
        ui_print(f"K neighbors: {k_neighbors}")
        if method == "louvain":
            ui_print(f"Resolution: {resolution}")
        else:
            ui_print(f"N clusters: {n_clusters}")

        ui_print(f"\nRunning cross-action-only clustering...")
        result, entropies, action_dists = cluster_cross_action_only(
            self.memories,
            trigger_history=self.trigger_history,
            method=method,
            n_clusters=n_clusters,
            k_neighbors=k_neighbors,
            resolution=resolution,
        )

        # Summary statistics
        num_clusters = len(result.clusters)
        cluster_sizes = [len(c.memory_ids) for c in result.clusters]
        entropy_values = list(entropies.values())
        types_per_cluster = [len(d) for d in action_dists.values()]

        ui_print(f"\n" + "-" * 40)
        ui_print(f"Clustering Results:")
        ui_print(f"  Number of clusters: {num_clusters}")
        ui_print(
            f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, median={np.median(cluster_sizes):.0f}"
        )
        ui_print(f"  Average entropy: {np.mean(entropy_values):.4f}")
        ui_print(f"  Avg action types per cluster: {np.mean(types_per_cluster):.1f}")

        # Build cluster details
        cluster_details: List[Dict[str, object]] = []
        for cluster in result.clusters:
            action_dist = action_dists.get(cluster.id, {})
            total = len(cluster.memory_ids)
            cluster_details.append(
                {
                    "cluster_id": cluster.id[:8],
                    "size": total,
                    "entropy": round(entropies.get(cluster.id, 0), 4),
                    "num_action_types": len(action_dist),
                    "action_distribution": {
                        k: {"count": v, "percentage": round(v / total * 100, 1)}
                        for k, v in sorted(action_dist.items(), key=lambda x: -x[1])
                    },
                }
            )

        # Sort by size descending
        cluster_details.sort(key=lambda x: -x["size"])

        results: Dict[str, object] = {
            "method": method,
            "k_neighbors": k_neighbors,
            "resolution": resolution if method == "louvain" else None,
            "n_clusters_param": n_clusters if method == "spectral" else None,
            "num_memories": len(self.valid_memories),
            "action_type_counts": self.action_type_counts,
            "num_clusters": num_clusters,
            "summary": {
                "avg_entropy": round(float(np.mean(entropy_values)), 4),
                "min_entropy": round(float(np.min(entropy_values)), 4),
                "max_entropy": round(float(np.max(entropy_values)), 4),
                "avg_types_per_cluster": round(float(np.mean(types_per_cluster)), 2),
                "cluster_size_distribution": {
                    "min": int(min(cluster_sizes)),
                    "max": int(max(cluster_sizes)),
                    "median": int(np.median(cluster_sizes)),
                    "mean": round(float(np.mean(cluster_sizes)), 1),
                },
            },
            "clusters": cluster_details,
        }

        self._save_results("v4_clustering", results)

        return results

    def run_coherence_evaluation(
        self,
        method: str = "louvain",
        k_neighbors: int = 15,
        resolution: float = 1.0,
        n_clusters: int = 12,
        num_samples: int = 3,
        sample_size: int = 5,
    ) -> Dict[str, object]:
        """
        Run clustering and evaluate semantic coherence of clusters.

        Uses blind coherence evaluation: sample memories from each cluster,
        ask LLM to identify themes, check if themes agree.
        """
        ui_print("\n" + "=" * 80)
        ui_print("V4: Coherence Evaluation of Cross-Action-Only Clusters")
        ui_print("=" * 80)

        ui_print(f"\nRunning cross-action-only clustering...")
        result, entropies, action_dists = cluster_cross_action_only(
            self.memories,
            trigger_history=self.trigger_history,
            method=method,
            n_clusters=n_clusters,
            k_neighbors=k_neighbors,
            resolution=resolution,
        )

        ui_print(f"Found {len(result.clusters)} clusters")

        ui_print("\nNaming clusters for reference...")
        named_clusters = name_all_clusters(
            clusters=result.clusters,
            memory_graph=self.memory_graph,
            state=self.state,
            llm=self.llm,
            model=self.model,
            approach=TopicNamingApproach.STRUCTURED,
        )

        ui_print("\nRunning blind coherence evaluation...")
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
                num_samples=num_samples,
                sample_size=sample_size,
            )

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
        entropy_values = [r["entropy"] for r in coherence_results]

        results: Dict[str, object] = {
            "method": method,
            "k_neighbors": k_neighbors,
            "resolution": resolution if method == "louvain" else None,
            "num_clusters": len(result.clusters),
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
                "clusters_above_0.5": sum(1 for s in agreement_scores if s >= 0.5),
                "avg_entropy": round(float(np.mean(entropy_values)), 4),
            },
            "interpretation": self._interpret_coherence(avg_agreement),
        }

        self._save_results("v4_coherence", results)

        # Print summary
        ui_print("\n" + "-" * 40)
        ui_print("Coherence Results:")
        ui_print(f"  Average agreement score: {avg_agreement:.4f}")
        ui_print(
            f"  Clusters with agreement >= 0.7: {results['summary']['clusters_above_0.7']}/{len(result.clusters)}"
        )
        ui_print(
            f"  Clusters with agreement >= 0.5: {results['summary']['clusters_above_0.5']}/{len(result.clusters)}"
        )
        ui_print(f"  Average entropy: {results['summary']['avg_entropy']:.4f}")
        ui_print(f"\n  Interpretation: {results['interpretation']}")

        return results

    def _interpret_coherence(self, avg_agreement: float) -> str:
        """Interpret coherence score."""
        if avg_agreement < 0.5:
            return "LOW coherence - clusters may be mixing unrelated memories despite cross-action structure."
        elif avg_agreement < 0.7:
            return "MODERATE coherence - some semantic structure found across action types."
        else:
            return "HIGH coherence - successfully finding topics that span multiple action types."

    def run_topic_inspection(
        self,
        method: str = "louvain",
        k_neighbors: int = 15,
        resolution: float = 1.0,
        n_clusters: int = 12,
        samples_per_type: int = 2,
    ) -> Dict[str, object]:
        """
        Inspect clusters by sampling memories from each action type.

        This helps manually verify that clusters contain semantically related
        memories across different action types.
        """
        ui_print("\n" + "=" * 80)
        ui_print("V4: Topic Inspection of Cross-Action-Only Clusters")
        ui_print("=" * 80)

        ui_print(f"\nRunning cross-action-only clustering...")
        result, entropies, action_dists = cluster_cross_action_only(
            self.memories,
            trigger_history=self.trigger_history,
            method=method,
            n_clusters=n_clusters,
            k_neighbors=k_neighbors,
            resolution=resolution,
        )

        ui_print("\nNaming clusters...")
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

        # Sort by size descending
        cluster_inspections.sort(key=lambda x: -x["size"])

        results: Dict[str, object] = {
            "method": method,
            "k_neighbors": k_neighbors,
            "resolution": resolution if method == "louvain" else None,
            "samples_per_type": samples_per_type,
            "num_clusters": len(result.clusters),
            "cluster_inspections": cluster_inspections,
        }

        self._save_results("v4_inspection", results)

        ui_print(f"\nSaved inspection results with {len(cluster_inspections)} clusters")
        ui_print(f"Top 5 clusters by size:")
        for c in cluster_inspections[:5]:
            ui_print(
                f"  - {c['cluster_name']}: {c['size']} memories, {c['num_action_types']} action types"
            )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="V4 Topic Clustering - Cross-Action-Only Similarity"
    )
    parser.add_argument(
        "--conversation",
        required=True,
        help="Conversation ID prefix to load",
    )
    parser.add_argument(
        "--conversations-dir",
        default="conversations",
        help="Directory containing conversation files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "output" / "results"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--method",
        choices=["louvain", "spectral"],
        default="louvain",
        help="Clustering method (default: louvain)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="Number of cross-action-type nearest neighbors per memory (default: 15)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Louvain resolution parameter (default: 1.0)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=12,
        help="Number of clusters for spectral method (default: 12)",
    )
    parser.add_argument(
        "--experiment",
        choices=["clustering", "coherence", "inspection", "all"],
        default="all",
        help="Which experiment to run (default: all)",
    )

    args = parser.parse_args()

    experiment = TopicClusteringExperimentV4(
        conversation_prefix=args.conversation,
        conversations_dir=args.conversations_dir,
        output_dir=args.output_dir,
    )

    if args.experiment in ["clustering", "all"]:
        experiment.run_clustering(
            method=args.method,
            k_neighbors=args.k,
            resolution=args.resolution,
            n_clusters=args.n_clusters,
        )

    if args.experiment in ["coherence", "all"]:
        experiment.run_coherence_evaluation(
            method=args.method,
            k_neighbors=args.k,
            resolution=args.resolution,
            n_clusters=args.n_clusters,
        )

    if args.experiment in ["inspection", "all"]:
        experiment.run_topic_inspection(
            method=args.method,
            k_neighbors=args.k,
            resolution=args.resolution,
            n_clusters=args.n_clusters,
        )

    ui_print("\n" + "=" * 80)
    ui_print("V4 Experiments Complete!")
    ui_print("=" * 80)


if __name__ == "__main__":
    main()
