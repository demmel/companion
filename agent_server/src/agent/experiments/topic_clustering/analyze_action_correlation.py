"""
Analyze whether clusters correlate with action types.
"""

import json
from collections import Counter
from pathlib import Path

from agent.conversation_persistence import ConversationPersistence
from agent.memory.dag.dag_memory_manager import DagMemoryManager
from agent.chain_of_action.action.action_types import ActionType

from .clustering import cluster_kmeans, prepare_embeddings


def analyze_action_type_correlation(conversation_prefix: str) -> dict:
    """
    Check if the K=12 clusters correlate with the 12 action types.
    """
    # Load data
    persistence = ConversationPersistence(conversations_dir="conversations")
    agent_data = persistence.load_agent_data(
        conversation_prefix, use_individual_formatting=True
    )

    memory = agent_data.memory
    if not isinstance(memory, DagMemoryManager):
        raise ValueError(f"Expected DagMemoryManager, got {type(memory)}")

    memory_graph = memory.get_memory_graph()

    # Get memories with embeddings
    memories = [
        mem
        for mem in memory_graph.elements.values()
        if mem.embedding_vector is not None
    ]

    # Get containers to map memories to their source triggers
    containers = memory_graph.containers

    # Build mapping: memory_id -> action types in its container
    memory_to_actions: dict[str, list[str]] = {}
    for container in containers.values():
        action_types = (
            [action.type.value for action in container.trigger.actions_taken]
            if hasattr(container.trigger, "actions_taken")
            else []
        )
        # Get action types from the trigger entry's actions
        trigger_actions = (
            [action.type.value for action in container.trigger.actions_taken]
            if hasattr(container.trigger, "actions_taken")
            else []
        )
        for elem_id in container.element_ids:
            memory_to_actions[elem_id] = trigger_actions

    # Run clustering with k=12
    result = cluster_kmeans(memories, k=12)

    # For each cluster, count action types of its memories
    cluster_action_analysis = []

    for cluster in result.clusters:
        action_counter: Counter = Counter()
        memories_with_actions = 0

        for mem_id in cluster.memory_ids:
            if mem_id in memory_to_actions:
                actions = memory_to_actions[mem_id]
                if actions:
                    memories_with_actions += 1
                    for action in actions:
                        action_counter[action] += 1

        # Get dominant action type
        if action_counter:
            dominant_action, dominant_count = action_counter.most_common(1)[0]
            total_actions = sum(action_counter.values())
            dominance_ratio = dominant_count / total_actions
        else:
            dominant_action = "unknown"
            dominant_count = 0
            dominance_ratio = 0

        cluster_action_analysis.append(
            {
                "cluster_id": cluster.id[:8],
                "cluster_size": len(cluster.memory_ids),
                "dominant_action": dominant_action,
                "dominant_count": dominant_count,
                "dominance_ratio": round(dominance_ratio, 3),
                "action_distribution": dict(action_counter.most_common(5)),
            }
        )

    # Check overall correlation
    dominant_actions = [c["dominant_action"] for c in cluster_action_analysis]
    unique_dominant = len(set(dominant_actions))

    return {
        "num_clusters": 12,
        "num_action_types": len(ActionType),
        "unique_dominant_actions_across_clusters": unique_dominant,
        "cluster_analysis": cluster_action_analysis,
        "correlation_likely": unique_dominant
        >= 8,  # If most clusters have different dominant actions
    }


def main():
    result = analyze_action_type_correlation("conversation_20251024_083630_306692")

    print("\n" + "=" * 80)
    print("ACTION TYPE CORRELATION ANALYSIS")
    print("=" * 80)

    print(f"\nNum clusters: {result['num_clusters']}")
    print(f"Num action types: {result['num_action_types']}")
    print(
        f"Unique dominant actions: {result['unique_dominant_actions_across_clusters']}"
    )
    print(f"Correlation likely: {result['correlation_likely']}")

    print("\n" + "-" * 80)
    print("Per-cluster breakdown:")
    print("-" * 80)

    for c in result["cluster_analysis"]:
        print(f"\nCluster {c['cluster_id']} (size={c['cluster_size']}):")
        print(f"  Dominant: {c['dominant_action']} ({c['dominance_ratio']*100:.1f}%)")
        print(f"  Distribution: {c['action_distribution']}")

    # Save results
    output_path = Path(
        "src/agent/experiments/topic_clustering/results/action_correlation_analysis.json"
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
