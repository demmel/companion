"""Ablation study for unified retrieval.

Tests the impact of removing individual components:
- Without reference detection
- Without query classification
- Without knowledge graph
- Without episode index
- Without topic clustering

Usage:
    uv run python -m agent.experiments.unified_retrieval.ablation --conversation <id>
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from agent.embedding_service import get_embedding_service

from .build_indices import build_all_indices, load_indices, CACHE_DIR
from .evaluate import load_test_queries, QueryResult, compute_strategy_metrics
from .models import QueryType, UnifiedRetrieverConfig
from .query_classifier import RuleBasedQueryClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Results for a single ablation configuration."""

    name: str
    description: str
    num_queries: int
    classification_accuracy: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_facts_retrieved: float
    avg_memories_retrieved: float
    avg_episodes_retrieved: float


@dataclass
class AblationStudyResult:
    """Complete ablation study results."""

    conversation_id: str
    baseline_result: AblationResult
    ablation_results: list[AblationResult]
    summary: str


def run_ablation_config(
    retriever,
    queries: list[dict],
    config_name: str,
    config_description: str,
) -> AblationResult:
    """Run evaluation with a specific configuration."""
    results: list[QueryResult] = []

    for query_data in queries:
        start_time = time.time()

        context = retriever.retrieve(
            user_input=query_data["query"],
            conversation_context=[],
        )

        latency_ms = (time.time() - start_time) * 1000

        expected_type = query_data["query_type"]
        predicted_type = context.query_type.value

        results.append(
            QueryResult(
                query_id=query_data["id"],
                query_text=query_data["query"],
                expected_type=expected_type,
                predicted_type=predicted_type,
                strategy_used=context.strategy_used,
                num_facts=len(context.facts),
                num_memories=len(context.memories),
                num_episodes=len(context.episodes),
                latency_ms=latency_ms,
                classification_correct=(expected_type == predicted_type),
            )
        )

    latencies = [r.latency_ms for r in results]
    correct = sum(1 for r in results if r.classification_correct)

    return AblationResult(
        name=config_name,
        description=config_description,
        num_queries=len(results),
        classification_accuracy=correct / len(results) if results else 0.0,
        avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        avg_facts_retrieved=float(np.mean([r.num_facts for r in results])),
        avg_memories_retrieved=float(np.mean([r.num_memories for r in results])),
        avg_episodes_retrieved=float(np.mean([r.num_episodes for r in results])),
    )


def run_ablation_study(
    conversation_id: str,
    max_memories: int | None = None,
    use_cached_indices: bool = True,
) -> AblationStudyResult:
    """Run complete ablation study.

    Tests the following configurations:
    1. Full pipeline (baseline)
    2. Without reference detection
    3. Without query classification
    4. Without KG retrieval
    5. Without episode index
    6. Without topic clustering
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    # Build or load indices
    cache_dir = CACHE_DIR / conversation_id
    embedding_service = get_embedding_service()

    if use_cached_indices and cache_dir.exists():
        print(f"Loading cached indices from {cache_dir}")
        kg, memory_index, episode_index, topic_clusters = load_indices(
            cache_dir, embedding_service
        )
    else:
        print("Building indices...")
        retriever = build_all_indices(
            conversation_id=conversation_id,
            max_memories=max_memories,
            output_dir=cache_dir,
        )
        kg = retriever.kg
        memory_index = retriever.memory_index
        episode_index = retriever.episode_index
        topic_clusters = retriever.topic_clusters

    queries = load_test_queries()
    print(f"\nLoaded {len(queries)} test queries")

    ablation_results: list[AblationResult] = []

    # 1. Baseline - full pipeline
    print("\n" + "-" * 70)
    print("Testing: Full pipeline (baseline)")

    from .unified_retriever import UnifiedRetriever

    full_retriever = UnifiedRetriever(
        kg=kg,
        memory_index=memory_index,
        episode_index=episode_index,
        topic_clusters=topic_clusters,
        embedding_service=embedding_service,
        classifier=RuleBasedQueryClassifier(),
        config=UnifiedRetrieverConfig(
            use_reference_detection=True,
            use_query_classification=True,
        ),
    )

    baseline_result = run_ablation_config(
        full_retriever,
        queries,
        "full_pipeline",
        "Full unified retrieval pipeline",
    )
    print(f"  Classification accuracy: {baseline_result.classification_accuracy:.1%}")
    print(f"  Avg latency: {baseline_result.avg_latency_ms:.1f}ms")

    # 2. Without reference detection
    print("\n" + "-" * 70)
    print("Testing: Without reference detection")

    no_ref_retriever = UnifiedRetriever(
        kg=kg,
        memory_index=memory_index,
        episode_index=episode_index,
        topic_clusters=topic_clusters,
        embedding_service=embedding_service,
        classifier=RuleBasedQueryClassifier(),
        config=UnifiedRetrieverConfig(
            use_reference_detection=False,
            use_query_classification=True,
        ),
    )

    no_ref_result = run_ablation_config(
        no_ref_retriever,
        queries,
        "no_reference_detection",
        "Without reference detection",
    )
    ablation_results.append(no_ref_result)
    print(f"  Classification accuracy: {no_ref_result.classification_accuracy:.1%}")
    print(f"  Avg latency: {no_ref_result.avg_latency_ms:.1f}ms")

    # 3. Without query classification
    print("\n" + "-" * 70)
    print("Testing: Without query classification (always proactive)")

    no_class_retriever = UnifiedRetriever(
        kg=kg,
        memory_index=memory_index,
        episode_index=episode_index,
        topic_clusters=topic_clusters,
        embedding_service=embedding_service,
        classifier=RuleBasedQueryClassifier(),
        config=UnifiedRetrieverConfig(
            use_reference_detection=True,
            use_query_classification=False,  # Always use proactive context
        ),
    )

    no_class_result = run_ablation_config(
        no_class_retriever,
        queries,
        "no_query_classification",
        "Without query classification (always proactive)",
    )
    ablation_results.append(no_class_result)
    print(f"  Classification accuracy: {no_class_result.classification_accuracy:.1%}")
    print(f"  Avg latency: {no_class_result.avg_latency_ms:.1f}ms")

    # 4. Without KG (similarity only)
    print("\n" + "-" * 70)
    print("Testing: Without knowledge graph")

    from .unified_retriever import SimpleMemoryIndex
    from agent.experiments.retrieval.knowledge_graph import KnowledgeGraph

    empty_kg = KnowledgeGraph(embedding_service)

    no_kg_retriever = UnifiedRetriever(
        kg=empty_kg,  # Empty KG
        memory_index=memory_index,
        episode_index=episode_index,
        topic_clusters=topic_clusters,
        embedding_service=embedding_service,
        classifier=RuleBasedQueryClassifier(),
        config=UnifiedRetrieverConfig(
            use_reference_detection=True,
            use_query_classification=True,
        ),
    )

    no_kg_result = run_ablation_config(
        no_kg_retriever,
        queries,
        "no_knowledge_graph",
        "Without knowledge graph",
    )
    ablation_results.append(no_kg_result)
    print(f"  Classification accuracy: {no_kg_result.classification_accuracy:.1%}")
    print(f"  Avg latency: {no_kg_result.avg_latency_ms:.1f}ms")

    # 5. Without episode index
    print("\n" + "-" * 70)
    print("Testing: Without episode index")

    from .unified_retriever import SimpleEpisodeIndex

    empty_episode_index = SimpleEpisodeIndex(embedding_service=embedding_service)

    no_episode_retriever = UnifiedRetriever(
        kg=kg,
        memory_index=memory_index,
        episode_index=empty_episode_index,  # Empty
        topic_clusters=topic_clusters,
        embedding_service=embedding_service,
        classifier=RuleBasedQueryClassifier(),
        config=UnifiedRetrieverConfig(
            use_reference_detection=True,
            use_query_classification=True,
        ),
    )

    no_episode_result = run_ablation_config(
        no_episode_retriever,
        queries,
        "no_episode_index",
        "Without episode index",
    )
    ablation_results.append(no_episode_result)
    print(f"  Classification accuracy: {no_episode_result.classification_accuracy:.1%}")
    print(f"  Avg latency: {no_episode_result.avg_latency_ms:.1f}ms")

    # 6. Without topic clustering
    print("\n" + "-" * 70)
    print("Testing: Without topic clustering")

    from .unified_retriever import SimpleTopicClusters

    empty_topic_clusters = SimpleTopicClusters(embedding_service=embedding_service)

    no_topic_retriever = UnifiedRetriever(
        kg=kg,
        memory_index=memory_index,
        episode_index=episode_index,
        topic_clusters=empty_topic_clusters,  # Empty
        embedding_service=embedding_service,
        classifier=RuleBasedQueryClassifier(),
        config=UnifiedRetrieverConfig(
            use_reference_detection=True,
            use_query_classification=True,
        ),
    )

    no_topic_result = run_ablation_config(
        no_topic_retriever,
        queries,
        "no_topic_clustering",
        "Without topic clustering",
    )
    ablation_results.append(no_topic_result)
    print(f"  Classification accuracy: {no_topic_result.classification_accuracy:.1%}")
    print(f"  Avg latency: {no_topic_result.avg_latency_ms:.1f}ms")

    # Generate summary
    summary_lines = [
        "",
        "=" * 70,
        "ABLATION STUDY SUMMARY",
        "=" * 70,
        "",
        f"{'Configuration':<30} {'Class.Acc':>12} {'Latency(ms)':>12} {'Facts':>8} {'Memories':>10}",
        "-" * 70,
        f"{'full_pipeline (baseline)':<30} {baseline_result.classification_accuracy:>11.1%} {baseline_result.avg_latency_ms:>12.1f} {baseline_result.avg_facts_retrieved:>8.1f} {baseline_result.avg_memories_retrieved:>10.1f}",
    ]

    for result in ablation_results:
        acc_delta = result.classification_accuracy - baseline_result.classification_accuracy
        acc_str = f"{result.classification_accuracy:>11.1%}"
        if acc_delta < -0.01:
            acc_str += " (worse)"
        elif acc_delta > 0.01:
            acc_str += " (better)"

        summary_lines.append(
            f"{result.name:<30} {acc_str:>13} {result.avg_latency_ms:>12.1f} {result.avg_facts_retrieved:>8.1f} {result.avg_memories_retrieved:>10.1f}"
        )

    summary_lines.extend([
        "",
        "=" * 70,
        "KEY FINDINGS",
        "=" * 70,
        "",
    ])

    # Find which component has the biggest impact
    impacts = []
    for result in ablation_results:
        impact = baseline_result.classification_accuracy - result.classification_accuracy
        impacts.append((result.name, impact))

    impacts.sort(key=lambda x: x[1], reverse=True)

    for name, impact in impacts:
        direction = "hurts" if impact > 0 else "helps"
        abs_impact = abs(impact) * 100
        if abs_impact > 1:
            summary_lines.append(f"  Removing {name} {direction} accuracy by {abs_impact:.1f}%")

    # Check for latency improvements
    fastest = min(ablation_results + [baseline_result], key=lambda r: r.avg_latency_ms)
    if fastest.name != "full_pipeline":
        speedup = baseline_result.avg_latency_ms - fastest.avg_latency_ms
        summary_lines.append(f"  {fastest.name} is {speedup:.1f}ms faster than baseline")

    summary = "\n".join(summary_lines)
    print(summary)

    return AblationStudyResult(
        conversation_id=conversation_id,
        baseline_result=baseline_result,
        ablation_results=ablation_results,
        summary=summary,
    )


def save_results(result: AblationStudyResult, output_file: Path) -> None:
    """Save results to JSON."""
    data = {
        "conversation_id": result.conversation_id,
        "baseline_result": asdict(result.baseline_result),
        "ablation_results": [asdict(r) for r in result.ablation_results],
        "summary": result.summary,
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run ablation study for unified retrieval"
    )
    parser.add_argument(
        "--conversation",
        type=str,
        required=True,
        help="Conversation ID to use",
    )
    parser.add_argument(
        "--max-memories",
        type=int,
        default=None,
        help="Maximum memories to load",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild indices instead of using cache",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results JSON",
    )

    args = parser.parse_args()

    result = run_ablation_study(
        conversation_id=args.conversation,
        max_memories=args.max_memories,
        use_cached_indices=not args.no_cache,
    )

    if args.output:
        save_results(result, Path(args.output))
    else:
        output_file = CACHE_DIR / f"{args.conversation}_ablation.json"
        save_results(result, output_file)


if __name__ == "__main__":
    main()
