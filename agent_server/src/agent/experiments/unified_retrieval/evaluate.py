"""Evaluation runner for the unified retrieval experiment.

Uses proper IR metrics (precision, recall, F1, MRR) against ground truth.

The key insight: evaluation must answer "Did we retrieve the right content?"
NOT "Can we answer the query?" (circular) or "Is the response better?" (subjective)

Usage:
    uv run python -m agent.experiments.unified_retrieval.evaluate \
        --dataset test_queries_groundtruth.json \
        --conversation <id>
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from agent.embedding_service import get_embedding_service

from .build_indices import build_all_indices, load_indices, CACHE_DIR
from .models import QueryType, RetrievalContext
from .query_classifier import RuleBasedQueryClassifier
from .unified_retriever import UnifiedRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class GroundTruthQuery:
    """A test query with ground truth for evaluation."""

    id: str
    query: str
    query_type: str
    expected_memory_ids: list[str] = field(default_factory=list)
    expected_facts: list[dict[str, str]] = field(default_factory=list)
    expected_entity: str | None = None
    expected_attribute: str | None = None
    confidence: float = 0.0
    needs_review: bool = True
    notes: str = ""


@dataclass
class RetrievalMetrics:
    """IR metrics for a single query."""

    precision: float  # What % of retrieved items are in expected set
    recall: float  # What % of expected items were retrieved
    f1: float  # Harmonic mean of precision and recall
    mrr: float  # Mean Reciprocal Rank (1/rank of first correct answer)
    num_retrieved: int
    num_expected: int
    num_correct: int


@dataclass
class QueryResult:
    """Result for a single query evaluation."""

    query_id: str
    query_text: str
    query_type: str
    strategy: str

    # What was retrieved
    retrieved_memory_ids: list[str]
    expected_memory_ids: list[str]

    # Metrics
    metrics: RetrievalMetrics

    # Timing
    latency_ms: float


@dataclass
class StrategyMetrics:
    """Aggregated metrics for a retrieval strategy."""

    strategy_name: str
    num_queries: int

    # IR metrics (averaged)
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_mrr: float

    # By query type
    precision_by_type: dict[str, float]
    recall_by_type: dict[str, float]
    f1_by_type: dict[str, float]

    # Latency
    avg_latency_ms: float
    p95_latency_ms: float


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    dataset_path: str
    conversation_id: str
    total_queries: int
    query_results: list[QueryResult]
    strategy_metrics: dict[str, StrategyMetrics]
    summary: str


# =============================================================================
# Ground Truth Loading
# =============================================================================


def load_ground_truth_queries(dataset_path: Path) -> list[GroundTruthQuery]:
    """Load test queries with ground truth from JSON file."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries: list[GroundTruthQuery] = []
    for q in data.get("queries", []):
        # Skip queries that need review (unless we have expected results)
        if q.get("needs_review", True) and not q.get("expected_memory_ids"):
            continue

        queries.append(GroundTruthQuery(
            id=q["id"],
            query=q["query"],
            query_type=q["query_type"],
            expected_memory_ids=q.get("expected_memory_ids", []),
            expected_facts=q.get("expected_facts", []),
            expected_entity=q.get("expected_entity"),
            expected_attribute=q.get("expected_attribute"),
            confidence=q.get("confidence", 0.0),
            needs_review=q.get("needs_review", True),
            notes=q.get("notes", ""),
        ))

    logger.info(f"Loaded {len(queries)} ground truth queries from {dataset_path}")
    return queries


# =============================================================================
# IR Metrics Computation
# =============================================================================


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> RetrievalMetrics:
    """Compute standard IR metrics.

    Args:
        retrieved_ids: Memory IDs that were actually retrieved
        expected_ids: Memory IDs that should have been retrieved (ground truth)

    Returns:
        RetrievalMetrics with precision, recall, F1, and MRR
    """
    if not expected_ids:
        # No ground truth - can't compute meaningful metrics
        return RetrievalMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            mrr=0.0,
            num_retrieved=len(retrieved_ids),
            num_expected=0,
            num_correct=0,
        )

    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)

    # True positives: retrieved AND expected
    correct = retrieved_set & expected_set
    num_correct = len(correct)

    # Precision: what % of retrieved items are in expected set
    precision = num_correct / len(retrieved_ids) if retrieved_ids else 0.0

    # Recall: what % of expected items were retrieved
    recall = num_correct / len(expected_ids)

    # F1: harmonic mean
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # MRR: 1 / rank of first correct answer
    mrr = 0.0
    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in expected_set:
            mrr = 1.0 / rank
            break

    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        mrr=mrr,
        num_retrieved=len(retrieved_ids),
        num_expected=len(expected_ids),
        num_correct=num_correct,
    )


# =============================================================================
# Retrieval Strategies
# =============================================================================


def retrieve_similarity_only(
    retriever: UnifiedRetriever,
    query: str,
    top_k: int = 10,
) -> RetrievalContext:
    """Baseline: pure similarity search."""
    return retriever.retrieve(
        user_input=query,
        override_query_type=QueryType.HISTORY,  # Forces similarity search
    )


def retrieve_kg_only(
    retriever: UnifiedRetriever,
    query: str,
    top_k: int = 10,
) -> RetrievalContext:
    """Baseline: route everything through KG."""
    return retriever.retrieve(
        user_input=query,
        override_query_type=QueryType.ENTITY_OVERVIEW,
    )


def retrieve_unified(
    retriever: UnifiedRetriever,
    query: str,
) -> RetrievalContext:
    """Full unified pipeline with query classification."""
    return retriever.retrieve(
        user_input=query,
        conversation_context=[],
    )


# =============================================================================
# Evaluation Runner
# =============================================================================


def evaluate_strategy(
    strategy_name: str,
    strategy_fn: type,
    retriever: UnifiedRetriever,
    queries: list[GroundTruthQuery],
) -> list[QueryResult]:
    """Evaluate a single retrieval strategy on all queries."""
    results: list[QueryResult] = []

    for query in queries:
        start_time = time.time()

        # Run retrieval
        context = strategy_fn(retriever, query.query)

        latency_ms = (time.time() - start_time) * 1000

        # Extract retrieved memory IDs
        retrieved_ids = [m.memory_id for m in context.memories]

        # Also add memory IDs from facts (if any)
        for fact in context.facts:
            if fact.source_memory_id and fact.source_memory_id not in retrieved_ids:
                retrieved_ids.append(fact.source_memory_id)

        # Compute metrics
        metrics = compute_retrieval_metrics(retrieved_ids, query.expected_memory_ids)

        results.append(QueryResult(
            query_id=query.id,
            query_text=query.query,
            query_type=query.query_type,
            strategy=strategy_name,
            retrieved_memory_ids=retrieved_ids,
            expected_memory_ids=query.expected_memory_ids,
            metrics=metrics,
            latency_ms=latency_ms,
        ))

    return results


def compute_strategy_metrics(
    strategy_name: str,
    results: list[QueryResult],
) -> StrategyMetrics:
    """Compute aggregated metrics for a strategy."""
    if not results:
        return StrategyMetrics(
            strategy_name=strategy_name,
            num_queries=0,
            avg_precision=0.0,
            avg_recall=0.0,
            avg_f1=0.0,
            avg_mrr=0.0,
            precision_by_type={},
            recall_by_type={},
            f1_by_type={},
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
        )

    # Overall averages
    avg_precision = float(np.mean([r.metrics.precision for r in results]))
    avg_recall = float(np.mean([r.metrics.recall for r in results]))
    avg_f1 = float(np.mean([r.metrics.f1 for r in results]))
    avg_mrr = float(np.mean([r.metrics.mrr for r in results]))

    # By query type
    by_type: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_type[r.query_type].append(r)

    precision_by_type = {
        qt: float(np.mean([r.metrics.precision for r in qrs]))
        for qt, qrs in by_type.items()
    }
    recall_by_type = {
        qt: float(np.mean([r.metrics.recall for r in qrs]))
        for qt, qrs in by_type.items()
    }
    f1_by_type = {
        qt: float(np.mean([r.metrics.f1 for r in qrs]))
        for qt, qrs in by_type.items()
    }

    # Latency
    latencies = [r.latency_ms for r in results]
    avg_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))

    return StrategyMetrics(
        strategy_name=strategy_name,
        num_queries=len(results),
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        avg_f1=avg_f1,
        avg_mrr=avg_mrr,
        precision_by_type=precision_by_type,
        recall_by_type=recall_by_type,
        f1_by_type=f1_by_type,
        avg_latency_ms=avg_latency_ms,
        p95_latency_ms=p95_latency_ms,
    )


def run_evaluation(
    dataset_path: Path,
    conversation_id: str,
    max_memories: int | None = None,
    use_cached_indices: bool = True,
) -> EvaluationResult:
    """Run full evaluation comparing all strategies with proper IR metrics."""
    print("\n" + "=" * 70)
    print("UNIFIED RETRIEVAL EVALUATION")
    print("Using Precision/Recall against Ground Truth")
    print("=" * 70)

    # Build or load indices
    cache_dir = CACHE_DIR / conversation_id

    if use_cached_indices and cache_dir.exists():
        print(f"Loading cached indices from {cache_dir}")
        embedding_service = get_embedding_service()
        kg, memory_index, episode_index, topic_clusters = load_indices(
            cache_dir, embedding_service
        )
        retriever = UnifiedRetriever(
            kg=kg,
            memory_index=memory_index,
            episode_index=episode_index,
            topic_clusters=topic_clusters,
            embedding_service=embedding_service,
            classifier=RuleBasedQueryClassifier(),
        )
    else:
        print("Building indices...")
        retriever = build_all_indices(
            conversation_id=conversation_id,
            max_memories=max_memories,
            output_dir=cache_dir,
        )

    # Load ground truth queries
    queries = load_ground_truth_queries(dataset_path)
    if not queries:
        raise ValueError(f"No valid ground truth queries in {dataset_path}")

    print(f"\nLoaded {len(queries)} ground truth queries")
    print(f"Memory index size: {len(retriever.memory_index.memories)}")

    # Define strategies to compare
    strategies = {
        "similarity_only": retrieve_similarity_only,
        "kg_only": retrieve_kg_only,
        "unified": retrieve_unified,
    }

    # Run evaluations
    all_results: list[QueryResult] = []
    strategy_metrics: dict[str, StrategyMetrics] = {}

    for strategy_name, strategy_fn in strategies.items():
        print(f"\n--- Evaluating {strategy_name} ---")
        results = evaluate_strategy(strategy_name, strategy_fn, retriever, queries)
        all_results.extend(results)

        metrics = compute_strategy_metrics(strategy_name, results)
        strategy_metrics[strategy_name] = metrics

        print(f"  Precision: {metrics.avg_precision:.3f}")
        print(f"  Recall: {metrics.avg_recall:.3f}")
        print(f"  F1: {metrics.avg_f1:.3f}")
        print(f"  MRR: {metrics.avg_mrr:.3f}")
        print(f"  Latency (avg): {metrics.avg_latency_ms:.1f}ms")

    # Generate summary
    summary = _generate_summary(strategy_metrics, queries)
    print(summary)

    return EvaluationResult(
        dataset_path=str(dataset_path),
        conversation_id=conversation_id,
        total_queries=len(queries),
        query_results=all_results,
        strategy_metrics=strategy_metrics,
        summary=summary,
    )


def _generate_summary(
    strategy_metrics: dict[str, StrategyMetrics],
    queries: list[GroundTruthQuery],
) -> str:
    """Generate human-readable summary."""
    lines = [
        "",
        "=" * 70,
        "EVALUATION SUMMARY",
        "=" * 70,
        "",
        "OVERALL METRICS:",
        f"{'Strategy':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MRR':>10} {'Latency':>12}",
        "-" * 70,
    ]

    for name, metrics in sorted(strategy_metrics.items()):
        lines.append(
            f"{name:<20} {metrics.avg_precision:>10.3f} {metrics.avg_recall:>10.3f} "
            f"{metrics.avg_f1:>10.3f} {metrics.avg_mrr:>10.3f} {metrics.avg_latency_ms:>10.1f}ms"
        )

    lines.extend([
        "",
        "=" * 70,
        "F1 BY QUERY TYPE:",
        "=" * 70,
        "",
    ])

    # Collect all query types
    all_types: set[str] = set()
    for metrics in strategy_metrics.values():
        all_types.update(metrics.f1_by_type.keys())

    header = f"{'Query Type':<20}"
    for name in sorted(strategy_metrics.keys()):
        header += f" {name:>15}"
    lines.append(header)
    lines.append("-" * 70)

    for qtype in sorted(all_types):
        row = f"{qtype:<20}"
        for name in sorted(strategy_metrics.keys()):
            f1 = strategy_metrics[name].f1_by_type.get(qtype, 0.0)
            row += f" {f1:>15.3f}"
        lines.append(row)

    lines.extend([
        "",
        "=" * 70,
        "KEY FINDINGS",
        "=" * 70,
        "",
    ])

    # Find best strategy
    best_f1 = max(strategy_metrics.values(), key=lambda m: m.avg_f1)
    lines.append(f"Best overall F1: {best_f1.strategy_name} ({best_f1.avg_f1:.3f})")

    # Compare unified vs baselines
    if "unified" in strategy_metrics and "similarity_only" in strategy_metrics:
        unified = strategy_metrics["unified"]
        baseline = strategy_metrics["similarity_only"]
        f1_diff = unified.avg_f1 - baseline.avg_f1
        if f1_diff > 0:
            lines.append(f"Unified beats similarity baseline by {f1_diff:.3f} F1")
        else:
            lines.append(f"Similarity baseline beats unified by {abs(f1_diff):.3f} F1")

    # Query type analysis
    lines.append("")
    lines.append("Per query type winners:")
    for qtype in sorted(all_types):
        best_strat = None
        best_score = 0.0
        for name, metrics in strategy_metrics.items():
            score = metrics.f1_by_type.get(qtype, 0.0)
            if score > best_score:
                best_score = score
                best_strat = name
        if best_strat:
            lines.append(f"  {qtype}: {best_strat} (F1={best_score:.3f})")

    lines.append("")

    return "\n".join(lines)


def save_results(result: EvaluationResult, output_file: Path) -> None:
    """Save evaluation results to JSON."""
    data = {
        "dataset_path": result.dataset_path,
        "conversation_id": result.conversation_id,
        "total_queries": result.total_queries,
        "query_results": [
            {
                "query_id": r.query_id,
                "query_text": r.query_text,
                "query_type": r.query_type,
                "strategy": r.strategy,
                "retrieved_memory_ids": r.retrieved_memory_ids,
                "expected_memory_ids": r.expected_memory_ids,
                "metrics": asdict(r.metrics),
                "latency_ms": r.latency_ms,
            }
            for r in result.query_results
        ],
        "strategy_metrics": {
            name: asdict(metrics)
            for name, metrics in result.strategy_metrics.items()
        },
        "summary": result.summary,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate unified retrieval pipeline with IR metrics"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to ground truth dataset JSON",
    )
    parser.add_argument(
        "--conversation",
        type=str,
        required=True,
        help="Conversation ID to evaluate",
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

    result = run_evaluation(
        dataset_path=Path(args.dataset),
        conversation_id=args.conversation,
        max_memories=args.max_memories,
        use_cached_indices=not args.no_cache,
    )

    if args.output:
        save_results(result, Path(args.output))
    else:
        output_file = CACHE_DIR / f"{args.conversation}_evaluation.json"
        save_results(result, output_file)


if __name__ == "__main__":
    main()
