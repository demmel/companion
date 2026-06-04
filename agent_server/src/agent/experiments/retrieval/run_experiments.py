"""Main experiment runner for retrieval experiments."""

import json
import logging
from datetime import datetime
from pathlib import Path

from agent.embedding_service import get_embedding_service
from agent.llm import LLM, create_llm, SupportedModel

from .models import Memory, QueryType, RetrievalResult, TestQuery
from .query_classifier import classify_query
from .query_generation import run_query_generation_experiment
from .temporal_retrieval import compare_strategies, retrieve_by_similarity
from .test_data import get_all_test_queries, get_all_test_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "output" / "results"


def run_experiment_1_classification(
    llm: LLM,
    model: SupportedModel,
) -> dict:
    """Experiment 1: Query Type Classification.

    Test whether we can reliably classify queries by type.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Query Type Classification")
    print("=" * 60)

    test_queries = get_all_test_queries()
    print(f"Testing {len(test_queries)} queries...")

    correct = 0
    results: list[dict] = []

    for query in test_queries:
        predicted_type, confidence = classify_query(query.query_text, llm, model)

        is_correct = predicted_type == query.query_type
        if is_correct:
            correct += 1

        result = {
            "query": query.query_text,
            "expected": query.query_type.value,
            "predicted": predicted_type.value,
            "confidence": confidence,
            "correct": is_correct,
        }
        results.append(result)

        status = "OK" if is_correct else "FAIL"
        print(
            f"  {status:4} [{query.query_type.value:12}] '{query.query_text[:50]}' -> {predicted_type.value} ({confidence:.0%})"
        )

    accuracy = correct / len(test_queries) if test_queries else 0

    print("\n" + "-" * 60)
    print(f"Classification Accuracy: {accuracy:.1%} ({correct}/{len(test_queries)})")

    # Confusion by type
    print("\nBy Query Type:")
    for qtype in QueryType:
        type_queries = [r for r in results if r["expected"] == qtype.value]
        if type_queries:
            type_correct = sum(1 for r in type_queries if r["correct"])
            print(f"  {qtype.value:12}: {type_correct}/{len(type_queries)} correct")

    return {
        "accuracy": accuracy,
        "total_queries": len(test_queries),
        "correct": correct,
        "results": results,
    }


def run_experiment_2_temporal(
    embedding_service,
) -> dict:
    """Experiment 2: Temporal Retrieval.

    Compare strategies for handling state/temporal queries.
    Focus: Does recency help for state queries?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Temporal Retrieval")
    print("=" * 60)

    sequences = get_all_test_sequences()

    strategy_metrics: dict[str, list[float]] = {
        "similarity": [],
        "recency_weighted_0.3": [],
        "recency_weighted_0.5": [],
        "most_recent_similar": [],
    }

    for sequence in sequences:
        print(f"\n--- {sequence.description} ---")

        # Filter to state queries only for this experiment
        state_queries = [
            q for q in sequence.test_queries if q.query_type == QueryType.STATE
        ]

        if not state_queries:
            print("  (no state queries in this sequence)")
            continue

        for query in state_queries:
            print(f"\n  Query: '{query.query_text}'")
            print(f"  Expected: {query.expected_memory_ids}")

            results = compare_strategies(query, sequence.memories, embedding_service)

            for strategy_name, result in results.items():
                strategy_metrics[strategy_name].append(result.reciprocal_rank)

                top_retrieved = (
                    result.retrieved_memory_ids[0]
                    if result.retrieved_memory_ids
                    else None
                )
                status = "OK" if result.correct else "FAIL"
                print(
                    f"    {status:4} {strategy_name:25}: {top_retrieved} (MRR={result.reciprocal_rank:.2f})"
                )

    # Aggregate metrics
    print("\n" + "-" * 60)
    print("AGGREGATE STATE QUERY METRICS")
    print("-" * 60)
    print(f"{'Strategy':<30} {'MRR':<10} {'Queries':<10}")
    print("-" * 60)

    summary = {}
    for strategy, mrr_values in strategy_metrics.items():
        if mrr_values:
            avg_mrr = sum(mrr_values) / len(mrr_values)
            print(f"{strategy:<30} {avg_mrr:<10.3f} {len(mrr_values):<10}")
            summary[strategy] = {
                "mrr": avg_mrr,
                "num_queries": len(mrr_values),
            }

    return summary


def run_experiment_3_all_query_types(
    embedding_service,
) -> dict:
    """Experiment 3: All Query Types.

    Test retrieval across all query types using best strategy for each.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: All Query Types")
    print("=" * 60)

    all_memories: list[Memory] = []
    all_queries: list[TestQuery] = []

    for sequence in get_all_test_sequences():
        all_memories.extend(sequence.memories)
        all_queries.extend(sequence.test_queries)

    print(f"Testing {len(all_queries)} queries against {len(all_memories)} memories...")

    # For now, use simple similarity for all query types
    # Future: route to different strategies based on type
    results_by_type: dict[str, list[RetrievalResult]] = {}

    for query in all_queries:
        # Get relevant memories (for this test, we use memories from same sequence)
        # In production, we'd search all memories
        retrieved = retrieve_by_similarity(
            query.query_text, all_memories, embedding_service, top_k=3
        )

        result = RetrievalResult(
            query=query,
            retrieved_memory_ids=[s.memory.memory_id for s in retrieved],
            scores=[s.combined_score for s in retrieved],
            correct=any(
                s.memory.memory_id in query.expected_memory_ids for s in retrieved
            ),
            reciprocal_rank=next(
                (
                    1.0 / (i + 1)
                    for i, s in enumerate(retrieved)
                    if s.memory.memory_id in query.expected_memory_ids
                ),
                0.0,
            ),
        )

        qtype = query.query_type.value
        if qtype not in results_by_type:
            results_by_type[qtype] = []
        results_by_type[qtype].append(result)

        status = "OK" if result.correct else "FAIL"
        top_id = result.retrieved_memory_ids[0] if result.retrieved_memory_ids else None
        expected = query.expected_memory_ids[0] if query.expected_memory_ids else None
        print(
            f"  {status:4} [{qtype:12}] '{query.query_text[:40]}' -> {top_id} (expected: {expected})"
        )

    # Summary by type
    print("\n" + "-" * 60)
    print("RESULTS BY QUERY TYPE (using naive similarity)")
    print("-" * 60)
    print(f"{'Type':<15} {'MRR':<10} {'Accuracy':<10} {'Count':<10}")
    print("-" * 60)

    summary = {}
    for qtype, results in results_by_type.items():
        mrr = sum(r.reciprocal_rank for r in results) / len(results) if results else 0
        accuracy = sum(1 for r in results if r.correct) / len(results) if results else 0
        print(f"{qtype:<15} {mrr:<10.3f} {accuracy:<10.1%} {len(results):<10}")
        summary[qtype] = {
            "mrr": mrr,
            "accuracy": accuracy,
            "count": len(results),
        }

    return summary


def save_results(results: dict, filename: str) -> None:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {filepath}")


def main() -> None:
    """Run all retrieval experiments."""
    print("\n" + "=" * 60)
    print("RETRIEVAL EXPERIMENTS")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Initialize
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    embedding_service = get_embedding_service()

    all_results = {"timestamp": datetime.now().isoformat()}

    # Experiment 1: Query Classification
    exp1_results = run_experiment_1_classification(llm, model)
    all_results["experiment_1_classification"] = exp1_results

    # Experiment 2: Temporal Retrieval
    exp2_results = run_experiment_2_temporal(embedding_service)
    all_results["experiment_2_temporal"] = exp2_results

    # Experiment 3: All Query Types
    exp3_results = run_experiment_3_all_query_types(embedding_service)
    all_results["experiment_3_all_types"] = exp3_results

    # Experiment 4: Query Generation from Context
    exp4_results = run_query_generation_experiment(llm, model)
    all_results["experiment_4_query_generation"] = exp4_results

    # Save results
    save_results(
        all_results,
        f"retrieval_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
