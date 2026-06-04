"""
Evaluation pipeline for temporal retrieval.

Measures:
- Time parse accuracy
- Episode recall/precision/F1
- Content relevance (LLM-judged)
- Latency
"""

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from agent.experiments.temporal_retrieval.episode_index import EpisodeIndex
from agent.experiments.temporal_retrieval.models import (
    EvaluationMetrics,
    RetrievalResult,
    StrategyComparison,
    TemporalQuery,
)
from agent.experiments.temporal_retrieval.strategies import (
    STRATEGIES,
    retrieve_with_strategy,
)
from agent.experiments.temporal_retrieval.time_parser import parse_time_reference
from agent.llm.models import SupportedModel
from agent.llm.router import LLM
from agent.memory.dag.models import MemoryElement


def evaluate_time_parsing(
    queries: list[TemporalQuery],
    now: datetime | None = None,
    llm: LLM | None = None,
    model: SupportedModel | None = None,
) -> dict[str, float]:
    """
    Evaluate time parsing accuracy.

    Args:
        queries: Test queries with expected time references
        now: Reference time for parsing
        llm: LLM instance for parsing (optional)
        model: Model for parsing (optional)

    Returns:
        Dict with accuracy metrics by type
    """
    if now is None:
        now = datetime.now()

    results = {
        "total": 0,
        "correct": 0,
        "relative_total": 0,
        "relative_correct": 0,
        "absolute_total": 0,
        "absolute_correct": 0,
        "emotional_total": 0,
        "emotional_correct": 0,
    }

    for query in queries:
        results["total"] += 1

        # Parse the query (uses LLM if provided)
        parsed = parse_time_reference(query.query_text, now, llm, model)

        # Check if type matches
        if parsed and parsed.ref_type == query.time_ref.ref_type:
            results["correct"] += 1

            # Type-specific counting
            if query.time_ref.ref_type == "relative":
                results["relative_total"] += 1
                results["relative_correct"] += 1
            elif query.time_ref.ref_type == "absolute":
                results["absolute_total"] += 1
                results["absolute_correct"] += 1
            elif query.time_ref.ref_type == "emotional":
                results["emotional_total"] += 1
                results["emotional_correct"] += 1
        else:
            # Wrong type or parse failure
            if query.time_ref.ref_type == "relative":
                results["relative_total"] += 1
            elif query.time_ref.ref_type == "absolute":
                results["absolute_total"] += 1
            elif query.time_ref.ref_type == "emotional":
                results["emotional_total"] += 1

    # Calculate accuracies
    return {
        "overall_accuracy": (
            results["correct"] / results["total"] if results["total"] > 0 else 0.0
        ),
        "relative_accuracy": (
            results["relative_correct"] / results["relative_total"]
            if results["relative_total"] > 0
            else 0.0
        ),
        "absolute_accuracy": (
            results["absolute_correct"] / results["absolute_total"]
            if results["absolute_total"] > 0
            else 0.0
        ),
        "emotional_accuracy": (
            results["emotional_correct"] / results["emotional_total"]
            if results["emotional_total"] > 0
            else 0.0
        ),
    }


def calculate_episode_metrics(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 for episode retrieval.

    Args:
        retrieved_ids: IDs of retrieved episodes
        expected_ids: IDs of expected episodes

    Returns:
        Tuple of (precision, recall, f1)
    """
    if not expected_ids:
        # No ground truth - can't calculate
        return 0.0, 0.0, 0.0

    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)

    true_positives = len(retrieved_set & expected_set)

    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def evaluate_content_relevance(
    result: RetrievalResult,
    query: TemporalQuery,
    llm: LLM,
    model: SupportedModel,
) -> float:
    """
    Evaluate content relevance using LLM as judge.

    Args:
        result: Retrieval result
        query: Original query
        llm: LLM router
        model: Model to use

    Returns:
        Relevance score (0.0 to 1.0)
    """
    if not result.retrieved_summaries:
        return 0.0

    # Format retrieved content
    content = "\n\n".join(result.retrieved_summaries[:3])

    prompt = f"""Evaluate how relevant the retrieved content is to the query.

Query: "{query.query_text}"

Retrieved content:
{content[:2000]}

Rate the relevance from 0 to 10:
- 0: Completely irrelevant
- 5: Somewhat relevant
- 10: Highly relevant and directly answers the query

Output only a number from 0 to 10.

Rating:"""

    response = llm.generate(
        model=model,
        prompt=prompt,
        caller="content_relevance_evaluation",
    )

    # Parse rating
    try:
        # Extract first number from response
        import re

        numbers = re.findall(r"\d+", response)
        if numbers:
            rating = int(numbers[0])
            return min(rating, 10) / 10.0
    except (ValueError, IndexError):
        pass

    return 0.5  # Default to middle if parsing fails


def evaluate_strategy(
    strategy_name: str,
    queries: list[TemporalQuery],
    index: EpisodeIndex,
    memories: list[MemoryElement],
    llm: LLM | None = None,
    model: SupportedModel | None = None,
    evaluate_relevance: bool = False,
    reference_time: datetime | None = None,
) -> EvaluationMetrics:
    """
    Evaluate a retrieval strategy on a test dataset.

    Args:
        strategy_name: Name of strategy (A, B, C, or D)
        queries: Test queries
        index: Episode index
        memories: All memories
        llm: LLM for relevance evaluation
        model: Model for relevance evaluation
        evaluate_relevance: Whether to evaluate content relevance
        reference_time: Reference time for parsing (defaults to max episode end time)

    Returns:
        EvaluationMetrics for the strategy
    """
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    latencies: list[float] = []
    relevance_scores: list[float] = []

    # Calculate reference time from episode data if not provided
    if reference_time is None:
        episodes = index.get_all_episodes()
        if episodes:
            reference_time = max(ep.end_time for ep in episodes)
        else:
            reference_time = datetime.now()

    for query in queries:
        # Run retrieval (uses LLM for time parsing if provided)
        result = retrieve_with_strategy(
            query=query.query_text,
            strategy_name=strategy_name,
            index=index,
            memories=memories,
            now=reference_time,
            top_k=5,
            llm=llm,
            model=model,
        )

        # Calculate episode metrics
        precision, recall, f1 = calculate_episode_metrics(
            result.retrieved_episode_ids,
            query.expected_episode_ids,
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        latencies.append(result.latency_ms)

        # Evaluate content relevance if requested
        if evaluate_relevance and llm and model:
            relevance = evaluate_content_relevance(result, query, llm, model)
            relevance_scores.append(relevance)

    # Calculate time parsing accuracy using the same reference time and LLM
    parsing_metrics = evaluate_time_parsing(queries, now=reference_time, llm=llm, model=model)

    # Aggregate metrics
    sorted_latencies = sorted(latencies)
    p95_index = int(len(sorted_latencies) * 0.95)

    return EvaluationMetrics(
        time_parse_accuracy=parsing_metrics["overall_accuracy"],
        relative_time_accuracy=parsing_metrics["relative_accuracy"],
        absolute_time_accuracy=parsing_metrics["absolute_accuracy"],
        emotional_time_accuracy=parsing_metrics["emotional_accuracy"],
        episode_precision=mean(precisions) if precisions else 0.0,
        episode_recall=mean(recalls) if recalls else 0.0,
        episode_f1=mean(f1s) if f1s else 0.0,
        content_relevance=mean(relevance_scores) if relevance_scores else 0.0,
        avg_latency_ms=mean(latencies) if latencies else 0.0,
        p95_latency_ms=sorted_latencies[p95_index] if sorted_latencies else 0.0,
    )


def run_full_evaluation(
    queries: list[TemporalQuery],
    index: EpisodeIndex,
    memories: list[MemoryElement],
    llm: LLM | None = None,
    model: SupportedModel | None = None,
    evaluate_relevance: bool = False,
) -> dict[str, StrategyComparison]:
    """
    Run full evaluation comparing all strategies.

    Args:
        queries: Test queries
        index: Episode index
        memories: All memories
        llm: LLM for relevance evaluation
        model: Model for relevance evaluation
        evaluate_relevance: Whether to evaluate content relevance

    Returns:
        Dict mapping strategy name to StrategyComparison
    """
    results: dict[str, StrategyComparison] = {}

    # Calculate reference time from episode data
    episodes = index.get_all_episodes()
    if episodes:
        reference_time = max(ep.end_time for ep in episodes)
    else:
        reference_time = datetime.now()

    for name in STRATEGIES:
        print(f"Evaluating strategy {name}...")
        metrics = evaluate_strategy(
            strategy_name=name,
            queries=queries,
            index=index,
            memories=memories,
            llm=llm,
            model=model,
            evaluate_relevance=evaluate_relevance,
            reference_time=reference_time,
        )

        results[name] = StrategyComparison(
            strategy_name=name,
            metrics=metrics,
            num_queries=len(queries),
        )

    return results


def save_evaluation_results(
    results: dict[str, StrategyComparison],
    filepath: Path,
) -> None:
    """Save evaluation results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "strategies": {},
    }

    for name, comparison in results.items():
        data["strategies"][name] = {
            "strategy_name": comparison.strategy_name,
            "num_queries": comparison.num_queries,
            "metrics": asdict(comparison.metrics),
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def print_evaluation_summary(results: dict[str, StrategyComparison]) -> None:
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Header
    print(f"\n{'Strategy':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Latency':<12}")
    print("-" * 70)

    # Results for each strategy
    for name in sorted(results.keys()):
        comparison = results[name]
        m = comparison.metrics
        print(
            f"{name:<25} {m.episode_precision:<12.3f} {m.episode_recall:<12.3f} "
            f"{m.episode_f1:<12.3f} {m.avg_latency_ms:<12.1f}ms"
        )

    # Time parsing accuracy
    print("\n" + "-" * 70)
    print("TIME PARSING ACCURACY")
    print("-" * 70)

    # Use first strategy's metrics (all should have same parsing results)
    first_metrics = list(results.values())[0].metrics
    print(f"  Overall:   {first_metrics.time_parse_accuracy:.1%}")
    print(f"  Relative:  {first_metrics.relative_time_accuracy:.1%}")
    print(f"  Absolute:  {first_metrics.absolute_time_accuracy:.1%}")
    print(f"  Emotional: {first_metrics.emotional_time_accuracy:.1%}")

    # Content relevance if available
    if first_metrics.content_relevance > 0:
        print("\n" + "-" * 70)
        print("CONTENT RELEVANCE")
        print("-" * 70)
        for name in sorted(results.keys()):
            m = results[name].metrics
            print(f"  {name}: {m.content_relevance:.1%}")

    print("\n" + "=" * 70)


def generate_findings_report(
    results: dict[str, StrategyComparison],
    output_path: Path,
) -> None:
    """Generate a FINDINGS.md report from evaluation results."""
    lines = [
        "# Temporal Retrieval Experiment Findings",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
    ]

    # Find best strategy by F1
    best_strategy = max(results.values(), key=lambda x: x.metrics.episode_f1)
    lines.extend(
        [
            f"**Best Strategy:** {best_strategy.strategy_name} "
            f"(F1: {best_strategy.metrics.episode_f1:.3f})",
            "",
            "## Results by Strategy",
            "",
            "| Strategy | Precision | Recall | F1 | Latency (ms) |",
            "|----------|-----------|--------|-----|--------------|",
        ]
    )

    for name in sorted(results.keys()):
        m = results[name].metrics
        lines.append(
            f"| {name} | {m.episode_precision:.3f} | {m.episode_recall:.3f} | "
            f"{m.episode_f1:.3f} | {m.avg_latency_ms:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Time Parsing Accuracy",
            "",
            f"- Overall: {best_strategy.metrics.time_parse_accuracy:.1%}",
            f"- Relative time: {best_strategy.metrics.relative_time_accuracy:.1%}",
            f"- Absolute time: {best_strategy.metrics.absolute_time_accuracy:.1%}",
            f"- Emotional time: {best_strategy.metrics.emotional_time_accuracy:.1%}",
            "",
            "## Recommendations",
            "",
        ]
    )

    # Generate recommendations based on results
    if best_strategy.strategy_name == "A":
        lines.append(
            "- **Episode Summary Only** performs best - summaries capture key information"
        )
    elif best_strategy.strategy_name == "C":
        lines.append(
            "- **Hybrid approach** performs best - balance of context and detail"
        )

    if best_strategy.metrics.emotional_time_accuracy < 0.8:
        lines.append("- Emotional time parsing needs improvement - consider LLM-based approach")

    if best_strategy.metrics.avg_latency_ms > 100:
        lines.append("- Latency is high - consider caching or pre-computing embeddings")

    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "1. Test with larger dataset",
            "2. Evaluate content relevance with human judges",
            "3. Optimize latency for production use",
            "4. Test edge cases in time parsing",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
