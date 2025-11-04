"""
Analysis tools for tiered memory experiment results.

Compares retrieval strategies, analyzes token efficiency,
and evaluates relevance at different granularity levels.
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from .models import (
    TieredRetrievalResults,
    MemoryTier,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Metrics for a single retrieval strategy."""

    strategy_name: str
    total_results: int
    results_by_tier: Dict[str, int]
    avg_score: float
    max_score: float
    min_score: float
    estimated_tokens_standard: int
    estimated_tokens_drill_down: int
    tiers_used: List[str]


@dataclass
class QueryAnalysis:
    """Analysis for a single query across strategies."""

    query: str
    strategy_metrics: Dict[str, StrategyMetrics]
    best_strategy_by_score: str
    best_strategy_by_efficiency: str
    tier_distribution: Dict[str, int]


class TieredMemoryAnalyzer:
    """Analyzer for tiered memory experiment results."""

    def __init__(self):
        """Initialize analyzer."""
        pass

    def analyze_query_results(
        self,
        query: str,
        strategy_results: Dict[str, Dict],
    ) -> QueryAnalysis:
        """
        Analyze results for a single query across strategies.

        Args:
            query: The query string
            strategy_results: Dict mapping strategy name to result data

        Returns:
            QueryAnalysis object
        """
        metrics = {}
        all_tier_counts = defaultdict(int)

        for strategy_name, data in strategy_results.items():
            retrieval_results: TieredRetrievalResults = data["retrieval_results"]

            # Calculate metrics
            scores = [r.score for r in retrieval_results.results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0
            min_score = min(scores) if scores else 0.0

            # Count results by tier
            tier_counts = defaultdict(int)
            for result in retrieval_results.results:
                tier_counts[result.tier.value] += 1
                all_tier_counts[result.tier.value] += 1

            metrics[strategy_name] = StrategyMetrics(
                strategy_name=strategy_name,
                total_results=retrieval_results.total_results,
                results_by_tier=dict(tier_counts),
                avg_score=avg_score,
                max_score=max_score,
                min_score=min_score,
                estimated_tokens_standard=data["standard_tokens"],
                estimated_tokens_drill_down=data["drill_down_tokens"],
                tiers_used=data["tiers_used"],
            )

        # Determine best strategies
        best_by_score = max(metrics.keys(), key=lambda s: metrics[s].avg_score)

        best_by_efficiency = min(
            metrics.keys(),
            key=lambda s: metrics[s].estimated_tokens_standard
            / (metrics[s].avg_score + 0.001),
        )

        return QueryAnalysis(
            query=query,
            strategy_metrics=metrics,
            best_strategy_by_score=best_by_score,
            best_strategy_by_efficiency=best_by_efficiency,
            tier_distribution=dict(all_tier_counts),
        )

    def analyze_all_results(
        self,
        experiment_results: Dict[str, Dict[str, Dict]],
    ) -> List[QueryAnalysis]:
        """
        Analyze all experiment results.

        Args:
            experiment_results: Dict mapping query -> strategy -> data

        Returns:
            List of QueryAnalysis objects
        """
        analyses = []

        for query, strategy_results in experiment_results.items():
            analysis = self.analyze_query_results(query, strategy_results)
            analyses.append(analysis)

        return analyses

    def generate_comparison_report(
        self,
        analyses: List[QueryAnalysis],
    ) -> str:
        """
        Generate a comprehensive comparison report.

        Args:
            analyses: List of QueryAnalysis objects

        Returns:
            Formatted report string
        """
        lines = []

        lines.append("=" * 100)
        lines.append("TIERED MEMORY EXPERIMENT ANALYSIS REPORT")
        lines.append("=" * 100)
        lines.append("")

        # Overall summary
        lines.append("## Overall Summary")
        lines.append("")
        lines.append(f"Total queries analyzed: {len(analyses)}")
        lines.append("")

        # Strategy comparison
        strategy_names = set()
        for analysis in analyses:
            strategy_names.update(analysis.strategy_metrics.keys())

        lines.append("## Strategy Comparison")
        lines.append("")

        for strategy in sorted(strategy_names):
            lines.append(f"\n### Strategy: {strategy}")
            lines.append("-" * 80)

            # Aggregate metrics across queries
            total_results = []
            avg_scores = []
            tokens_standard = []
            tokens_drill_down = []

            for analysis in analyses:
                if strategy in analysis.strategy_metrics:
                    m = analysis.strategy_metrics[strategy]
                    total_results.append(m.total_results)
                    avg_scores.append(m.avg_score)
                    tokens_standard.append(m.estimated_tokens_standard)
                    tokens_drill_down.append(m.estimated_tokens_drill_down)

            if avg_scores:
                lines.append(
                    f"Average results per query: {sum(total_results) / len(total_results):.1f}"
                )
                lines.append(
                    f"Average relevance score: {sum(avg_scores) / len(avg_scores):.3f}"
                )
                lines.append(
                    f"Average tokens (standard): {sum(tokens_standard) / len(tokens_standard):.0f}"
                )
                lines.append(
                    f"Average tokens (drill-down): {sum(tokens_drill_down) / len(tokens_drill_down):.0f}"
                )

                efficiency = (sum(avg_scores) / len(avg_scores)) / (
                    sum(tokens_standard) / len(tokens_standard)
                )
                lines.append(f"Efficiency (score per token): {efficiency:.6f}")

        # Per-query analysis
        lines.append("\n\n## Per-Query Analysis")
        lines.append("")

        for i, analysis in enumerate(analyses, 1):
            lines.append(f"\n### Query {i}: '{analysis.query}'")
            lines.append("-" * 80)

            lines.append(f"Best strategy (by score): {analysis.best_strategy_by_score}")
            lines.append(
                f"Best strategy (by efficiency): {analysis.best_strategy_by_efficiency}"
            )
            lines.append("")

            lines.append("Strategy Performance:")
            lines.append("")
            lines.append(
                f"{'Strategy':<20} {'Results':<10} {'Avg Score':<12} {'Tokens (std)':<15} {'Efficiency':<12}"
            )
            lines.append("-" * 80)

            for strategy, metrics in sorted(analysis.strategy_metrics.items()):
                efficiency = metrics.avg_score / (
                    metrics.estimated_tokens_standard + 0.001
                )
                lines.append(
                    f"{strategy:<20} "
                    f"{metrics.total_results:<10} "
                    f"{metrics.avg_score:<12.3f} "
                    f"{metrics.estimated_tokens_standard:<15} "
                    f"{efficiency:<12.6f}"
                )

            lines.append("")
            lines.append(f"Tier distribution across all strategies:")
            for tier, count in sorted(analysis.tier_distribution.items()):
                lines.append(f"  {tier}: {count} results")

        # Tier usage analysis
        lines.append("\n\n## Tier Usage Analysis")
        lines.append("")

        tier_usage = defaultdict(int)
        for analysis in analyses:
            for tier, count in analysis.tier_distribution.items():
                tier_usage[tier] += count

        total_tier_results = sum(tier_usage.values())
        lines.append(
            f"Total results across all queries and strategies: {total_tier_results}"
        )
        lines.append("")
        lines.append("Distribution by tier:")
        for tier in sorted(tier_usage.keys()):
            count = tier_usage[tier]
            percentage = (
                (count / total_tier_results) * 100 if total_tier_results > 0 else 0
            )
            lines.append(f"  {tier}: {count} ({percentage:.1f}%)")

        # Recommendations
        lines.append("\n\n## Recommendations")
        lines.append("")
        lines.append(self._generate_recommendations(analyses))

        lines.append("\n" + "=" * 100)
        lines.append("END OF ANALYSIS REPORT")
        lines.append("=" * 100)

        return "\n".join(lines)

    def _generate_recommendations(self, analyses: List[QueryAnalysis]) -> str:
        """Generate recommendations based on analysis."""
        lines = []

        # Determine which strategy wins most often
        score_winners = defaultdict(int)
        efficiency_winners = defaultdict(int)

        for analysis in analyses:
            score_winners[analysis.best_strategy_by_score] += 1
            efficiency_winners[analysis.best_strategy_by_efficiency] += 1

        best_score_strategy = max(score_winners.items(), key=lambda x: x[1])
        best_efficiency_strategy = max(efficiency_winners.items(), key=lambda x: x[1])

        lines.append(f"1. For maximum relevance:")
        lines.append(f"   Use '{best_score_strategy[0]}' strategy")
        lines.append(f"   (Best for {best_score_strategy[1]}/{len(analyses)} queries)")
        lines.append("")

        lines.append(f"2. For maximum token efficiency:")
        lines.append(f"   Use '{best_efficiency_strategy[0]}' strategy")
        lines.append(
            f"   (Best for {best_efficiency_strategy[1]}/{len(analyses)} queries)"
        )
        lines.append("")

        lines.append(
            f"3. Consider using different strategies for different query types:"
        )
        lines.append(f"   - Broad topic queries: Start with tier 4 (semantic clusters)")
        lines.append(
            f"   - Specific event queries: Use tier 2-3 (triggers/conversations)"
        )
        lines.append(f"   - Detailed action queries: Use tier 1 (atomic elements)")

        return "\n".join(lines)

    def generate_token_efficiency_report(
        self,
        analyses: List[QueryAnalysis],
    ) -> str:
        """
        Generate a focused report on token efficiency.

        Args:
            analyses: List of QueryAnalysis objects

        Returns:
            Formatted report string
        """
        lines = []

        lines.append("=" * 100)
        lines.append("TOKEN EFFICIENCY ANALYSIS")
        lines.append("=" * 100)
        lines.append("")

        # Collect all metrics
        all_metrics = []
        for analysis in analyses:
            for strategy, metrics in analysis.strategy_metrics.items():
                all_metrics.append((analysis.query, strategy, metrics))

        # Sort by efficiency (score per token)
        all_metrics.sort(
            key=lambda x: x[2].avg_score / (x[2].estimated_tokens_standard + 0.001),
            reverse=True,
        )

        lines.append("## Top 10 Most Efficient Query-Strategy Pairs")
        lines.append("")
        lines.append(
            f"{'Query':<40} {'Strategy':<20} {'Score':<10} {'Tokens':<10} {'Efficiency':<12}"
        )
        lines.append("-" * 100)

        for query, strategy, metrics in all_metrics[:10]:
            efficiency = metrics.avg_score / (metrics.estimated_tokens_standard + 0.001)
            query_short = query[:37] + "..." if len(query) > 40 else query
            lines.append(
                f"{query_short:<40} "
                f"{strategy:<20} "
                f"{metrics.avg_score:<10.3f} "
                f"{metrics.estimated_tokens_standard:<10} "
                f"{efficiency:<12.6f}"
            )

        lines.append("")
        lines.append("## Bottom 10 Least Efficient Query-Strategy Pairs")
        lines.append("")
        lines.append(
            f"{'Query':<40} {'Strategy':<20} {'Score':<10} {'Tokens':<10} {'Efficiency':<12}"
        )
        lines.append("-" * 100)

        for query, strategy, metrics in all_metrics[-10:]:
            efficiency = metrics.avg_score / (metrics.estimated_tokens_standard + 0.001)
            query_short = query[:37] + "..." if len(query) > 40 else query
            lines.append(
                f"{query_short:<40} "
                f"{strategy:<20} "
                f"{metrics.avg_score:<10.3f} "
                f"{metrics.estimated_tokens_standard:<10} "
                f"{efficiency:<12.6f}"
            )

        lines.append("")
        lines.append("=" * 100)

        return "\n".join(lines)


def analyze_experiment_results(
    experiment_results: Dict[str, Dict[str, Dict]],
    output_path: Optional[str] = None,
) -> Tuple[List[QueryAnalysis], str, str]:
    """
    Analyze experiment results and optionally save reports.

    Args:
        experiment_results: Dict mapping query -> strategy -> data
        output_path: Optional path to save reports

    Returns:
        Tuple of (analyses, comparison_report, efficiency_report)
    """
    analyzer = TieredMemoryAnalyzer()

    # Analyze all results
    analyses = analyzer.analyze_all_results(experiment_results)

    # Generate reports
    comparison_report = analyzer.generate_comparison_report(analyses)
    efficiency_report = analyzer.generate_token_efficiency_report(analyses)

    # Save if path provided
    if output_path:
        with open(f"{output_path}/comparison_report.txt", "w") as f:
            f.write(comparison_report)
        logger.info(f"Saved comparison report to {output_path}/comparison_report.txt")

        with open(f"{output_path}/efficiency_report.txt", "w") as f:
            f.write(efficiency_report)
        logger.info(f"Saved efficiency report to {output_path}/efficiency_report.txt")

    return analyses, comparison_report, efficiency_report
