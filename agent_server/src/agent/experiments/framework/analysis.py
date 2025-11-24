"""
Experiment analyzer for calculating metrics from saved data.

Loads raw experiment data and calculates metrics on-demand,
allowing recalculation without re-running expensive experiments.
"""

import logging
from typing import Dict, List, Tuple, Optional
import statistics
from dataclasses import dataclass
from anthropic import BaseModel
from scipy import stats
import numpy as np

from .base import MetricsCalculator
from .data import (
    Metric,
    RunMetadata,
    TestCaseMetrics,
    VariantResults,
    ExperimentResults,
)
from .storage import ExperimentStorage

logger = logging.getLogger(__name__)


@dataclass
class TestComparison:
    """Comparison data for a single test between current and baseline."""

    test_name: str
    current_mean: float
    current_ci: Tuple[float, float]
    baseline_mean: float
    baseline_ci: Tuple[float, float]
    delta: float
    p_value: float


class ExperimentAnalyzer:
    """
    Analyzes saved experiment data.

    Loads raw data from disk and calculates metrics using provided
    MetricsCalculator. Allows recalculating metrics without re-running
    experiments.

    Works with heterogeneous experiments - each test case may have
    different data types. Type deserialization is handled automatically
    by the storage layer.
    """

    def __init__(self, storage: ExperimentStorage):
        """
        Initialize analyzer.

        Args:
            storage: Storage instance for loading data
        """
        self.storage = storage

    def calculate_metrics(
        self, run_ts: str, calculator: MetricsCalculator
    ) -> ExperimentResults:
        """
        Calculate metrics for all runs in an experiment.

        Works with heterogeneous types - each test case may have different
        input/output types. Storage handles deserialization automatically.

        Args:
            run_ts: Experiment run timestamp
            calculator: MetricsCalculator to use (handles BaseModel polymorphically)

        Returns:
            ExperimentResults with typed structure
        """
        # Extract type information from calculator's generic parameters
        import typing
        from pydantic import BaseModel

        output_type = BaseModel
        expected_type = BaseModel

        orig_bases = getattr(type(calculator), '__orig_bases__', None)
        if orig_bases:
            args = typing.get_args(orig_bases[0])
            if len(args) >= 2:
                output_type = args[0]
                expected_type = args[1]

        variant_results = {}
        variants = self.storage.list_variants(run_ts)

        for variant_name in variants:
            test_case_metrics = {}
            test_cases = self.storage.list_test_cases(run_ts, variant_name)

            for test_case_name in test_cases:
                # Load all runs for this variant/test case
                run_indices = self.storage.list_runs(
                    run_ts, variant_name, test_case_name
                )

                run_metrics_list = []
                metadata_list = []
                metric_names = None  # Track metric names from first successful run

                for run_index in run_indices:
                    # Load run with type information from calculator
                    run_data, metadata = self.storage.load_run_with_types(
                        run_ts, variant_name, test_case_name, run_index,
                        output_type, expected_type
                    )

                    metadata_list.append(metadata)

                    # Calculate metrics if run was successful
                    if metadata.success and run_data.output_data is not None:
                        metrics = calculator.calculate(
                            run_data.output_data, run_data.expected_output
                        )
                        if metric_names is None:
                            metric_names = set(metrics.keys())
                        run_metrics_list.append(metrics)
                    else:
                        # Failed run - will backfill with 0.0 later
                        run_metrics_list.append(None)

                # Backfill failed runs with 0.0 for all metrics
                if metric_names:
                    for i, metrics in enumerate(run_metrics_list):
                        if metrics is None:
                            run_metrics_list[i] = {name: 0.0 for name in metric_names}
                else:
                    # All runs failed - no metrics to calculate
                    run_metrics_list = []

                # Aggregate metrics across runs
                test_case_metrics[test_case_name] = self._aggregate_metrics(
                    run_metrics_list, metadata_list
                )

            variant_results[variant_name] = VariantResults(test_cases=test_case_metrics)

        # Calculate comparative metrics if supported by calculator
        self._calculate_comparative_metrics(run_ts, calculator, variant_results)

        return ExperimentResults(run_ts=run_ts, variants=variant_results)

    def _calculate_comparative_metrics(
        self,
        run_ts: str,
        calculator: MetricsCalculator,
        variant_results: Dict[str, VariantResults],
    ) -> None:
        """
        Calculate comparative metrics across variants for each test case.

        Modifies variant_results in-place by adding comparative_metrics to each TestCaseMetrics.

        Args:
            run_ts: Experiment run timestamp
            calculator: MetricsCalculator to use
            variant_results: Dict mapping variant_name -> VariantResults (modified in-place)
        """
        # Find all unique test case names
        all_test_cases = set()
        for variant_result in variant_results.values():
            all_test_cases.update(variant_result.test_cases.keys())

        # For each test case, collect outputs from all variants and calculate comparative metrics
        for test_case_name in all_test_cases:
            # Collect all outputs for this test case across all variants
            variant_outputs: Dict[str, List[BaseModel]] = {}

            for variant_name in variant_results.keys():
                # Check if this variant has this test case
                if test_case_name not in variant_results[variant_name].test_cases:
                    continue

                # Load all outputs for this variant/test case
                run_indices = self.storage.list_runs(
                    run_ts, variant_name, test_case_name
                )
                outputs = []

                for run_index in run_indices:
                    run_data, metadata = self.storage.load_run(
                        run_ts, variant_name, test_case_name, run_index
                    )

                    if metadata.success and run_data.output_data is not None:
                        outputs.append(run_data.output_data)

                if outputs:
                    variant_outputs[variant_name] = outputs

            # Calculate comparative metrics for this test case
            if len(variant_outputs) >= 2:  # Need at least 2 variants to compare
                comparative_metrics = calculator.calculate_comparative(
                    test_case_name, variant_outputs
                )
                # comparative_metrics is Dict[metric_name, Dict[variant_name, score]]
                # Example: {"richness": {"json": 0.9, "xml": 0.7}}

                # Store comparative metrics back into each variant's TestCaseMetrics
                for variant_name in variant_outputs.keys():
                    if test_case_name in variant_results[variant_name].test_cases:
                        # Extract this variant's scores across all comparative metrics
                        variant_comparative_scores = {}
                        for metric_name, variant_scores in comparative_metrics.items():
                            if variant_name in variant_scores:
                                variant_comparative_scores[metric_name] = (
                                    variant_scores[variant_name]
                                )

                        # Store in comparative_metrics dict
                        variant_results[variant_name].test_cases[
                            test_case_name
                        ].comparative_metrics = variant_comparative_scores

    def _aggregate_metrics(
        self, run_metrics_list: List[Dict[str, float]], metadata_list: List[RunMetadata]
    ) -> TestCaseMetrics:
        """
        Aggregate metrics across multiple runs.

        Args:
            run_metrics_list: List of metric dicts from each run
            metadata_list: List of metadata from each run

        Returns:
            Aggregated metrics as TestCaseMetrics
        """
        total_runs = len(metadata_list)
        successful_runs = sum(1 for m in metadata_list if m.success)

        metrics: Dict[str, Metric] = {}

        # Aggregate each metric across runs
        if run_metrics_list:
            metric_names = run_metrics_list[0].keys()

            for metric_name in metric_names:
                values = [m[metric_name] for m in run_metrics_list]
                metrics[metric_name] = Metric.from_values(values)

        return TestCaseMetrics(
            total_runs=total_runs,
            successful_runs=successful_runs,
            success_rate=successful_runs / total_runs if total_runs > 0 else 0.0,
            duration=Metric.from_values([m.duration_seconds for m in metadata_list]),
            retries=Metric.from_values([m.retry_count() for m in metadata_list]),
            metrics=metrics,
        )

    def generate_report(self, results: ExperimentResults) -> str:
        """
        Generate human-readable text report.

        Args:
            results: Results from calculate_metrics()

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENT RESULTS")
        lines.append("=" * 80)
        lines.append(f"\nExperiment: {results.run_ts}")
        lines.append(f"Variants: {len(results.variants)}")

        # Overall summary
        lines.append("\n" + "-" * 80)
        lines.append("OVERALL SUMMARY")
        lines.append("-" * 80)

        for variant_name, variant_results in results.variants.items():
            lines.append(f"\n{variant_name}:")
            for test_case_name, test_metrics in variant_results.test_cases.items():
                lines.append(f"  {test_case_name}:")
                lines.append(f"    Success rate: {test_metrics.success_rate:.1%}")

                # Show metric means if available
                for key, value in test_metrics.metrics.items():
                    if key.endswith("_mean") and not key.startswith("mean_"):
                        metric_name = key.replace("_mean", "")
                        std = test_metrics.metrics.get(f"{metric_name}_std", 0.0)
                        lines.append(f"    {metric_name}: {value:.3f} ± {std:.3f}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def compare_variants(
        self,
        results: ExperimentResults,
        metric_name: str = "f1_mean",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare variants on a specific metric.

        Args:
            results: Results from calculate_metrics()
            metric_name: Name of metric to compare

        Returns:
            Dict mapping variant_name -> test_case_name -> metric_value
        """
        comparison = {}

        for variant_name, variant_results in results.variants.items():
            comparison[variant_name] = {}
            for test_case_name, test_metrics in variant_results.test_cases.items():
                comparison[variant_name][test_case_name] = test_metrics.metrics.get(
                    metric_name, 0.0
                )

        return comparison

    def _perform_pairwise_tests_from_metrics(
        self, variant_metrics: Dict[str, Metric]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """
        Perform pairwise t-tests between all variants using pre-calculated metrics.

        Args:
            variant_metrics: Dict mapping variant_name -> Metric

        Returns:
            Dict mapping (variant1, variant2) -> (t_statistic, p_value)
        """
        results = {}
        variant_names = list(variant_metrics.keys())

        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                v1, v2 = variant_names[i], variant_names[j]
                metric1 = variant_metrics[v1]
                metric2 = variant_metrics[v2]

                if metric1.n > 1 and metric2.n > 1:
                    # Welch's t-test from summary statistics
                    t_stat, p_val = stats.ttest_ind_from_stats(
                        metric1.mean, metric1.stddev, metric1.n,
                        metric2.mean, metric2.stddev, metric2.n,
                        equal_var=False
                    )
                    results[(v1, v2)] = (t_stat, p_val)
                else:
                    results[(v1, v2)] = (0.0, 1.0)  # Not enough data

        return results

    def _perform_pairwise_tests(
        self, variant_values: Dict[str, List[float]]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """
        Perform pairwise t-tests between all variants.

        DEPRECATED: Use _perform_pairwise_tests_from_metrics for pre-calculated metrics.

        Args:
            variant_values: Dict mapping variant_name -> list of metric values

        Returns:
            Dict mapping (variant1, variant2) -> (t_statistic, p_value)
        """
        results = {}
        variant_names = list(variant_values.keys())

        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                v1, v2 = variant_names[i], variant_names[j]
                values1 = variant_values[v1]
                values2 = variant_values[v2]

                if len(values1) > 1 and len(values2) > 1:
                    t_stat, p_val = stats.ttest_ind(values1, values2)
                    results[(v1, v2)] = (t_stat, p_val)
                else:
                    results[(v1, v2)] = (0.0, 1.0)  # Not enough data

        return results

    def _calculate_significance_markers(
        self,
        variant_cis: Dict[str, Tuple[float, float, float]],
        pairwise_tests: Dict[Tuple[str, str], Tuple[float, float]],
    ) -> Dict[str, str]:
        """
        Calculate significance markers comparing all variants to the lowest-ranked.

        Args:
            variant_cis: Dict mapping variant_name -> (mean, lower_ci, upper_ci)
            pairwise_tests: Dict mapping (variant1, variant2) -> (t_statistic, p_value)

        Returns:
            Dict mapping variant_name -> significance_marker ("*", "**", "***", or empty)
        """
        # Sort variants by mean value (descending)
        sorted_variants = sorted(
            variant_cis.items(), key=lambda x: x[1][0], reverse=True
        )

        if len(sorted_variants) <= 1:
            return {}

        baseline_variant = sorted_variants[-1][0]  # Lowest ranked
        significance_markers = {}

        for variant_name, _ in sorted_variants[:-1]:
            # Find p-value comparing this variant to baseline
            p_val = None
            for (v1, v2), (t, p) in pairwise_tests.items():
                if (v1 == variant_name and v2 == baseline_variant) or (
                    v2 == variant_name and v1 == baseline_variant
                ):
                    p_val = p
                    break

            if p_val is not None:
                if p_val < 0.001:
                    significance_markers[variant_name] = "***"
                elif p_val < 0.01:
                    significance_markers[variant_name] = "**"
                elif p_val < 0.05:
                    significance_markers[variant_name] = "*"

        return significance_markers

    def _calculate_chart_range(
        self, values: List[float], padding: float = 0.05
    ) -> tuple[float, float, float]:
        """
        Calculate min, max, and range for chart with padding.

        Args:
            values: List of values to calculate range for
            padding: Fraction of range to add as padding (default 0.05 = 5%)

        Returns:
            Tuple of (min_val, max_val, range_val) with padding applied
        """
        min_val = min(values)
        max_val = max(values)

        # Add padding
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1.0
        min_val -= range_val * padding
        max_val += range_val * padding
        range_val = max_val - min_val

        return (min_val, max_val, range_val)

    def _generate_scale_line(
        self, min_val: float, range_val: float, width: int = 50, left_padding: int = 16
    ) -> str:
        """
        Generate scale label line for charts.

        Args:
            min_val: Minimum value for the scale
            range_val: Range of values (max - min)
            width: Width of the chart in characters
            left_padding: Space for variant names and value column

        Returns:
            Formatted scale line with 5 evenly-spaced labels
        """
        scale_line = " " * left_padding
        for i in range(5):
            val = min_val + (range_val * i / 4)
            pos = int(width * i / 4)
            label = f"{val:.2f}"
            scale_line += " " * max(0, pos - len(scale_line) + left_padding) + label
        return scale_line

    def _render_comparative_metric_chart(
        self,
        variant_scores: Dict[str, float],
        metric_name: str,
        width: int = 50,
    ) -> List[str]:
        """
        Render ASCII bar chart for comparative metrics (without error bars).

        Args:
            variant_scores: Dict mapping variant_name -> score
            metric_name: Name of the metric being displayed
            width: Width of the chart in characters

        Returns:
            List of lines to display
        """
        lines = []

        if not variant_scores:
            return lines

        # Filter out NaN values
        valid_scores = {k: v for k, v in variant_scores.items() if not np.isnan(v)}

        if not valid_scores:
            lines.append("(No valid data for this metric)")
            return lines

        # Determine value range
        min_val, max_val, range_val = self._calculate_chart_range(
            list(valid_scores.values())
        )

        # Sort variants by score (descending)
        sorted_variants = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)

        # Render each variant
        max_name_len = max(len(v) for v in valid_scores.keys())

        # Header with metric name
        lines.append(f"\n{metric_name.upper()} (comparative)")

        # Scale labels - account for name column + value column
        left_padding = max_name_len + 8
        lines.append(self._generate_scale_line(min_val, range_val, width, left_padding))

        for variant_name, score in sorted_variants:
            # Calculate position
            score_pos = int((score - min_val) / range_val * width)
            score_pos = max(0, min(width - 1, score_pos))

            # Build the chart line (simple bar)
            chart = [" "] * width
            for i in range(score_pos + 1):
                chart[i] = "="

            chart_str = "".join(chart)
            line = f"{variant_name:<{max_name_len}}  {score:.3f}  {chart_str}"
            lines.append(line)

        return lines

    def _render_error_bar_chart(
        self,
        variant_data: Dict[str, Tuple[float, float, float]],
        metric_name: str,
        significance: Optional[Dict[str, str]] = None,
        width: int = 50,
        max_name_len: Optional[int] = None,
    ) -> List[str]:
        """
        Render ASCII error bar chart.

        Args:
            variant_data: Dict mapping variant_name -> (mean, lower_ci, upper_ci)
            metric_name: Name of the metric being displayed
            significance: Optional dict mapping variant_name -> significance marker
            width: Width of the chart in characters

        Returns:
            List of lines to display
        """
        lines = []

        if not variant_data:
            return lines

        # Determine value range
        all_values = []
        for mean, lower, upper in variant_data.values():
            # Filter out NaN values
            if not np.isnan(mean) and not np.isnan(lower) and not np.isnan(upper):
                all_values.extend([lower, mean, upper])

        if not all_values:
            lines.append("(No valid data for this metric)")
            return lines

        # Calculate value range with padding
        min_val, max_val, range_val = self._calculate_chart_range(all_values)

        # Check if variants have timestamp suffixes (temporal comparison)
        # If yes, sort chronologically; otherwise sort by score
        variant_list = [(k, v) for k, v in variant_data.items() if not np.isnan(v[0])]

        # Check if any variant has a timestamp pattern like "(run_2025-11-23_13-17-05)"
        has_timestamps = any("(run_" in k for k, v in variant_list)

        if has_timestamps:
            # Temporal comparison: sort chronologically (earlier runs first)
            sorted_variants = sorted(variant_list, key=lambda x: x[0])
        else:
            # Regular variant comparison: sort by mean value (descending)
            sorted_variants = sorted(variant_list, key=lambda x: x[1][0], reverse=True)

        if not sorted_variants:
            lines.append("(No valid data for this metric)")
            return lines

        # Use provided max_name_len or calculate from current data
        if max_name_len is None:
            max_name_len = max(len(v) for v in variant_data.keys())

        # Header with metric name
        lines.append(f"\n{metric_name.upper()}")

        # Scale labels - account for name column + value column (8 chars for "  X.XXX  ")
        left_padding = max_name_len + 8
        lines.append(self._generate_scale_line(min_val, range_val, width, left_padding))

        for variant_name, (mean, lower_ci, upper_ci) in sorted_variants:
            # Skip if any value is NaN (shouldn't happen, but safety check)
            if np.isnan(mean) or np.isnan(lower_ci) or np.isnan(upper_ci):
                continue

            # Calculate positions
            mean_pos = int((mean - min_val) / range_val * width)
            lower_pos = int((lower_ci - min_val) / range_val * width)
            upper_pos = int((upper_ci - min_val) / range_val * width)

            # Ensure positions are within bounds
            mean_pos = max(0, min(width - 1, mean_pos))
            lower_pos = max(0, min(width - 1, lower_pos))
            upper_pos = max(0, min(width - 1, upper_pos))

            # Build the chart line
            chart = [" "] * width

            # Draw error bar line
            for i in range(lower_pos, upper_pos + 1):
                chart[i] = "-"

            # Draw brackets
            if lower_pos < width:
                chart[lower_pos] = "["
            if upper_pos < width:
                chart[upper_pos] = "]"

            # Draw mean indicator
            if mean_pos < width:
                chart[mean_pos] = "="

            chart_str = "".join(chart)

            # Add significance marker if provided
            sig_marker = ""
            if significance and variant_name in significance:
                sig_marker = f"  {significance[variant_name]}"

            line = (
                f"{variant_name:<{max_name_len}}  {mean:.3f}  {chart_str}{sig_marker}"
            )
            lines.append(line)

        return lines

    def _extract_category_from_test_case(self, test_case_name: str) -> str:
        """
        Extract category from test case name.

        Assumes test case names follow pattern: category_specific_name
        E.g., "fact_extraction_byzantine" -> "fact_extraction"

        Args:
            test_case_name: Name of the test case

        Returns:
            Category name
        """
        # Try to find common category prefixes
        common_categories = [
            "fact_extraction",
            "memory_query",
            "state_initialization",
            "action_planning",
        ]

        for category in common_categories:
            if test_case_name.startswith(category):
                return category

        # Fallback: use first part before underscore
        parts = test_case_name.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"

        return "uncategorized"

    def _aggregate_by_category(
        self,
        results: ExperimentResults,
        metric_name: str,
    ) -> Dict[str, Dict[str, Metric]]:
        """
        Aggregate metric summary statistics by category.

        Args:
            results: Experiment results with pre-calculated metrics
            metric_name: Name of metric to aggregate

        Returns:
            Dict mapping category -> variant_name -> Metric (aggregated across test cases)
        """
        category_data: Dict[str, Dict[str, List[float]]] = {}

        # Collect mean values from each test case
        for variant_name, variant_results in results.variants.items():
            for test_case_name, test_metrics in variant_results.test_cases.items():
                category = self._extract_category_from_test_case(test_case_name)

                if category not in category_data:
                    category_data[category] = {}

                if variant_name not in category_data[category]:
                    category_data[category][variant_name] = []

                # Use pre-calculated metric mean from test case
                metric = test_metrics.metrics.get(metric_name)
                if metric:
                    category_data[category][variant_name].append(metric.mean)

        # Convert lists of means to Metric objects (includes n automatically)
        category_metrics: Dict[str, Dict[str, Metric]] = {}
        for category, variant_means in category_data.items():
            category_metrics[category] = {}
            for variant_name, means in variant_means.items():
                if means:
                    category_metrics[category][variant_name] = Metric.from_values(means)

        return category_metrics

    def _aggregate_comparative_by_category(
        self,
        results: ExperimentResults,
        metric_name: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate comparative metric scores by category.

        Args:
            results: Experiment results
            metric_name: Name of comparative metric to aggregate

        Returns:
            Dict mapping category -> variant_name -> average score
        """
        category_data: Dict[str, Dict[str, List[float]]] = {}

        for variant_name, variant_results in results.variants.items():
            for test_case_name, test_metrics in variant_results.test_cases.items():
                if metric_name in test_metrics.comparative_metrics:
                    category = self._extract_category_from_test_case(test_case_name)

                    if category not in category_data:
                        category_data[category] = {}

                    if variant_name not in category_data[category]:
                        category_data[category][variant_name] = []

                    category_data[category][variant_name].append(
                        test_metrics.comparative_metrics[metric_name]
                    )

        # Average the scores for each variant within each category
        averaged_data: Dict[str, Dict[str, float]] = {}
        for category, variant_scores in category_data.items():
            averaged_data[category] = {}
            for variant_name, scores in variant_scores.items():
                if scores:
                    averaged_data[category][variant_name] = statistics.mean(scores)

        return averaged_data

    def generate_comparison_report(
        self,
        runs: List[ExperimentResults],
    ) -> str:
        """
        Generate comparative analysis report with error bars and statistical significance.

        Args:
            runs: List of experiment results to compare (can be from different runs/times)

        Returns:
            Formatted comparison report string
        """
        lines = []

        # Merge all runs into a single structure for display
        # Label variants with run timestamp if multiple runs
        from agent.experiments.framework.data import ExperimentResults

        merged_variants = {}
        run_labels = []

        for run_results in runs:
            run_label = run_results.run_ts
            run_labels.append(run_label)

            # Add variants from this run with timestamp suffix (if multiple runs)
            label_suffix = f" ({run_label})" if len(runs) > 1 else ""

            for variant_name, variant_data in run_results.variants.items():
                merged_variants[f"{variant_name}{label_suffix}"] = variant_data

        merged_results = ExperimentResults(
            run_ts=" vs ".join(run_labels) if len(runs) > 1 else run_labels[0],
            variants=merged_variants
        )

        # Auto-discover per-run metrics from the results
        metrics_to_compare = set()
        for variant_results in merged_results.variants.values():
            for test_metrics in variant_results.test_cases.values():
                # Metrics are stored as Metric objects, not _mean suffixes
                metrics_to_compare.update(test_metrics.metrics.keys())

        metrics_to_compare = sorted(metrics_to_compare)

        # Auto-discover comparative metrics
        comparative_metrics = set()
        for variant_results in merged_results.variants.values():
            for test_metrics in variant_results.test_cases.values():
                comparative_metrics.update(test_metrics.comparative_metrics.keys())

        comparative_metrics = sorted(comparative_metrics)

        lines.append("=" * 80)
        lines.append("COMPARATIVE EXPERIMENT ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"\nExperiment: {merged_results.run_ts}")
        lines.append(f"Variants: {len(merged_results.variants)}")

        # Count test cases per category
        all_test_cases = set()
        for variant_results in merged_results.variants.values():
            all_test_cases.update(variant_results.test_cases.keys())

        category_counts = {}
        for test_case in all_test_cases:
            category = self._extract_category_from_test_case(test_case)
            category_counts[category] = category_counts.get(category, 0) + 1

        # PER-CATEGORY ANALYSIS (shown first for bottom-up reading)
        for category in sorted(category_counts.keys()):
            lines.append("\n" + "=" * 80)
            lines.append(
                f"CATEGORY: {category} ({category_counts[category]} test cases)"
            )
            lines.append("=" * 80)

            # Per-run metrics with error bars
            for metric_name in metrics_to_compare:
                # Aggregate metrics by category from pre-calculated results
                category_data = self._aggregate_by_category(
                    merged_results, metric_name
                )

                if category not in category_data:
                    continue

                variant_metric_data = category_data[category]

                # Skip if no valid data
                if not variant_metric_data:
                    continue

                # Calculate CIs for each variant from pre-calculated metrics
                variant_cis = {}
                for variant_name, metric in variant_metric_data.items():
                    lower, upper = metric.confidence_interval()
                    variant_cis[variant_name] = (metric.mean, lower, upper)

                # Perform statistical tests using Metric summary statistics
                pairwise_tests = self._perform_pairwise_tests_from_metrics(variant_metric_data)

                # Determine significance markers (compare all to lowest-ranked)
                significance_markers = self._calculate_significance_markers(
                    variant_cis, pairwise_tests
                )

                # Render error bar chart
                chart_lines = self._render_error_bar_chart(
                    variant_cis, metric_name, significance_markers
                )
                lines.extend(chart_lines)

            # Comparative metrics (no error bars)
            for metric_name in comparative_metrics:
                category_scores = self._aggregate_comparative_by_category(
                    merged_results, metric_name
                )

                if category not in category_scores:
                    continue

                variant_scores = category_scores[category]

                # Skip if no valid data
                if not variant_scores:
                    continue

                # Render comparative metric chart
                chart_lines = self._render_comparative_metric_chart(
                    variant_scores, metric_name
                )
                lines.extend(chart_lines)

        # OVERALL ANALYSIS (shown last for easy visibility)
        lines.append("\n" + "=" * 80)
        lines.append(
            f"OVERALL PERFORMANCE (averaged across {len(all_test_cases)} test cases)"
        )
        lines.append("=" * 80)

        # Per-run metrics with error bars
        for metric_name in metrics_to_compare:
            # Aggregate all test case means across all variants
            variant_means_data: Dict[str, List[float]] = {}

            for variant_name, variant_results in merged_results.variants.items():
                variant_means_data[variant_name] = []
                for test_case_name, test_metrics in variant_results.test_cases.items():
                    metric = test_metrics.metrics.get(metric_name)
                    if metric:
                        variant_means_data[variant_name].append(metric.mean)

            # Skip if no valid data
            if not variant_means_data or all(not v for v in variant_means_data.values()):
                continue

            # Convert to Metric objects
            variant_metrics = {}
            for variant_name, means in variant_means_data.items():
                if means:
                    variant_metrics[variant_name] = Metric.from_values(means)

            # Calculate CIs for each variant
            variant_cis = {}
            for variant_name, metric in variant_metrics.items():
                lower, upper = metric.confidence_interval()
                variant_cis[variant_name] = (metric.mean, lower, upper)

            # Perform statistical tests
            pairwise_tests = self._perform_pairwise_tests_from_metrics(variant_metrics)

            # Determine significance markers
            significance_markers = self._calculate_significance_markers(
                variant_cis, pairwise_tests
            )

            # Render error bar chart
            chart_lines = self._render_error_bar_chart(
                variant_cis, metric_name, significance_markers
            )
            lines.extend(chart_lines)

        # Comparative metrics (no error bars)
        for metric_name in comparative_metrics:
            # Aggregate all comparative metric scores across all test cases
            variant_comp_scores: Dict[str, List[float]] = {}

            for variant_name, variant_results in merged_results.variants.items():
                variant_comp_scores[variant_name] = []
                for test_metrics in variant_results.test_cases.values():
                    if metric_name in test_metrics.comparative_metrics:
                        variant_comp_scores[variant_name].append(
                            test_metrics.comparative_metrics[metric_name]
                        )

            # Average the scores for each variant
            averaged_scores = {}
            for variant_name, scores in variant_comp_scores.items():
                if scores:
                    averaged_scores[variant_name] = statistics.mean(scores)

            # Skip if no valid data
            if not averaged_scores:
                continue

            # Render comparative metric chart
            chart_lines = self._render_comparative_metric_chart(
                averaged_scores, metric_name
            )
            lines.extend(chart_lines)

        # Add duration analysis
        lines.append("\n" + "=" * 80)
        lines.append("EXECUTION TIME (seconds per test case)")
        lines.append("=" * 80)

        # Collect duration means from pre-calculated metrics
        variant_duration_data: Dict[str, List[float]] = {}
        for variant_name, variant_results in merged_results.variants.items():
            variant_duration_data[variant_name] = []
            for test_metrics in variant_results.test_cases.values():
                # Duration is stored as a Metric in TestCaseMetrics
                variant_duration_data[variant_name].append(test_metrics.duration.mean)

        # Convert to Metric objects
        variant_duration_metrics = {}
        for variant_name, means in variant_duration_data.items():
            if means:
                variant_duration_metrics[variant_name] = Metric.from_values(means)

        # Calculate CIs for duration
        duration_cis = {}
        for variant_name, metric in variant_duration_metrics.items():
            lower, upper = metric.confidence_interval()
            duration_cis[variant_name] = (metric.mean, lower, upper)

        # Render duration chart (no significance testing for duration)
        duration_chart = self._render_error_bar_chart(
            duration_cis, "duration (seconds)"
        )
        lines.extend(duration_chart)

        # Statistical significance legend
        lines.append("\n" + "-" * 80)
        lines.append("STATISTICAL SIGNIFICANCE")
        lines.append("* p<0.05  ** p<0.01  *** p<0.001")
        lines.append("(compared to lowest-ranked variant)")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _render_test_comparison_section(
        self,
        tests: List[TestComparison],
        show_significance: bool = False,
    ) -> List[str]:
        """
        Render a section of test comparisons with aligned charts.

        Args:
            tests: List of test comparisons
            show_significance: Whether to show significance markers

        Returns:
            List of chart lines
        """
        lines = []
        if not tests:
            return lines

        # Calculate max name length for alignment
        max_test_name_len = max(len(f"{test.test_name} (baseline)") for test in tests)

        # Render each test separately to keep current/baseline adjacent
        for test in tests:
            # Create mini-chart for this test only
            variant_cis = {
                f"{test.test_name} (current)": (test.current_mean, test.current_ci[0], test.current_ci[1]),
                f"{test.test_name} (baseline)": (test.baseline_mean, test.baseline_ci[0], test.baseline_ci[1]),
            }

            significance_markers = None
            if show_significance:
                sig_marker = "***" if test.p_value < 0.001 else "**" if test.p_value < 0.01 else "*"
                significance_markers = {f"{test.test_name} (current)": sig_marker}

            chart_lines = self._render_error_bar_chart(
                variant_cis, "SCORE", significance_markers, max_name_len=max_test_name_len
            )
            lines.extend(chart_lines)

        return lines

    def generate_baseline_comparison(
        self,
        current_results: ExperimentResults,
        baseline_results: ExperimentResults,
        variant_name: Optional[str] = None,
    ) -> str:
        """
        Generate a report comparing current results to a baseline.

        Args:
            current_results: Results from current run
            baseline_results: Results from baseline run
            variant_name: Optional specific variant to compare (if None, compares all)

        Returns:
            Formatted comparison report showing improvements/regressions
        """
        lines = []
        lines.append("=" * 80)
        lines.append("BASELINE COMPARISON")
        lines.append("=" * 80)
        lines.append(f"Current:  {current_results.run_ts}")
        lines.append(f"Baseline: {baseline_results.run_ts}")
        lines.append("")

        # Determine which variants to compare
        if variant_name:
            variants_to_compare = [variant_name]
        else:
            variants_to_compare = list(
                set(current_results.variants.keys())
                & set(baseline_results.variants.keys())
            )

        for var_name in sorted(variants_to_compare):
            current_variant = current_results.variants.get(var_name)
            baseline_variant = baseline_results.variants.get(var_name)

            if not current_variant or not baseline_variant:
                lines.append(f"\n⚠️  Variant '{var_name}' not found in both runs")
                continue

            lines.append(f"\nVariant: {var_name}")
            lines.append("-" * 80)

            # Collect all test cases
            all_tests = set(current_variant.test_cases.keys()) | set(
                baseline_variant.test_cases.keys()
            )

            sig_improvements = []  # p < 0.05, current > baseline
            improvements = []  # p >= 0.05, current > baseline
            sig_regressions = []  # p < 0.05, current < baseline
            regressions = []  # p >= 0.05, current < baseline
            no_change = []  # overlapping CIs or p >= 0.05 with small delta
            new_tests = []
            removed_tests = []

            for test_name in sorted(all_tests):
                current_test = current_variant.test_cases.get(test_name)
                baseline_test = baseline_variant.test_cases.get(test_name)

                # Handle new/removed tests
                if not baseline_test:
                    new_tests.append(test_name)
                    continue
                if not current_test:
                    removed_tests.append(test_name)
                    continue

                # Get metrics
                current_metric = current_test.metrics.get("score")
                baseline_metric = baseline_test.metrics.get("score")

                if not current_metric or not baseline_metric:
                    continue

                # Extract summary statistics
                current_mean = current_metric.mean
                current_std = current_metric.stddev
                current_n = current_test.total_runs

                baseline_mean = baseline_metric.mean
                baseline_std = baseline_metric.stddev
                baseline_n = baseline_test.total_runs

                # Calculate CIs from summary statistics
                # Handle cases where std=0 (no variance) or n=1
                if current_n > 1 and current_std > 1e-10:
                    current_se = current_std / np.sqrt(current_n)
                    current_ci = stats.t.interval(
                        0.95, current_n - 1, loc=current_mean, scale=current_se
                    )
                else:
                    # No variance or single run - CI is just the point estimate
                    current_ci = (current_mean, current_mean)

                if baseline_n > 1 and baseline_std > 1e-10:
                    baseline_se = baseline_std / np.sqrt(baseline_n)
                    baseline_ci = stats.t.interval(
                        0.95, baseline_n - 1, loc=baseline_mean, scale=baseline_se
                    )
                else:
                    # No variance or single run - CI is just the point estimate
                    baseline_ci = (baseline_mean, baseline_mean)

                # Perform t-test
                if current_n > 1 and baseline_n > 1:
                    t_stat, p_value = stats.ttest_ind_from_stats(
                        current_mean, current_std, current_n,
                        baseline_mean, baseline_std, baseline_n,
                        equal_var=False  # Welch's t-test
                    )
                else:
                    p_value = 1.0  # Not enough data for significance test

                delta = current_mean - baseline_mean

                # Categorize based on significance and direction
                test_comparison = TestComparison(
                    test_name=test_name,
                    current_mean=current_mean,
                    current_ci=current_ci,
                    baseline_mean=baseline_mean,
                    baseline_ci=baseline_ci,
                    delta=delta,
                    p_value=p_value,
                )

                if p_value < 0.05:
                    if delta > 0:
                        sig_improvements.append(test_comparison)
                    elif delta < 0:
                        sig_regressions.append(test_comparison)
                    else:
                        no_change.append(test_comparison)
                else:
                    if abs(delta) < 0.01:
                        no_change.append(test_comparison)
                    elif delta > 0:
                        improvements.append(test_comparison)
                    else:
                        regressions.append(test_comparison)

            # Render charts for significant improvements
            if sig_improvements:
                lines.append(f"\n✅ SIGNIFICANT IMPROVEMENTS ({len(sig_improvements)}) - p < 0.05:")
                sorted_improvements = sorted(sig_improvements, key=lambda x: x.delta, reverse=True)
                lines.extend(self._render_test_comparison_section(sorted_improvements, show_significance=True))

            # Render charts for non-significant improvements
            if improvements:
                lines.append(f"\n✅ Improvements ({len(improvements)}) - not statistically significant:")
                sorted_improvements = sorted(improvements, key=lambda x: x.delta, reverse=True)
                lines.extend(self._render_test_comparison_section(sorted_improvements))

            # Render charts for significant regressions
            if sig_regressions:
                lines.append(f"\n❌ SIGNIFICANT REGRESSIONS ({len(sig_regressions)}) - p < 0.05:")
                sorted_regressions = sorted(sig_regressions, key=lambda x: x.delta)
                lines.extend(self._render_test_comparison_section(sorted_regressions, show_significance=True))

            # Render charts for non-significant regressions
            if regressions:
                lines.append(f"\n❌ Regressions ({len(regressions)}) - not statistically significant:")
                sorted_regressions = sorted(regressions, key=lambda x: x.delta)
                lines.extend(self._render_test_comparison_section(sorted_regressions))

            # Report new tests
            if new_tests:
                lines.append(f"\n🆕 New tests ({len(new_tests)}):")
                for test in sorted(new_tests):
                    curr_test = current_variant.test_cases[test]
                    curr_metric = curr_test.metrics.get("score")
                    if curr_metric:
                        lines.append(f"  {test:35} {curr_metric.mean:.3f}")

            # Report removed tests
            if removed_tests:
                lines.append(f"\n🗑️  Removed tests ({len(removed_tests)}):")
                for test in sorted(removed_tests):
                    lines.append(f"  {test}")

            # Summary statistics
            total = len(all_tests) - len(new_tests) - len(removed_tests)
            if total > 0:
                lines.append(f"\n{'-'*80}")
                lines.append(f"SUMMARY")
                lines.append(f"{'-'*80}")
                lines.append(f"Total comparable tests: {total}")
                lines.append(f"Significant improvements: {len(sig_improvements):3} ({len(sig_improvements)/total*100:5.1f}%)")
                lines.append(f"Improvements (not sig):   {len(improvements):3} ({len(improvements)/total*100:5.1f}%)")
                lines.append(f"No significant change:    {len(no_change):3} ({len(no_change)/total*100:5.1f}%)")
                lines.append(f"Regressions (not sig):    {len(regressions):3} ({len(regressions)/total*100:5.1f}%)")
                lines.append(f"Significant regressions:  {len(sig_regressions):3} ({len(sig_regressions)/total*100:5.1f}%)")

                # Net change (significant only)
                sig_net = len(sig_improvements) - len(sig_regressions)
                if sig_net != 0:
                    net_sign = "+" if sig_net > 0 else ""
                    lines.append(f"\nNet significant change: {net_sign}{sig_net}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
