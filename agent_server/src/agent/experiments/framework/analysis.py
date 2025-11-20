"""
Experiment analyzer for calculating metrics from saved data.

Loads raw experiment data and calculates metrics on-demand,
allowing recalculation without re-running expensive experiments.
"""

import logging
from typing import Dict, List, Tuple, Optional
import statistics
from functools import lru_cache
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
        self._temp_calculator: Optional[MetricsCalculator] = None

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

                for run_index in run_indices:
                    # Load run - type deserialization handled automatically
                    run_data, metadata = self.storage.load_run(
                        run_ts, variant_name, test_case_name, run_index
                    )

                    metadata_list.append(metadata)

                    # Calculate metrics if run was successful
                    if metadata.success and run_data.output_data is not None:
                        metrics = calculator.calculate(
                            run_data.output_data, run_data.expected_output
                        )
                        run_metrics_list.append(metrics)

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

    @lru_cache(maxsize=None)
    def _get_raw_metric_values_cached(
        self,
        run_ts: str,
        variant_name: str,
        test_case_name: str,
        metric_name: str,
    ) -> Tuple[float, ...]:
        """
        Calculate raw metric values for all runs (cached version).

        Returns tuple instead of list for hashability.
        Uses self._temp_calculator which must be set before calling.
        """
        if self._temp_calculator is None:
            raise RuntimeError(
                "_temp_calculator must be set before calling this method"
            )

        run_indices = self.storage.list_runs(run_ts, variant_name, test_case_name)
        values = []

        for run_index in run_indices:
            # Load run data and metadata
            run_data, metadata = self.storage.load_run(
                run_ts, variant_name, test_case_name, run_index
            )

            # Only calculate metrics for successful runs with output data
            if metadata.success and run_data.output_data is not None:
                metrics = self._temp_calculator.calculate(
                    run_data.output_data, run_data.expected_output
                )
                if metric_name in metrics:
                    value = metrics[metric_name]
                    # Filter out NaN values
                    if not np.isnan(value):
                        values.append(value)

        return tuple(values)

    def _get_raw_metric_values(
        self,
        run_ts: str,
        variant_name: str,
        test_case_name: str,
        metric_name: str,
    ) -> List[float]:
        """
        Calculate raw metric values for all runs of a specific variant/test case.

        Args:
            run_ts: Experiment run timestamp
            variant_name: Name of the variant
            test_case_name: Name of the test case
            metric_name: Name of the metric to extract

        Returns:
            List of metric values from all successful runs
        """
        # Use cached version and convert back to list
        return list(
            self._get_raw_metric_values_cached(
                run_ts, variant_name, test_case_name, metric_name
            )
        )

    def _calculate_confidence_interval(
        self, values: List[float], confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate mean and confidence interval.

        Args:
            values: List of metric values
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        if not values:
            return (0.0, 0.0, 0.0)

        if len(values) == 1:
            return (values[0], values[0], values[0])

        # Check if all values are identical (or very close)
        std_dev = float(np.std(values))
        if std_dev < 1e-10:  # Essentially zero
            mean_val = float(np.mean(values))
            return (mean_val, mean_val, mean_val)

        mean_val = float(np.mean(values))
        std_err = stats.sem(values)  # Standard error of the mean

        # Handle case where std_err is invalid
        if std_err == 0 or np.isnan(std_err) or np.isinf(std_err):
            return (mean_val, mean_val, mean_val)

        try:
            ci = stats.t.interval(
                confidence, len(values) - 1, loc=mean_val, scale=std_err
            )
            lower = float(ci[0])
            upper = float(ci[1])

            # Check for NaN or infinite results
            if np.isnan(lower) or np.isnan(upper) or np.isinf(lower) or np.isinf(upper):
                return (mean_val, mean_val, mean_val)

            return (mean_val, lower, upper)
        except Exception:
            # Fallback to just the mean if CI calculation fails
            return (mean_val, mean_val, mean_val)

    def _perform_pairwise_tests(
        self, variant_values: Dict[str, List[float]]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """
        Perform pairwise t-tests between all variants.

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
        min_val = min(valid_scores.values())
        max_val = max(valid_scores.values())

        # Add some padding
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1.0
        min_val -= range_val * 0.05
        max_val += range_val * 0.05
        range_val = max_val - min_val

        # Sort variants by score (descending)
        sorted_variants = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)

        # Header with metric name
        lines.append(f"\n{metric_name.upper()} (comparative)")

        # Scale labels
        scale_line = " " * 16  # Space for variant names
        for i in range(5):
            val = min_val + (range_val * i / 4)
            pos = int(width * i / 4)
            label = f"{val:.2f}"
            scale_line += " " * max(0, pos - len(scale_line) + 16) + label
        lines.append(scale_line)

        # Render each variant
        max_name_len = max(len(v) for v in valid_scores.keys())

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

        min_val = min(all_values)
        max_val = max(all_values)

        # Add some padding
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1.0
        min_val -= range_val * 0.05
        max_val += range_val * 0.05
        range_val = max_val - min_val

        # Sort variants by mean value (descending), filter out NaN
        sorted_variants = sorted(
            [(k, v) for k, v in variant_data.items() if not np.isnan(v[0])],
            key=lambda x: x[1][0],
            reverse=True,
        )

        if not sorted_variants:
            lines.append("(No valid data for this metric)")
            return lines

        # Header with metric name
        lines.append(f"\n{metric_name.upper()}")

        # Scale labels
        scale_line = " " * 16  # Space for variant names
        for i in range(5):
            val = min_val + (range_val * i / 4)
            pos = int(width * i / 4)
            label = f"{val:.2f}"
            scale_line += " " * max(0, pos - len(scale_line) + 16) + label
        lines.append(scale_line)

        # Render each variant
        max_name_len = max(len(v) for v in variant_data.keys())

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
        run_ts: str,
        metric_name: str,
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Aggregate metric values by category.

        Args:
            results: Experiment results
            run_ts: Experiment run timestamp
            metric_name: Name of metric to aggregate

        Returns:
            Dict mapping category -> variant_name -> list of values
        """
        category_data: Dict[str, Dict[str, List[float]]] = {}

        for variant_name, variant_results in results.variants.items():
            for test_case_name in variant_results.test_cases.keys():
                category = self._extract_category_from_test_case(test_case_name)

                if category not in category_data:
                    category_data[category] = {}

                if variant_name not in category_data[category]:
                    category_data[category][variant_name] = []

                # Load raw values for this test case
                values = self._get_raw_metric_values(
                    run_ts, variant_name, test_case_name, metric_name
                )
                category_data[category][variant_name].extend(values)

        return category_data

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
        results: ExperimentResults,
        calculator: MetricsCalculator,
    ) -> str:
        """
        Generate comparative analysis report with error bars and statistical significance.

        Args:
            results: Results from calculate_metrics()
            calculator: MetricsCalculator used to compute metrics

        Returns:
            Formatted comparison report string
        """
        # Set temp calculator for cached method
        self._temp_calculator = calculator
        # Clear the cache for a fresh analysis
        self._get_raw_metric_values_cached.cache_clear()

        try:
            lines = []
            run_ts = results.run_ts

            # Auto-discover per-run metrics from the results
            metrics_to_compare = set()
            for variant_results in results.variants.values():
                for test_metrics in variant_results.test_cases.values():
                    for key in test_metrics.metrics.keys():
                        if key.endswith("_mean"):
                            metric_name = key.replace("_mean", "")
                            metrics_to_compare.add(metric_name)

            metrics_to_compare = sorted(metrics_to_compare)

            # Auto-discover comparative metrics
            comparative_metrics = set()
            for variant_results in results.variants.values():
                for test_metrics in variant_results.test_cases.values():
                    comparative_metrics.update(test_metrics.comparative_metrics.keys())

            comparative_metrics = sorted(comparative_metrics)

            lines.append("=" * 80)
            lines.append("COMPARATIVE EXPERIMENT ANALYSIS")
            lines.append("=" * 80)
            lines.append(f"\nExperiment: {run_ts}")
            lines.append(f"Variants: {len(results.variants)}")

            # Count test cases per category
            all_test_cases = set()
            for variant_results in results.variants.values():
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
                    # Aggregate values by category
                    category_data = self._aggregate_by_category(
                        results, run_ts, metric_name
                    )

                    if category not in category_data:
                        continue

                    variant_values = category_data[category]

                    # Skip if no valid data
                    if not variant_values or all(
                        not v for v in variant_values.values()
                    ):
                        continue

                    # Calculate CIs for each variant
                    variant_cis = {}
                    for variant_name, values in variant_values.items():
                        if values:
                            variant_cis[variant_name] = (
                                self._calculate_confidence_interval(values)
                            )

                    # Perform statistical tests
                    pairwise_tests = self._perform_pairwise_tests(variant_values)

                    # Determine significance markers (compare all to lowest-ranked)
                    sorted_variants = sorted(
                        variant_cis.items(), key=lambda x: x[1][0], reverse=True
                    )
                    if len(sorted_variants) > 1:
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
                    else:
                        significance_markers = {}

                    # Render error bar chart
                    chart_lines = self._render_error_bar_chart(
                        variant_cis, metric_name, significance_markers
                    )
                    lines.extend(chart_lines)

                # Comparative metrics (no error bars)
                for metric_name in comparative_metrics:
                    category_scores = self._aggregate_comparative_by_category(
                        results, metric_name
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
                # Aggregate all values across all test cases for each variant
                variant_values = {}

                for variant_name, variant_results in results.variants.items():
                    variant_values[variant_name] = []
                    for test_case_name in variant_results.test_cases.keys():
                        values = self._get_raw_metric_values(
                            run_ts, variant_name, test_case_name, metric_name
                        )
                        variant_values[variant_name].extend(values)

                # Skip if no valid data
                if not variant_values or all(not v for v in variant_values.values()):
                    continue

                # Calculate CIs for each variant
                variant_cis = {}
                for variant_name, values in variant_values.items():
                    if values:
                        variant_cis[variant_name] = self._calculate_confidence_interval(
                            values
                        )

                # Perform statistical tests
                pairwise_tests = self._perform_pairwise_tests(variant_values)

                # Determine significance markers
                sorted_variants = sorted(
                    variant_cis.items(), key=lambda x: x[1][0], reverse=True
                )
                if len(sorted_variants) > 1:
                    baseline_variant = sorted_variants[-1][0]
                    significance_markers = {}

                    for variant_name, _ in sorted_variants[:-1]:
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
                else:
                    significance_markers = {}

                # Render error bar chart
                chart_lines = self._render_error_bar_chart(
                    variant_cis, metric_name, significance_markers
                )
                lines.extend(chart_lines)

            # Comparative metrics (no error bars)
            for metric_name in comparative_metrics:
                # Aggregate all comparative metric scores across all test cases
                variant_scores: Dict[str, List[float]] = {}

                for variant_name, variant_results in results.variants.items():
                    variant_scores[variant_name] = []
                    for test_metrics in variant_results.test_cases.values():
                        if metric_name in test_metrics.comparative_metrics:
                            variant_scores[variant_name].append(
                                test_metrics.comparative_metrics[metric_name]
                            )

                # Average the scores for each variant
                averaged_scores = {}
                for variant_name, scores in variant_scores.items():
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

            # Collect duration values from metadata
            variant_durations = {}
            for variant_name, variant_results in results.variants.items():
                variant_durations[variant_name] = []
                for test_case_name in variant_results.test_cases.keys():
                    run_indices = self.storage.list_runs(
                        run_ts, variant_name, test_case_name
                    )
                    for run_index in run_indices:
                        run_path = (
                            self.storage.base_dir
                            / run_ts
                            / f"variant_{variant_name}"
                            / f"testcase_{test_case_name}"
                            / f"run_{run_index}"
                        )
                        _, metadata = self.storage.load_run(run_path)
                        if metadata.success:
                            variant_durations[variant_name].append(
                                metadata.duration_seconds
                            )

            # Calculate CIs for duration
            duration_cis = {}
            for variant_name, durations in variant_durations.items():
                if durations:
                    duration_cis[variant_name] = self._calculate_confidence_interval(
                        durations
                    )

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
        finally:
            # Clear temp calculator
            self._temp_calculator = None
