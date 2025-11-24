"""
Report generation for experiment analysis.

Contains functions for generating comparison reports and baseline comparisons.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

from .data import ExperimentResults, Metric
from .aggregation import (
    collect_metric_means,
    means_to_metrics,
    metrics_to_cis,
    collect_comparative_scores,
    _extract_category_from_test_case,
)
from .charts import render_error_bar_chart, render_comparative_metric_chart
from .statistics import (
    perform_pairwise_tests_from_metrics,
    calculate_significance_markers,
)


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


def _render_metric_section(
    variant_metrics: dict[str, Metric],
    metric_name: str,
    include_significance: bool = True,
) -> List[str]:
    """
    Render a complete metric section with CIs, significance tests, and chart.

    Args:
        variant_metrics: Dict mapping variant_name -> Metric
        metric_name: Name of the metric
        include_significance: Whether to calculate and show significance markers

    Returns:
        List of chart lines
    """
    if not variant_metrics:
        return []

    # Convert to CIs
    variant_cis = metrics_to_cis(variant_metrics)

    # Calculate significance if requested
    significance_markers = None
    if include_significance and len(variant_metrics) > 1:
        pairwise_tests = perform_pairwise_tests_from_metrics(variant_metrics)
        significance_markers = calculate_significance_markers(
            variant_cis, pairwise_tests
        )

    # Render chart
    return render_error_bar_chart(variant_cis, metric_name, significance_markers)


def generate_comparison_report(runs: List[ExperimentResults]) -> str:
    """
    Generate comparative analysis report with error bars and statistical significance.

    Args:
        runs: List of experiment results to compare (can be from different runs/times)

    Returns:
        Formatted comparison report string
    """
    lines = []

    # Merge all runs into a single structure for display
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
        variants=merged_variants,
    )

    # Auto-discover per-run metrics from the results
    metrics_to_compare = set()
    for variant_results in merged_results.variants.values():
        for test_metrics in variant_results.test_cases.values():
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
        category = _extract_category_from_test_case(test_case)
        category_counts[category] = category_counts.get(category, 0) + 1

    # PER-CATEGORY ANALYSIS (shown first for bottom-up reading)
    for category in sorted(category_counts.keys()):
        lines.append("\n" + "=" * 80)
        lines.append(f"CATEGORY: {category} ({category_counts[category]} test cases)")
        lines.append("=" * 80)

        # Per-run metrics with error bars
        for metric_name in metrics_to_compare:
            variant_means = collect_metric_means(
                merged_results, metric_name, category_filter=category
            )
            variant_metrics = means_to_metrics(variant_means)

            if variant_metrics:
                chart_lines = _render_metric_section(
                    variant_metrics, metric_name, include_significance=True
                )
                lines.extend(chart_lines)

        # Comparative metrics (no error bars)
        for metric_name in comparative_metrics:
            variant_scores = collect_comparative_scores(
                merged_results, metric_name, category_filter=category
            )

            if variant_scores:
                chart_lines = render_comparative_metric_chart(
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
        variant_means = collect_metric_means(merged_results, metric_name)
        variant_metrics = means_to_metrics(variant_means)

        if variant_metrics:
            chart_lines = _render_metric_section(
                variant_metrics, metric_name, include_significance=True
            )
            lines.extend(chart_lines)

    # Comparative metrics (no error bars)
    for metric_name in comparative_metrics:
        variant_scores = collect_comparative_scores(merged_results, metric_name)

        if variant_scores:
            chart_lines = render_comparative_metric_chart(variant_scores, metric_name)
            lines.extend(chart_lines)

    # Add duration analysis
    lines.append("\n" + "=" * 80)
    lines.append("EXECUTION TIME (seconds per test case)")
    lines.append("=" * 80)

    # Collect and render duration
    variant_means = collect_metric_means(merged_results, "duration")
    variant_metrics = means_to_metrics(variant_means)

    if variant_metrics:
        chart_lines = _render_metric_section(
            variant_metrics, "duration (seconds)", include_significance=False
        )
        lines.extend(chart_lines)

    # Statistical significance legend
    lines.append("\n" + "-" * 80)
    lines.append("STATISTICAL SIGNIFICANCE")
    lines.append("* p<0.05  ** p<0.01  *** p<0.001")
    lines.append("(compared to lowest-ranked variant)")
    lines.append("=" * 80)

    return "\n".join(lines)


def _render_test_comparison_section(
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
            f"{test.test_name} (current)": (
                test.current_mean,
                test.current_ci[0],
                test.current_ci[1],
            ),
            f"{test.test_name} (baseline)": (
                test.baseline_mean,
                test.baseline_ci[0],
                test.baseline_ci[1],
            ),
        }

        significance_markers = None
        if show_significance:
            sig_marker = (
                "***" if test.p_value < 0.001 else "**" if test.p_value < 0.01 else "*"
            )
            significance_markers = {f"{test.test_name} (current)": sig_marker}

        chart_lines = render_error_bar_chart(
            variant_cis, "SCORE", significance_markers, max_name_len=max_test_name_len
        )
        lines.extend(chart_lines)

    return lines


def generate_baseline_comparison(
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
            set(current_results.variants.keys()) & set(baseline_results.variants.keys())
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

            # Use pre-calculated CIs from Metric objects
            current_ci = current_metric.confidence_interval()
            baseline_ci = baseline_metric.confidence_interval()

            # Perform t-test using Metric summary statistics
            if current_metric.n > 1 and baseline_metric.n > 1:
                t_stat, p_value = stats.ttest_ind_from_stats(
                    current_metric.mean,
                    current_metric.stddev,
                    current_metric.n,
                    baseline_metric.mean,
                    baseline_metric.stddev,
                    baseline_metric.n,
                    equal_var=False,  # Welch's t-test
                )
            else:
                p_value = 1.0  # Not enough data for significance test

            delta = current_metric.mean - baseline_metric.mean

            # Categorize based on significance and direction
            test_comparison = TestComparison(
                test_name=test_name,
                current_mean=current_metric.mean,
                current_ci=current_ci,
                baseline_mean=baseline_metric.mean,
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
            lines.append(
                f"\n✅ SIGNIFICANT IMPROVEMENTS ({len(sig_improvements)}) - p < 0.05:"
            )
            sorted_improvements = sorted(
                sig_improvements, key=lambda x: x.delta, reverse=True
            )
            lines.extend(
                _render_test_comparison_section(
                    sorted_improvements, show_significance=True
                )
            )

        # Render charts for non-significant improvements
        if improvements:
            lines.append(
                f"\n✅ Improvements ({len(improvements)}) - not statistically significant:"
            )
            sorted_improvements = sorted(
                improvements, key=lambda x: x.delta, reverse=True
            )
            lines.extend(_render_test_comparison_section(sorted_improvements))

        # Render charts for significant regressions
        if sig_regressions:
            lines.append(
                f"\n❌ SIGNIFICANT REGRESSIONS ({len(sig_regressions)}) - p < 0.05:"
            )
            sorted_regressions = sorted(sig_regressions, key=lambda x: x.delta)
            lines.extend(
                _render_test_comparison_section(
                    sorted_regressions, show_significance=True
                )
            )

        # Render charts for non-significant regressions
        if regressions:
            lines.append(
                f"\n❌ Regressions ({len(regressions)}) - not statistically significant:"
            )
            sorted_regressions = sorted(regressions, key=lambda x: x.delta)
            lines.extend(_render_test_comparison_section(sorted_regressions))

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
            lines.append(
                f"Significant improvements: {len(sig_improvements):3} ({len(sig_improvements)/total*100:5.1f}%)"
            )
            lines.append(
                f"Improvements (not sig):   {len(improvements):3} ({len(improvements)/total*100:5.1f}%)"
            )
            lines.append(
                f"No significant change:    {len(no_change):3} ({len(no_change)/total*100:5.1f}%)"
            )
            lines.append(
                f"Regressions (not sig):    {len(regressions):3} ({len(regressions)/total*100:5.1f}%)"
            )
            lines.append(
                f"Significant regressions:  {len(sig_regressions):3} ({len(sig_regressions)/total*100:5.1f}%)"
            )

            # Net change (significant only)
            sig_net = len(sig_improvements) - len(sig_regressions)
            if sig_net != 0:
                net_sign = "+" if sig_net > 0 else ""
                lines.append(f"\nNet significant change: {net_sign}{sig_net}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
