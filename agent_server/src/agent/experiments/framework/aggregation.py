"""
Metric aggregation helpers for experiment analysis.
"""

import statistics
from typing import Dict, List, Optional

from .data import ExperimentResults, Metric


def _extract_category_from_test_case(test_case_name: str) -> str:
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


def collect_metric_means(
    results: ExperimentResults,
    metric_name: str,
    category_filter: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Collect metric means across test cases for each variant.

    Args:
        results: Experiment results
        metric_name: Name of metric to collect (can be 'duration' for duration.mean)
        category_filter: Optional category to filter by

    Returns:
        Dict mapping variant_name -> list of means across test cases
    """
    variant_means: Dict[str, List[float]] = {}

    for variant_name, variant_results in results.variants.items():
        variant_means[variant_name] = []

        for test_case_name, test_metrics in variant_results.test_cases.items():
            # Apply category filter if specified
            if category_filter is not None:
                test_category = _extract_category_from_test_case(test_case_name)
                if test_category != category_filter:
                    continue

            # Handle special case for duration
            if metric_name == "duration":
                variant_means[variant_name].append(test_metrics.duration.mean)
            else:
                metric = test_metrics.metrics.get(metric_name)
                if metric:
                    variant_means[variant_name].append(metric.mean)

    return variant_means


def means_to_metrics(variant_means: Dict[str, List[float]]) -> Dict[str, Metric]:
    """
    Convert lists of means to Metric objects.

    Args:
        variant_means: Dict mapping variant_name -> list of means

    Returns:
        Dict mapping variant_name -> Metric
    """
    variant_metrics = {}
    for variant_name, means in variant_means.items():
        if means:
            variant_metrics[variant_name] = Metric.from_values(means)
    return variant_metrics


def metrics_to_cis(
    variant_metrics: Dict[str, Metric],
) -> Dict[str, tuple[float, float, float]]:
    """
    Convert Metric objects to (mean, lower_ci, upper_ci) tuples.

    Args:
        variant_metrics: Dict mapping variant_name -> Metric

    Returns:
        Dict mapping variant_name -> (mean, lower_ci, upper_ci)
    """
    variant_cis = {}
    for variant_name, metric in variant_metrics.items():
        lower, upper = metric.confidence_interval()
        variant_cis[variant_name] = (metric.mean, lower, upper)
    return variant_cis


def collect_comparative_scores(
    results: ExperimentResults,
    metric_name: str,
    category_filter: Optional[str] = None,
) -> Dict[str, float]:
    """
    Collect and average comparative metric scores.

    Args:
        results: Experiment results
        metric_name: Name of comparative metric
        category_filter: Optional category to filter by

    Returns:
        Dict mapping variant_name -> averaged score
    """
    variant_scores: Dict[str, List[float]] = {}

    for variant_name, variant_results in results.variants.items():
        variant_scores[variant_name] = []

        for test_case_name, test_metrics in variant_results.test_cases.items():
            # Apply category filter if specified
            if category_filter is not None:
                test_category = _extract_category_from_test_case(test_case_name)
                if test_category != category_filter:
                    continue

            if metric_name in test_metrics.comparative_metrics:
                variant_scores[variant_name].append(
                    test_metrics.comparative_metrics[metric_name]
                )

    # Average the scores for each variant
    averaged_scores = {}
    for variant_name, scores in variant_scores.items():
        if scores:
            averaged_scores[variant_name] = statistics.mean(scores)

    return averaged_scores
