"""
Statistical analysis for experiment comparisons.
"""

from typing import Dict, Tuple
from scipy import stats

from .data import Metric


def perform_pairwise_tests_from_metrics(
    variant_metrics: Dict[str, Metric],
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
                    metric1.mean,
                    metric1.stddev,
                    metric1.n,
                    metric2.mean,
                    metric2.stddev,
                    metric2.n,
                    equal_var=False,
                )
                results[(v1, v2)] = (t_stat, p_val)
            else:
                results[(v1, v2)] = (0.0, 1.0)  # Not enough data

    return results


def calculate_significance_markers(
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
    sorted_variants = sorted(variant_cis.items(), key=lambda x: x[1][0], reverse=True)

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
