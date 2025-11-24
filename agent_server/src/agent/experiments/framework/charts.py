"""
Chart rendering for experiment analysis reports.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def _calculate_chart_range(
    values: List[float], padding: float = 0.05
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
    min_val: float, range_val: float, width: int = 50, left_padding: int = 16
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


def render_comparative_metric_chart(
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
    min_val, max_val, range_val = _calculate_chart_range(list(valid_scores.values()))

    # Sort variants by score (descending)
    sorted_variants = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)

    # Render each variant
    max_name_len = max(len(v) for v in valid_scores.keys())

    # Header with metric name
    lines.append(f"\n{metric_name.upper()} (comparative)")

    # Scale labels - account for name column + value column
    left_padding = max_name_len + 8
    lines.append(_generate_scale_line(min_val, range_val, width, left_padding))

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


def render_error_bar_chart(
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
        max_name_len: Optional override for max name length (for alignment across charts)

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
    min_val, max_val, range_val = _calculate_chart_range(all_values)

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
    lines.append(_generate_scale_line(min_val, range_val, width, left_padding))

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

        line = f"{variant_name:<{max_name_len}}  {mean:.3f}  {chart_str}{sig_marker}"
        lines.append(line)

    return lines
