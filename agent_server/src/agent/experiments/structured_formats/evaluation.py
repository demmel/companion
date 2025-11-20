"""
Evaluation functions for comparing extracted vs expected results.

Calculates precision, recall, and F1 scores.
"""

from typing import Tuple, Any, Type, List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


def evaluate_correctness(
    extracted: BaseModel, expected: BaseModel
) -> Tuple[float, float, float]:
    """
    Evaluate correctness of extracted result against ground truth.

    Args:
        extracted: Result from LLM extraction
        expected: Ground truth expected result

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    # Convert to dicts for comparison
    extracted_dict = extracted.model_dump()
    expected_dict = expected.model_dump()

    # Calculate field-by-field correctness
    true_positives, false_positives, false_negatives = _compare_dicts(
        extracted_dict, expected_dict
    )

    # Calculate metrics
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def calculate_richness(result: BaseModel, model: Type[BaseModel]) -> float:
    """
    Calculate richness score: percentage of optional fields that were populated.

    Args:
        result: Extracted result
        model: Pydantic model class

    Returns:
        Richness score (0.0-1.0)
    """
    result_dict = result.model_dump()

    total_optional = 0
    populated_optional = 0

    for field_name, field_info in model.model_fields.items():
        # Check if field is optional (has default or default_factory)
        if field_info.default is not None or field_info.default_factory is not None:
            total_optional += 1

            # Check if populated in result
            value = result_dict.get(field_name)
            if value is not None and value != "":
                if isinstance(value, list) and len(value) > 0:
                    populated_optional += 1
                elif not isinstance(value, list):
                    populated_optional += 1

    if total_optional == 0:
        return 1.0  # No optional fields = fully rich

    return populated_optional / total_optional


def _compare_dicts(
    extracted: dict, expected: dict, path: str = ""
) -> Tuple[int, int, int]:
    """
    Recursively compare two dictionaries.

    Args:
        extracted: Extracted data
        expected: Expected data
        path: Current field path (for logging)

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    tp = 0
    fp = 0
    fn = 0

    # Fields in expected but not in extracted = false negatives
    for key in expected:
        if key not in extracted:
            fn += 1
            continue

        expected_value = expected[key]
        extracted_value = extracted[key]

        # Handle None/null
        if expected_value is None:
            if extracted_value is None:
                tp += 1
            else:
                fp += 1
            continue

        # Handle lists
        if isinstance(expected_value, list):
            if not isinstance(extracted_value, list):
                fp += 1
                continue

            # Compare lists (order-independent for now)
            tp_list, fp_list, fn_list = _compare_lists(extracted_value, expected_value)
            tp += tp_list
            fp += fp_list
            fn += fn_list
            continue

        # Handle nested dicts
        if isinstance(expected_value, dict):
            if not isinstance(extracted_value, dict):
                fp += 1
                continue

            tp_dict, fp_dict, fn_dict = _compare_dicts(
                extracted_value, expected_value, f"{path}.{key}" if path else key
            )
            tp += tp_dict
            fp += fp_dict
            fn += fn_dict
            continue

        # Handle primitives (string, int, float, bool)
        if _values_match(extracted_value, expected_value):
            tp += 1
        else:
            fp += 1

    # Fields in extracted but not in expected = false positives
    for key in extracted:
        if key not in expected:
            fp += 1

    return tp, fp, fn


def _compare_lists(extracted: List[Any], expected: List[Any]) -> Tuple[int, int, int]:
    """
    Compare two lists (order-independent matching).

    Args:
        extracted: Extracted list
        expected: Expected list

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    # For lists of primitives, use set comparison
    if extracted and not isinstance(extracted[0], dict):
        extracted_set = set(extracted)
        expected_set = set(expected)

        tp = len(extracted_set & expected_set)
        fp = len(extracted_set - expected_set)
        fn = len(expected_set - extracted_set)

        return tp, fp, fn

    # For lists of dicts/objects, match greedily
    expected_matched = [False] * len(expected)
    extracted_matched = [False] * len(extracted)

    tp = 0
    fp = 0
    fn = 0

    # Try to match each extracted item to an expected item
    for i, extracted_item in enumerate(extracted):
        best_match = -1
        best_score = 0

        for j, expected_item in enumerate(expected):
            if expected_matched[j]:
                continue

            # Calculate similarity
            if isinstance(extracted_item, dict) and isinstance(expected_item, dict):
                item_tp, item_fp, item_fn = _compare_dicts(
                    extracted_item, expected_item
                )
                score = item_tp - item_fp - item_fn
                if score > best_score:
                    best_score = score
                    best_match = j

        if best_match >= 0:
            expected_matched[best_match] = True
            extracted_matched[i] = True
            # Count this as true positive
            tp += 1
        else:
            # No good match = false positive
            fp += 1

    # Unmatched expected items = false negatives
    fn = sum(1 for matched in expected_matched if not matched)

    return tp, fp, fn


def _values_match(extracted: Any, expected: Any) -> bool:
    """
    Check if two values match (with some tolerance for strings).

    Args:
        extracted: Extracted value
        expected: Expected value

    Returns:
        True if values match
    """
    # Exact match
    if extracted == expected:
        return True

    # String comparison (case-insensitive, whitespace-normalized)
    if isinstance(extracted, str) and isinstance(expected, str):
        return extracted.lower().strip() == expected.lower().strip()

    # Numeric comparison (with small tolerance)
    if isinstance(extracted, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(extracted) - float(expected)) < 0.01

    return False
