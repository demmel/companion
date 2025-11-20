"""
Flexible evaluation system with lenient scoring.

The strict exact-match evaluation gives low F1 scores even when extraction is semantically correct.
This module provides more flexible scoring that recognizes semantic similarity.
"""

from typing import Tuple, Any, Dict, List, Set
from pydantic import BaseModel
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


def flexible_evaluate(
    extracted: BaseModel, expected: BaseModel, strict: bool = False
) -> Tuple[float, float, float]:
    """
    Flexible evaluation that's more lenient than exact matching.

    Args:
        extracted: Result from LLM extraction
        expected: Ground truth expected result
        strict: If True, use exact matching (original behavior)

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    extracted_dict = extracted.model_dump()
    expected_dict = expected.model_dump()

    if strict:
        # Use original strict evaluation
        from .evaluation import evaluate_correctness

        return evaluate_correctness(extracted, expected)

    # Flexible evaluation
    tp, fp, fn = _flexible_compare_dicts(extracted_dict, expected_dict)

    # Calculate metrics
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    logger.debug(
        f"Flexible eval: TP={tp}, FP={fp}, FN={fn} → P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}"
    )

    return precision, recall, f1


def _flexible_compare_dicts(
    extracted: dict, expected: dict, path: str = ""
) -> Tuple[float, float, float]:
    """
    Recursively compare dictionaries with flexible matching.

    Returns fractional TP/FP/FN to give partial credit.
    """
    tp = 0.0
    fp = 0.0
    fn = 0.0

    # Check each expected field
    for key in expected:
        if key not in extracted:
            fn += 1.0
            continue

        expected_value = expected[key]
        extracted_value = extracted[key]

        # Handle None/null
        if expected_value is None:
            if extracted_value is None:
                tp += 1.0
            else:
                fp += 0.5  # Partial penalty - provided value for optional field
            continue

        # Handle lists (special case for facts, entities, etc.)
        if isinstance(expected_value, list):
            if not isinstance(extracted_value, list):
                fp += 1.0
                continue

            tp_list, fp_list, fn_list = _flexible_compare_lists(
                extracted_value, expected_value, f"{path}.{key}" if path else key
            )
            tp += tp_list
            fp += fp_list
            fn += fn_list
            continue

        # Handle nested dicts
        if isinstance(expected_value, dict):
            if not isinstance(extracted_value, dict):
                fp += 1.0
                continue

            tp_dict, fp_dict, fn_dict = _flexible_compare_dicts(
                extracted_value, expected_value, f"{path}.{key}" if path else key
            )
            tp += tp_dict
            fp += fp_dict
            fn += fn_dict
            continue

        # Handle primitives with fuzzy matching
        match_score = _flexible_value_match(extracted_value, expected_value, key)
        tp += match_score
        if match_score < 1.0:
            fp += 1.0 - match_score

    # Fields in extracted but not expected = false positives
    for key in extracted:
        if key not in expected:
            fp += 0.5  # Partial penalty - extra info isn't as bad as wrong info

    return tp, fp, fn


def _flexible_compare_lists(
    extracted: List[Any], expected: List[Any], field_name: str
) -> Tuple[float, float, float]:
    """
    Compare lists with flexible matching and partial credit.
    """
    # For lists of primitives, use fuzzy set matching
    if extracted and not isinstance(extracted[0], dict):
        return _fuzzy_set_compare(extracted, expected)

    # For lists of dicts/objects, match with partial credit
    expected_scores = [0.0] * len(expected)  # Best match score for each expected item
    extracted_used = [False] * len(extracted)

    # Try to match each expected item to extracted items
    for i, expected_item in enumerate(expected):
        best_score = 0.0
        best_idx = -1

        for j, extracted_item in enumerate(extracted):
            if extracted_used[j]:
                continue

            if isinstance(expected_item, dict) and isinstance(extracted_item, dict):
                # Calculate similarity score
                score = _dict_similarity(extracted_item, expected_item)
                if score > best_score:
                    best_score = score
                    best_idx = j

        if best_idx >= 0:
            expected_scores[i] = best_score
            extracted_used[best_idx] = True

    # Calculate metrics
    tp = sum(expected_scores)  # Sum of match scores
    fn = len(expected) - tp  # Unmatched expected items
    fp = sum(1.0 for used in extracted_used if not used)  # Unmatched extracted items

    return tp, fp, fn


def _fuzzy_set_compare(
    extracted: List[Any], expected: List[Any]
) -> Tuple[float, float, float]:
    """Compare lists as fuzzy sets with string similarity."""
    extracted_strs = [str(v).lower() for v in extracted]
    expected_strs = [str(v).lower() for v in expected]

    matched_expected = set()
    matched_extracted = set()

    # Find fuzzy matches
    for i, exp_str in enumerate(expected_strs):
        best_score = 0.0
        best_j = -1

        for j, ext_str in enumerate(extracted_strs):
            if j in matched_extracted:
                continue

            score = SequenceMatcher(None, exp_str, ext_str).ratio()
            if score > 0.8:  # 80% similarity threshold
                if score > best_score:
                    best_score = score
                    best_j = j

        if best_j >= 0:
            matched_expected.add(i)
            matched_extracted.add(best_j)

    tp = len(matched_expected)
    fn = len(expected) - tp
    fp = len(extracted) - len(matched_extracted)

    return float(tp), float(fp), float(fn)


def _dict_similarity(extracted: dict, expected: dict) -> float:
    """
    Calculate similarity score (0.0-1.0) between two dicts.

    Uses flexible matching on field values.
    """
    if not expected:
        return 1.0 if not extracted else 0.0

    scores = []

    for key in expected:
        if key not in extracted:
            scores.append(0.0)
            continue

        match_score = _flexible_value_match(extracted[key], expected[key], key)
        scores.append(match_score)

    # Penalize extra fields (but less than missing fields)
    extra_fields = len(extracted) - len(expected)
    if extra_fields > 0:
        penalty = 0.1 * extra_fields / len(expected)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return max(0.0, avg_score - penalty)

    return sum(scores) / len(scores) if scores else 0.0


def _flexible_value_match(extracted: Any, expected: Any, field_name: str) -> float:
    """
    Check if two values match with fuzzy logic.

    Returns a score from 0.0 (no match) to 1.0 (perfect match).
    """
    # Exact match
    if extracted == expected:
        return 1.0

    # Both are strings - use fuzzy matching
    if isinstance(extracted, str) and isinstance(expected, str):
        # Case-insensitive, whitespace-normalized exact match
        if extracted.lower().strip() == expected.lower().strip():
            return 1.0

        # For field-specific fuzzy matching
        if field_name in ["predicate", "relationship", "action"]:
            # Predicates can vary but mean the same thing
            # e.g., "began_in" vs "started_in" vs "originated_in"
            return _predicate_similarity(extracted, expected)

        # General string similarity
        similarity = SequenceMatcher(None, extracted.lower(), expected.lower()).ratio()
        if similarity > 0.8:  # 80% similar
            return similarity

        return 0.0

    # Numeric comparison with tolerance
    if isinstance(extracted, (int, float)) and isinstance(expected, (int, float)):
        if abs(float(extracted) - float(expected)) < 0.01:
            return 1.0
        # Partial credit for close values
        diff = abs(float(extracted) - float(expected))
        max_val = max(abs(float(expected)), 1.0)
        if diff / max_val < 0.1:  # Within 10%
            return 0.8
        return 0.0

    return 0.0


def _predicate_similarity(pred_a: str, pred_b: str) -> float:
    """
    Calculate similarity between predicate names.

    Recognizes common synonyms and variations.
    """
    pred_a_lower = pred_a.lower().replace("_", " ").replace("-", " ")
    pred_b_lower = pred_b.lower().replace("_", " ").replace("-", " ")

    # Exact match after normalization
    if pred_a_lower == pred_b_lower:
        return 1.0

    # Common predicate synonyms
    synonyms = [
        {
            "began",
            "started",
            "commenced",
            "originated",
            "initiated",
            "began in",
            "started in",
            "originated in",
        },
        {"ended", "concluded", "finished", "terminated", "ended in"},
        {"traded", "exchanged", "bartered", "traded with"},
        {"ruled", "governed", "controlled", "led", "ruled by", "governed by"},
        {"created", "made", "produced", "built", "constructed"},
        {"discovered", "found", "uncovered", "identified"},
        {"located", "situated", "positioned", "placed", "located in", "situated in"},
    ]

    # Check if predicates are synonyms
    for synonym_set in synonyms:
        if pred_a_lower in synonym_set and pred_b_lower in synonym_set:
            return 0.9  # High score for synonyms

    # Fuzzy string match
    similarity = SequenceMatcher(None, pred_a_lower, pred_b_lower).ratio()
    if similarity > 0.7:
        return similarity

    return 0.0
