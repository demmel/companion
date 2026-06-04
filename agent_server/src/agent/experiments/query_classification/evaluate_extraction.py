"""Evaluate query extraction quality.

This module evaluates how well the extractor identifies references and assigns
correct query types. The evaluation focuses on:

1. Reference detection recall - Did we find all important references?
2. Query type accuracy - Is the query type correct for retrieval routing?
3. No-retrieval detection - Did we correctly identify no-retrieval cases?

Run:
    uv run python -m agent.experiments.query_classification.evaluate_extraction
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from agent.llm import LLM, SupportedModel, create_llm

from .create_extraction_dataset import (
    build_extraction_dataset,
    load_extraction_dataset,
)
from .extractor import ExtractorConfig, QueryExtractor
from .models import (
    ExtractedQuery,
    ExtractionDataset,
    ExtractionResult,
    LabeledExtractionExample,
    QueryType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Metrics
# =============================================================================


@dataclass
class ExtractionMetrics:
    """Metrics for evaluating query extraction."""

    # Reference detection
    total_expected_queries: int = 0
    total_extracted_queries: int = 0
    matched_queries: int = 0  # Queries that match expected (reference + type)
    reference_matches: int = 0  # Queries that match reference only

    # Query type accuracy (for matched references)
    type_correct: int = 0
    type_incorrect: int = 0
    type_confusion: dict[tuple[QueryType, QueryType], int] = field(
        default_factory=dict
    )

    # No-retrieval detection
    no_retrieval_expected: int = 0
    no_retrieval_predicted: int = 0
    no_retrieval_correct: int = 0  # True negatives

    # Per-type metrics
    per_type_expected: dict[QueryType, int] = field(default_factory=dict)
    per_type_extracted: dict[QueryType, int] = field(default_factory=dict)
    per_type_correct: dict[QueryType, int] = field(default_factory=dict)

    def compute_recall(self) -> float:
        """Compute reference detection recall."""
        if self.total_expected_queries == 0:
            return 1.0
        return self.reference_matches / self.total_expected_queries

    def compute_precision(self) -> float:
        """Compute extraction precision (matched / extracted)."""
        if self.total_extracted_queries == 0:
            return 1.0 if self.total_expected_queries == 0 else 0.0
        return self.reference_matches / self.total_extracted_queries

    def compute_type_accuracy(self) -> float:
        """Compute query type accuracy for matched references."""
        total = self.type_correct + self.type_incorrect
        if total == 0:
            return 1.0
        return self.type_correct / total

    def compute_no_retrieval_accuracy(self) -> float:
        """Compute accuracy for no-retrieval detection."""
        total = self.no_retrieval_expected + (
            self.total_expected_queries > 0
            and self.no_retrieval_expected == 0
        )
        if self.no_retrieval_expected == 0:
            # No no-retrieval cases in dataset
            return 1.0
        return self.no_retrieval_correct / self.no_retrieval_expected

    def compute_f1(self) -> float:
        """Compute F1 score for reference detection."""
        precision = self.compute_precision()
        recall = self.compute_recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


# =============================================================================
# Matching Logic
# =============================================================================


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def reference_matches(expected: ExtractedQuery, extracted: ExtractedQuery) -> bool:
    """Check if extracted query matches expected based on reference.

    We consider it a match if:
    - The references have significant overlap
    - OR the query texts have significant overlap

    This is fuzzy matching since exact phrasing may vary.
    """
    exp_ref = normalize_text(expected.reference)
    ext_ref = normalize_text(extracted.reference)
    exp_query = normalize_text(expected.query_text)
    ext_query = normalize_text(extracted.query_text)

    # Check for substring matches in either direction
    if exp_ref and ext_ref:
        if exp_ref in ext_ref or ext_ref in exp_ref:
            return True

    if exp_query and ext_query:
        # Check for significant word overlap
        exp_words = set(exp_query.split())
        ext_words = set(ext_query.split())
        overlap = exp_words & ext_words
        # Consider it a match if >50% of expected words are present
        if len(exp_words) > 0 and len(overlap) / len(exp_words) > 0.5:
            return True

    # Check if entity names match
    exp_words = set(exp_ref.split())
    ext_words = set(ext_ref.split())
    # If any significant word matches (not just 'the', 'my', etc.)
    stopwords = {"the", "my", "a", "an", "that", "this", "it"}
    exp_significant = exp_words - stopwords
    ext_significant = ext_words - stopwords
    if exp_significant & ext_significant:
        return True

    return False


def find_best_match(
    expected: ExtractedQuery,
    extracted_queries: list[ExtractedQuery],
    already_matched: set[int],
) -> tuple[int, bool]:
    """Find the best matching extracted query for an expected query.

    Returns:
        (index of match, type_matches) or (-1, False) if no match
    """
    for i, extracted in enumerate(extracted_queries):
        if i in already_matched:
            continue
        if reference_matches(expected, extracted):
            type_matches = expected.query_type == extracted.query_type
            return i, type_matches
    return -1, False


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_single_example(
    example: LabeledExtractionExample,
    result: ExtractionResult,
    metrics: ExtractionMetrics,
) -> dict[str, list[str]]:
    """Evaluate extraction result against expected queries.

    Returns a dict with details for logging.
    """
    details: dict[str, list[str]] = {
        "matched": [],
        "type_errors": [],
        "missed": [],
        "extra": [],
    }

    expected = example.expected_queries
    extracted = result.queries

    # Handle no-retrieval cases
    if len(expected) == 0:
        metrics.no_retrieval_expected += 1
        if len(extracted) == 0:
            metrics.no_retrieval_correct += 1
            metrics.no_retrieval_predicted += 1
        return details

    if len(extracted) == 0:
        metrics.no_retrieval_predicted += 1
        # All expected queries are missed
        for exp in expected:
            metrics.total_expected_queries += 1
            metrics.per_type_expected[exp.query_type] = (
                metrics.per_type_expected.get(exp.query_type, 0) + 1
            )
            details["missed"].append(f"{exp.query_text} ({exp.query_type.value})")
        return details

    # Update counts
    metrics.total_expected_queries += len(expected)
    metrics.total_extracted_queries += len(extracted)

    for exp in expected:
        metrics.per_type_expected[exp.query_type] = (
            metrics.per_type_expected.get(exp.query_type, 0) + 1
        )

    for ext in extracted:
        metrics.per_type_extracted[ext.query_type] = (
            metrics.per_type_extracted.get(ext.query_type, 0) + 1
        )

    # Match expected to extracted
    already_matched: set[int] = set()
    for exp in expected:
        match_idx, type_matches = find_best_match(exp, extracted, already_matched)

        if match_idx >= 0:
            already_matched.add(match_idx)
            metrics.reference_matches += 1

            if type_matches:
                metrics.type_correct += 1
                metrics.matched_queries += 1
                metrics.per_type_correct[exp.query_type] = (
                    metrics.per_type_correct.get(exp.query_type, 0) + 1
                )
                details["matched"].append(
                    f"{exp.query_text} ({exp.query_type.value})"
                )
            else:
                metrics.type_incorrect += 1
                ext = extracted[match_idx]
                confusion_key = (exp.query_type, ext.query_type)
                metrics.type_confusion[confusion_key] = (
                    metrics.type_confusion.get(confusion_key, 0) + 1
                )
                details["type_errors"].append(
                    f"{exp.query_text}: expected {exp.query_type.value}, "
                    f"got {ext.query_type.value}"
                )
        else:
            details["missed"].append(f"{exp.query_text} ({exp.query_type.value})")

    # Track extra queries (not matched to any expected)
    for i, ext in enumerate(extracted):
        if i not in already_matched:
            details["extra"].append(f"{ext.query_text} ({ext.query_type.value})")

    return details


def evaluate_extraction(
    extractor: QueryExtractor,
    dataset: ExtractionDataset,
    verbose: bool = True,
) -> ExtractionMetrics:
    """Evaluate extractor on a dataset.

    Args:
        extractor: The query extractor to evaluate
        dataset: Dataset of labeled examples
        verbose: Whether to print per-example results

    Returns:
        ExtractionMetrics with evaluation results
    """
    metrics = ExtractionMetrics()

    for i, example in enumerate(dataset.examples):
        if verbose:
            print(f"\n--- Example {i + 1}/{len(dataset.examples)} ---")
            print(f"Message: {example.message}")
            print(f"Context: {example.context}")
            print(f"Expected queries: {len(example.expected_queries)}")

        result = extractor.extract(
            message=example.message,
            context=example.context,
        )

        if verbose:
            print(f"Extracted queries: {len(result.queries)}")

        details = evaluate_single_example(example, result, metrics)

        if verbose:
            if details["matched"]:
                print(f"  Matched: {details['matched']}")
            if details["type_errors"]:
                print(f"  Type errors: {details['type_errors']}")
            if details["missed"]:
                print(f"  Missed: {details['missed']}")
            if details["extra"]:
                print(f"  Extra: {details['extra']}")

    return metrics


def print_metrics(metrics: ExtractionMetrics) -> None:
    """Print evaluation metrics in a readable format."""
    print("\n" + "=" * 60)
    print("Extraction Evaluation Results")
    print("=" * 60)

    print("\n## Overall Metrics")
    print(f"Reference Detection Recall: {metrics.compute_recall():.2%}")
    print(f"Extraction Precision: {metrics.compute_precision():.2%}")
    print(f"F1 Score: {metrics.compute_f1():.2%}")
    print(f"Query Type Accuracy: {metrics.compute_type_accuracy():.2%}")

    print("\n## Counts")
    print(f"Total expected queries: {metrics.total_expected_queries}")
    print(f"Total extracted queries: {metrics.total_extracted_queries}")
    print(f"Reference matches: {metrics.reference_matches}")
    print(f"Full matches (ref + type): {metrics.matched_queries}")

    print("\n## No-Retrieval Detection")
    print(f"Expected no-retrieval: {metrics.no_retrieval_expected}")
    print(f"Predicted no-retrieval: {metrics.no_retrieval_predicted}")
    print(f"Correct no-retrieval: {metrics.no_retrieval_correct}")

    print("\n## Per-Type Performance")
    all_types = set(metrics.per_type_expected.keys()) | set(
        metrics.per_type_extracted.keys()
    )
    for qt in sorted(all_types, key=lambda x: x.value):
        expected = metrics.per_type_expected.get(qt, 0)
        extracted = metrics.per_type_extracted.get(qt, 0)
        correct = metrics.per_type_correct.get(qt, 0)
        recall = correct / expected if expected > 0 else 0
        print(f"  {qt.value}:")
        print(f"    Expected: {expected}, Extracted: {extracted}, Correct: {correct}")
        print(f"    Recall: {recall:.2%}")

    if metrics.type_confusion:
        print("\n## Type Confusion Matrix")
        print("(expected -> predicted)")
        for (exp, pred), count in sorted(
            metrics.type_confusion.items(), key=lambda x: -x[1]
        ):
            print(f"  {exp.value} -> {pred.value}: {count}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run extraction evaluation."""
    logging.basicConfig(level=logging.INFO)

    # Try to load dataset, or build it if not exists
    experiment_dir = Path(__file__).parent
    dataset_path = experiment_dir / "output" / "dataset" / "extraction_v1.json"
    # Also check for older naming convention
    if not dataset_path.exists():
        dataset_path = experiment_dir / "output" / "dataset" / "extraction_extraction_v1.json"

    if dataset_path.exists():
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_extraction_dataset(dataset_path)
    else:
        logger.info("Building dataset...")
        dataset = build_extraction_dataset()

    print(f"\nLoaded {len(dataset.examples)} examples")

    # Create extractor
    logger.info("Creating extractor...")
    llm = create_llm()

    # Test with different models
    models_to_test = [
        SupportedModel.MISTRAL_SMALL_3_2_Q4,
        # Uncomment to test other models:
        # SupportedModel.CLAUDE_HAIKU_4_5,
    ]

    for model in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"Testing with model: {model.value}")
        print("=" * 60)

        config = ExtractorConfig(model=model)
        extractor = QueryExtractor(llm, config)

        # Run evaluation
        metrics = evaluate_extraction(extractor, dataset, verbose=True)

        # Print results
        print_metrics(metrics)

        # Save results
        results_dir = experiment_dir / "output" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"extraction_eval_{model.value}.json"

        results = {
            "model": model.value,
            "recall": metrics.compute_recall(),
            "precision": metrics.compute_precision(),
            "f1": metrics.compute_f1(),
            "type_accuracy": metrics.compute_type_accuracy(),
            "total_expected": metrics.total_expected_queries,
            "total_extracted": metrics.total_extracted_queries,
            "reference_matches": metrics.reference_matches,
            "full_matches": metrics.matched_queries,
            "no_retrieval_expected": metrics.no_retrieval_expected,
            "no_retrieval_predicted": metrics.no_retrieval_predicted,
            "no_retrieval_correct": metrics.no_retrieval_correct,
            "per_type_expected": {
                k.value: v for k, v in metrics.per_type_expected.items()
            },
            "per_type_correct": {
                k.value: v for k, v in metrics.per_type_correct.items()
            },
            "type_confusion": {
                f"{k[0].value}->{k[1].value}": v
                for k, v in metrics.type_confusion.items()
            },
        }

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
