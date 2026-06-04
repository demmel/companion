"""Error analysis for query classification experiment."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from .create_dataset import load_dataset
from .models import QueryType

logger = logging.getLogger(__name__)


def load_predictions(path: Path) -> list[dict]:
    """Load predictions from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def analyze_errors(predictions: list[dict]) -> dict:
    """Analyze classification errors."""
    errors = [p for p in predictions if not p["correct"]]
    correct = [p for p in predictions if p["correct"]]

    # Error patterns
    error_patterns: dict[str, list[dict]] = defaultdict(list)
    for error in errors:
        pattern = f"{error['true_type']} -> {error['predicted_type']}"
        error_patterns[pattern].append(error)

    # Confidence analysis
    error_confidences = [e["confidence"] for e in errors]
    correct_confidences = [c["confidence"] for c in correct]

    avg_error_confidence = sum(error_confidences) / len(error_confidences) if errors else 0
    avg_correct_confidence = sum(correct_confidences) / len(correct_confidences) if correct else 0

    # Low confidence errors (classifier was unsure and wrong)
    low_conf_errors = [e for e in errors if e["confidence"] < 0.7]

    # High confidence errors (classifier was confident but wrong)
    high_conf_errors = [e for e in errors if e["confidence"] >= 0.9]

    return {
        "total_errors": len(errors),
        "total_correct": len(correct),
        "error_rate": len(errors) / len(predictions) if predictions else 0,
        "error_patterns": {k: len(v) for k, v in error_patterns.items()},
        "error_pattern_details": error_patterns,
        "avg_error_confidence": avg_error_confidence,
        "avg_correct_confidence": avg_correct_confidence,
        "low_confidence_errors": len(low_conf_errors),
        "high_confidence_errors": len(high_conf_errors),
    }


def find_ambiguous_queries(
    predictions_by_classifier: dict[str, list[dict]],
) -> list[dict]:
    """Find queries where classifiers disagree or all are wrong."""
    # Get all unique queries
    all_queries = set()
    for preds in predictions_by_classifier.values():
        for p in preds:
            all_queries.add(p["query"])

    ambiguous = []
    for query in all_queries:
        query_preds = {}
        true_type = None
        for clf_name, preds in predictions_by_classifier.items():
            for p in preds:
                if p["query"] == query:
                    query_preds[clf_name] = p["predicted_type"]
                    true_type = p["true_type"]
                    break

        # Check if classifiers disagree
        pred_types = set(query_preds.values())
        if len(pred_types) > 1:
            ambiguous.append({
                "query": query,
                "true_type": true_type,
                "predictions": query_preds,
                "reason": "classifiers_disagree",
            })
        # Check if all are wrong
        elif list(pred_types)[0] != true_type:
            ambiguous.append({
                "query": query,
                "true_type": true_type,
                "predictions": query_preds,
                "reason": "all_wrong",
            })

    return ambiguous


def generate_recommendations(
    analysis: dict,
    ambiguous_queries: list[dict],
) -> list[str]:
    """Generate recommendations based on analysis."""
    recommendations = []

    # Check for systematic error patterns
    error_patterns = analysis.get("error_patterns", {})
    for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1]):
        if count >= 3:
            recommendations.append(
                f"Systematic confusion ({count} cases): {pattern}. "
                f"Consider adding more training examples or clarifying the distinction."
            )

    # Check for high-confidence errors
    if analysis.get("high_confidence_errors", 0) > 0:
        recommendations.append(
            f"Found {analysis['high_confidence_errors']} high-confidence errors "
            f"(confidence >= 0.9). Review these cases - they indicate model overconfidence."
        )

    # Check if embedding vs LLM differ significantly
    if len(ambiguous_queries) > 0:
        disagree_count = sum(1 for q in ambiguous_queries if q["reason"] == "classifiers_disagree")
        if disagree_count > 0:
            recommendations.append(
                f"{disagree_count} queries have classifier disagreement. "
                f"These may be genuinely ambiguous or need clearer type definitions."
            )

    # General recommendations based on error rate
    error_rate = analysis.get("error_rate", 0)
    if error_rate > 0.1:
        recommendations.append(
            "Error rate is above 10%. Consider: "
            "(1) More diverse training examples, "
            "(2) Better type definitions in prompts, "
            "(3) Hybrid approach with LLM fallback."
        )

    return recommendations


def main() -> None:
    """Run error analysis."""
    logging.basicConfig(level=logging.INFO)

    # Set up paths
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "output" / "results"
    dataset_dir = experiment_dir / "output" / "dataset"

    # Check if results exist
    if not results_dir.exists():
        logger.error(f"Results directory not found at {results_dir}")
        logger.error("Run evaluate.py first")
        return

    # Load test dataset for reference
    test_dataset = load_dataset(dataset_dir / "queries_test.json")
    ground_truth = {q.query: q for q in test_dataset.queries}

    # Load all prediction files
    predictions_by_classifier: dict[str, list[dict]] = {}
    for pred_file in results_dir.glob("*_predictions.json"):
        classifier_name = pred_file.stem.replace("_predictions", "")
        predictions_by_classifier[classifier_name] = load_predictions(pred_file)

    if not predictions_by_classifier:
        logger.error("No prediction files found")
        return

    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    # Analyze each classifier
    all_analyses: dict[str, dict] = {}
    for clf_name, predictions in predictions_by_classifier.items():
        print(f"\n--- {clf_name} ---")
        analysis = analyze_errors(predictions)
        all_analyses[clf_name] = analysis

        print(f"Total errors: {analysis['total_errors']} / {analysis['total_errors'] + analysis['total_correct']}")
        print(f"Error rate: {analysis['error_rate']:.1%}")
        print(f"Avg confidence (correct): {analysis['avg_correct_confidence']:.3f}")
        print(f"Avg confidence (errors): {analysis['avg_error_confidence']:.3f}")

        if analysis["error_patterns"]:
            print("\nTop error patterns:")
            for pattern, count in sorted(
                analysis["error_patterns"].items(), key=lambda x: -x[1]
            )[:5]:
                print(f"  {pattern}: {count}")

    # Find ambiguous queries
    print("\n" + "=" * 70)
    print("AMBIGUOUS QUERIES")
    print("=" * 70)

    ambiguous = find_ambiguous_queries(predictions_by_classifier)
    if ambiguous:
        print(f"\nFound {len(ambiguous)} ambiguous queries:")

        # Group by reason
        by_reason: dict[str, list] = defaultdict(list)
        for q in ambiguous:
            by_reason[q["reason"]].append(q)

        for reason, queries in by_reason.items():
            print(f"\n{reason.replace('_', ' ').title()} ({len(queries)}):")
            for q in queries[:5]:  # Show first 5
                print(f"  Query: \"{q['query']}\"")
                print(f"  True: {q['true_type']}, Predicted: {q['predictions']}")

                # Get reasoning from dataset
                gt = ground_truth.get(q["query"])
                if gt:
                    print(f"  Reasoning: {gt.reasoning}")
                print()
    else:
        print("\nNo ambiguous queries found - all classifiers agree!")

    # Generate recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Use the best performing classifier's analysis for recommendations
    best_clf = min(all_analyses.items(), key=lambda x: x[1]["error_rate"])
    recommendations = generate_recommendations(best_clf[1], ambiguous)

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\nNo specific recommendations - classification looks good!")

    # Save detailed error analysis
    error_analysis = {
        "classifiers": all_analyses,
        "ambiguous_queries": ambiguous,
        "recommendations": recommendations,
    }

    with open(results_dir / "error_analysis.json", "w") as f:
        json.dump(error_analysis, f, indent=2, default=str)

    print(f"\n\nDetailed analysis saved to {results_dir / 'error_analysis.json'}")


if __name__ == "__main__":
    main()
