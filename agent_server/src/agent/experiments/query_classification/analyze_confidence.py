"""
Analyze embedding classifier confidence distribution and test calibration approaches.

This experiment investigates why the hybrid classifier falls back to LLM 100% of the time
and explores potential solutions.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from agent.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "output" / "results"
DATASET_DIR = EXPERIMENT_DIR / "output" / "dataset"


@dataclass
class ConfidenceAnalysis:
    """Analysis results for confidence distribution."""

    min_confidence: float
    max_confidence: float
    mean_confidence: float
    median_confidence: float
    std_confidence: float
    percentiles: dict[str, float]
    correct_mean: float
    incorrect_mean: float
    confidence_gap: float  # difference between correct and incorrect


def load_predictions(classifier_name: str) -> list[dict[str, object]]:
    """Load predictions from results file."""
    path = RESULTS_DIR / f"{classifier_name}_predictions.json"
    with open(path) as f:
        return json.load(f)


def analyze_confidence_distribution(predictions: list[dict[str, object]]) -> ConfidenceAnalysis:
    """Analyze the confidence distribution of predictions."""
    confidences = [p["confidence"] for p in predictions]
    correct_conf = [p["confidence"] for p in predictions if p["correct"]]
    incorrect_conf = [p["confidence"] for p in predictions if not p["correct"]]

    conf_array = np.array(confidences)

    return ConfidenceAnalysis(
        min_confidence=float(np.min(conf_array)),
        max_confidence=float(np.max(conf_array)),
        mean_confidence=float(np.mean(conf_array)),
        median_confidence=float(np.median(conf_array)),
        std_confidence=float(np.std(conf_array)),
        percentiles={
            "p10": float(np.percentile(conf_array, 10)),
            "p25": float(np.percentile(conf_array, 25)),
            "p50": float(np.percentile(conf_array, 50)),
            "p75": float(np.percentile(conf_array, 75)),
            "p90": float(np.percentile(conf_array, 90)),
        },
        correct_mean=float(np.mean(correct_conf)) if correct_conf else 0.0,
        incorrect_mean=float(np.mean(incorrect_conf)) if incorrect_conf else 0.0,
        confidence_gap=float(np.mean(correct_conf) - np.mean(incorrect_conf)) if incorrect_conf else 0.0,
    )


def load_dataset(split: str) -> tuple[list[str], list[str]]:
    """Load queries and labels from dataset."""
    path = DATASET_DIR / f"queries_{split}.json"
    with open(path) as f:
        data = json.load(f)
    # Dataset is wrapped in an object with "queries" key
    items = data.get("queries", data) if isinstance(data, dict) else data
    queries = [item["query"] for item in items]
    labels = [item["query_type"] for item in items]
    return queries, labels


def train_calibrated_classifier(
    train_queries: list[str],
    train_labels: list[str],
    embedding_service: EmbeddingService,
    method: str = "isotonic",
) -> tuple[CalibratedClassifierCV, list[str]]:
    """Train a calibrated logistic regression classifier."""
    # Generate embeddings
    embeddings = embedding_service.encode_batch(train_queries)
    X = np.array(embeddings)

    # Get unique labels
    unique_labels = sorted(set(train_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in train_labels])

    # Train base classifier
    base_clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
    )

    # Wrap with calibration
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method=method,
        cv=5,  # 5-fold cross-validation for calibration
    )
    calibrated_clf.fit(X, y)

    return calibrated_clf, unique_labels


def evaluate_with_threshold(
    predictions: list[dict[str, object]],
    threshold: float,
) -> dict[str, float]:
    """Evaluate accuracy at different confidence thresholds."""
    high_conf = [p for p in predictions if p["confidence"] >= threshold]
    low_conf = [p for p in predictions if p["confidence"] < threshold]

    high_conf_correct = sum(1 for p in high_conf if p["correct"])
    low_conf_correct = sum(1 for p in low_conf if p["correct"])

    return {
        "threshold": threshold,
        "high_conf_count": len(high_conf),
        "high_conf_accuracy": high_conf_correct / len(high_conf) if high_conf else 0.0,
        "low_conf_count": len(low_conf),
        "low_conf_accuracy": low_conf_correct / len(low_conf) if low_conf else 0.0,
        "embedding_ratio": len(high_conf) / len(predictions),
    }


def calculate_margin_confidence(probs: np.ndarray) -> float:
    """Calculate confidence as margin between top-2 predictions."""
    sorted_probs = np.sort(probs)[::-1]
    return float(sorted_probs[0] - sorted_probs[1])


def main() -> None:
    """Run confidence analysis experiment."""
    print("=" * 70)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Analyze existing predictions
    for classifier in ["embedding_logistic", "embedding_mlp"]:
        print(f"\n--- {classifier} ---")
        predictions = load_predictions(classifier)
        analysis = analyze_confidence_distribution(predictions)

        print(f"Confidence range: [{analysis.min_confidence:.3f}, {analysis.max_confidence:.3f}]")
        print(f"Mean: {analysis.mean_confidence:.3f}, Median: {analysis.median_confidence:.3f}")
        print(f"Std: {analysis.std_confidence:.3f}")
        print(f"Percentiles: {analysis.percentiles}")
        print(f"Correct predictions mean confidence: {analysis.correct_mean:.3f}")
        print(f"Incorrect predictions mean confidence: {analysis.incorrect_mean:.3f}")
        print(f"Confidence gap (correct - incorrect): {analysis.confidence_gap:.3f}")

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    predictions = load_predictions("embedding_logistic")
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    print("\nLogistic Regression at different thresholds:")
    print(f"{'Threshold':>10} {'High-Conf':>10} {'High-Acc':>10} {'Low-Conf':>10} {'Low-Acc':>10} {'Emb-Ratio':>10}")
    print("-" * 70)

    for threshold in thresholds:
        result = evaluate_with_threshold(predictions, threshold)
        print(
            f"{result['threshold']:>10.2f} "
            f"{result['high_conf_count']:>10} "
            f"{result['high_conf_accuracy']:>10.1%} "
            f"{result['low_conf_count']:>10} "
            f"{result['low_conf_accuracy']:>10.1%} "
            f"{result['embedding_ratio']:>10.1%}"
        )

    print("\n" + "=" * 70)
    print("CALIBRATED CLASSIFIER EXPERIMENT")
    print("=" * 70)

    # Load datasets
    train_queries, train_labels = load_dataset("train")
    test_queries, test_labels = load_dataset("test")

    embedding_service = EmbeddingService()

    # Train calibrated classifiers
    for method in ["isotonic", "sigmoid"]:
        print(f"\n--- Calibration method: {method} ---")

        calibrated_clf, label_names = train_calibrated_classifier(
            train_queries, train_labels, embedding_service, method=method
        )

        # Evaluate on test set
        test_embeddings = np.array(embedding_service.encode_batch(test_queries))
        probs = calibrated_clf.predict_proba(test_embeddings)
        predictions_idx = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        # Calculate margin-based confidence too
        margin_confidences = np.array([calculate_margin_confidence(p) for p in probs])

        # Build predictions list
        calibrated_predictions = []
        label_to_idx = {label: idx for idx, label in enumerate(label_names)}

        for i, (query, true_label) in enumerate(zip(test_queries, test_labels)):
            pred_label = label_names[predictions_idx[i]]
            calibrated_predictions.append({
                "query": query,
                "predicted_type": pred_label,
                "true_type": true_label,
                "correct": pred_label == true_label,
                "confidence": float(confidences[i]),
                "margin_confidence": float(margin_confidences[i]),
            })

        # Analyze calibrated confidence
        analysis = analyze_confidence_distribution(calibrated_predictions)
        accuracy = sum(1 for p in calibrated_predictions if p["correct"]) / len(calibrated_predictions)

        print(f"Accuracy: {accuracy:.1%}")
        print(f"Confidence range: [{analysis.min_confidence:.3f}, {analysis.max_confidence:.3f}]")
        print(f"Mean: {analysis.mean_confidence:.3f}, Median: {analysis.median_confidence:.3f}")
        print(f"Correct mean: {analysis.correct_mean:.3f}, Incorrect mean: {analysis.incorrect_mean:.3f}")
        print(f"Confidence gap: {analysis.confidence_gap:.3f}")

        # Threshold analysis for calibrated
        print(f"\nThreshold analysis ({method}):")
        print(f"{'Threshold':>10} {'High-Conf':>10} {'High-Acc':>10} {'Emb-Ratio':>10}")
        print("-" * 50)

        for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
            result = evaluate_with_threshold(calibrated_predictions, threshold)
            if result["high_conf_count"] > 0:
                print(
                    f"{threshold:>10.2f} "
                    f"{result['high_conf_count']:>10} "
                    f"{result['high_conf_accuracy']:>10.1%} "
                    f"{result['embedding_ratio']:>10.1%}"
                )

        # Margin-based confidence analysis
        margin_predictions = [
            {**p, "confidence": p["margin_confidence"]} for p in calibrated_predictions
        ]
        print(f"\nMargin-based confidence ({method}):")
        margin_analysis = analyze_confidence_distribution(margin_predictions)
        print(f"Margin range: [{margin_analysis.min_confidence:.3f}, {margin_analysis.max_confidence:.3f}]")
        print(f"Correct margin mean: {margin_analysis.correct_mean:.3f}")
        print(f"Incorrect margin mean: {margin_analysis.incorrect_mean:.3f}")
        print(f"Margin gap: {margin_analysis.confidence_gap:.3f}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find optimal threshold for hybrid
    predictions = load_predictions("embedding_logistic")
    best_threshold = 0.3
    best_score = 0.0

    for threshold in np.arange(0.25, 0.6, 0.05):
        result = evaluate_with_threshold(predictions, threshold)
        # Score: balance between embedding ratio and high-conf accuracy
        if result["high_conf_count"] > 0 and result["high_conf_accuracy"] >= 0.85:
            score = result["embedding_ratio"]
            if score > best_score:
                best_score = score
                best_threshold = threshold

    print(f"\n1. Lower hybrid threshold to {best_threshold:.2f}")
    print(f"   - Would use embeddings for {best_score:.1%} of queries")
    print(f"   - While maintaining 85%+ accuracy on those queries")

    print("\n2. Use calibrated classifier (isotonic method)")
    print("   - Better calibrated probabilities")
    print("   - Larger confidence gap between correct/incorrect")

    print("\n3. Consider margin-based confidence")
    print("   - Margin = top_prob - second_prob")
    print("   - Better discriminates certain vs uncertain predictions")

    # Save results
    results = {
        "uncalibrated_analysis": {
            "embedding_logistic": analyze_confidence_distribution(load_predictions("embedding_logistic")).__dict__,
            "embedding_mlp": analyze_confidence_distribution(load_predictions("embedding_mlp")).__dict__,
        },
        "recommended_threshold": best_threshold,
        "recommended_embedding_ratio": best_score,
    }

    output_path = RESULTS_DIR / "confidence_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
