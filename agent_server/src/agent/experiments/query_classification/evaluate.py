"""Evaluate all classifiers on test dataset."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from agent.embedding_service import create_embedding_service
from agent.llm import SupportedModel, create_llm

from .classifiers.embedding_classifier import EmbeddingClassifier
from .classifiers.hybrid_classifier import HybridClassifier
from .classifiers.llm_few_shot import LLMFewShotClassifier
from .classifiers.llm_zero_shot import LLMZeroShotClassifier
from .create_dataset import load_dataset
from .models import ClassificationResult, Dataset, EvaluationMetrics, QueryType

logger = logging.getLogger(__name__)


def compute_metrics(
    results: list[ClassificationResult],
    dataset: Dataset,
) -> EvaluationMetrics:
    """Compute evaluation metrics from classification results."""
    # Build ground truth lookup
    ground_truth = {q.query: q.query_type for q in dataset.queries}

    # Compute confusion matrix and metrics
    confusion: dict[tuple[QueryType, QueryType], int] = defaultdict(int)
    correct = 0
    total = 0
    total_latency = 0.0

    # Per-class stats
    true_positives: dict[QueryType, int] = defaultdict(int)
    false_positives: dict[QueryType, int] = defaultdict(int)
    false_negatives: dict[QueryType, int] = defaultdict(int)

    for result in results:
        true_type = ground_truth.get(result.query)
        if true_type is None:
            logger.warning(f"Query not found in dataset: {result.query}")
            continue

        pred_type = result.predicted_type
        confusion[(true_type, pred_type)] += 1

        if true_type == pred_type:
            correct += 1
            true_positives[true_type] += 1
        else:
            false_positives[pred_type] += 1
            false_negatives[true_type] += 1

        total += 1
        total_latency += result.latency_ms

    # Compute per-class precision, recall, F1
    precision: dict[QueryType, float] = {}
    recall: dict[QueryType, float] = {}
    f1: dict[QueryType, float] = {}

    for qt in QueryType:
        tp = true_positives[qt]
        fp = false_positives[qt]
        fn = false_negatives[qt]

        precision[qt] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[qt] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision[qt] + recall[qt] > 0:
            f1[qt] = 2 * precision[qt] * recall[qt] / (precision[qt] + recall[qt])
        else:
            f1[qt] = 0.0

    return EvaluationMetrics(
        accuracy=correct / total if total > 0 else 0.0,
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        confusion_matrix=dict(confusion),
        total_samples=total,
        correct_predictions=correct,
        avg_latency_ms=total_latency / total if total > 0 else 0.0,
    )


def print_metrics(name: str, metrics: EvaluationMetrics) -> None:
    """Print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"Classifier: {name}")
    print(f"{'='*60}")

    print(f"\nOverall Accuracy: {metrics.accuracy:.2%}")
    print(f"Total Samples: {metrics.total_samples}")
    print(f"Correct: {metrics.correct_predictions}")
    print(f"Avg Latency: {metrics.avg_latency_ms:.1f}ms")

    print("\nPer-Class Metrics:")
    print(f"{'Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 50)

    for qt in QueryType:
        p = metrics.per_class_precision.get(qt, 0.0)
        r = metrics.per_class_recall.get(qt, 0.0)
        f = metrics.per_class_f1.get(qt, 0.0)
        print(f"{qt.value:<20} {p:>10.2%} {r:>10.2%} {f:>10.2%}")

    # Print confusion matrix
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    types = list(QueryType)
    header = "True\\Pred".ljust(18) + "".join(t.value[:8].rjust(10) for t in types)
    print(header)
    print("-" * len(header))

    for true_type in types:
        row = true_type.value[:16].ljust(18)
        for pred_type in types:
            count = metrics.confusion_matrix.get((true_type, pred_type), 0)
            row += str(count).rjust(10)
        print(row)


def save_results(
    results_dir: Path,
    classifier_name: str,
    metrics: EvaluationMetrics,
    predictions: list[ClassificationResult],
    dataset: Dataset,
) -> None:
    """Save evaluation results to files."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build ground truth lookup
    ground_truth = {q.query: q.query_type for q in dataset.queries}

    # Save metrics
    metrics_data = {
        "classifier": classifier_name,
        "accuracy": metrics.accuracy,
        "total_samples": metrics.total_samples,
        "correct_predictions": metrics.correct_predictions,
        "avg_latency_ms": metrics.avg_latency_ms,
        "per_class_precision": {k.value: v for k, v in metrics.per_class_precision.items()},
        "per_class_recall": {k.value: v for k, v in metrics.per_class_recall.items()},
        "per_class_f1": {k.value: v for k, v in metrics.per_class_f1.items()},
        "confusion_matrix": {
            f"{k[0].value}->{k[1].value}": v
            for k, v in metrics.confusion_matrix.items()
        },
    }

    with open(results_dir / f"{classifier_name}_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Save predictions
    predictions_data = []
    for pred in predictions:
        true_type = ground_truth.get(pred.query)
        predictions_data.append({
            "query": pred.query,
            "predicted_type": pred.predicted_type.value,
            "true_type": true_type.value if true_type else "unknown",
            "correct": pred.predicted_type == true_type if true_type else False,
            "confidence": pred.confidence,
            "reasoning": pred.reasoning,
            "latency_ms": pred.latency_ms,
        })

    with open(results_dir / f"{classifier_name}_predictions.json", "w") as f:
        json.dump(predictions_data, f, indent=2)


def evaluate_classifier(
    classifier_name: str,
    classify_fn: callable,
    dataset: Dataset,
    results_dir: Path,
) -> EvaluationMetrics:
    """Evaluate a classifier and save results."""
    logger.info(f"Evaluating {classifier_name}...")

    # Classify all queries
    queries = [q.query for q in dataset.queries]
    predictions = classify_fn(queries)

    # Compute metrics
    metrics = compute_metrics(predictions, dataset)

    # Print and save results
    print_metrics(classifier_name, metrics)
    save_results(results_dir, classifier_name, metrics, predictions, dataset)

    return metrics


def main() -> None:
    """Evaluate all classifiers."""
    logging.basicConfig(level=logging.INFO)

    # Set up paths
    experiment_dir = Path(__file__).parent
    dataset_dir = experiment_dir / "output" / "dataset"
    models_dir = experiment_dir / "output" / "models"
    results_dir = experiment_dir / "output" / "results"

    # Load test dataset
    test_path = dataset_dir / "queries_test.json"
    if not test_path.exists():
        logger.error(f"Test dataset not found at {test_path}")
        logger.error("Run create_dataset.py first")
        return

    logger.info(f"Loading test dataset from {test_path}")
    test_dataset = load_dataset(test_path)
    logger.info(f"Loaded {len(test_dataset.queries)} test examples")

    # Print test set distribution
    print("\nTest Set Distribution:")
    for qt, count in test_dataset.get_distribution().items():
        print(f"  {qt.value}: {count}")

    # Create services
    llm = create_llm()
    embedding_service = create_embedding_service()

    all_metrics: dict[str, EvaluationMetrics] = {}

    # 1. Evaluate LLM Zero-Shot
    print("\n" + "=" * 70)
    print("EVALUATING LLM ZERO-SHOT CLASSIFIER")
    print("=" * 70)
    zero_shot = LLMZeroShotClassifier(llm, SupportedModel.MISTRAL_SMALL_3_2_Q4)
    all_metrics["llm_zero_shot"] = evaluate_classifier(
        "llm_zero_shot",
        zero_shot.classify_batch,
        test_dataset,
        results_dir,
    )

    # 2. Evaluate LLM Few-Shot
    print("\n" + "=" * 70)
    print("EVALUATING LLM FEW-SHOT CLASSIFIER")
    print("=" * 70)
    few_shot = LLMFewShotClassifier(llm, SupportedModel.MISTRAL_SMALL_3_2_Q4)
    all_metrics["llm_few_shot"] = evaluate_classifier(
        "llm_few_shot",
        few_shot.classify_batch,
        test_dataset,
        results_dir,
    )

    # 3. Evaluate Embedding Classifiers (if trained)
    logistic_path = models_dir / "logistic_classifier.pkl"
    mlp_path = models_dir / "mlp_classifier.pkl"

    if logistic_path.exists():
        print("\n" + "=" * 70)
        print("EVALUATING LOGISTIC EMBEDDING CLASSIFIER")
        print("=" * 70)
        logistic = EmbeddingClassifier(embedding_service, "logistic")
        logistic.load(logistic_path)
        all_metrics["embedding_logistic"] = evaluate_classifier(
            "embedding_logistic",
            logistic.classify_batch,
            test_dataset,
            results_dir,
        )

    if mlp_path.exists():
        print("\n" + "=" * 70)
        print("EVALUATING MLP EMBEDDING CLASSIFIER")
        print("=" * 70)
        mlp = EmbeddingClassifier(embedding_service, "mlp")
        mlp.load(mlp_path)
        all_metrics["embedding_mlp"] = evaluate_classifier(
            "embedding_mlp",
            mlp.classify_batch,
            test_dataset,
            results_dir,
        )

        # 4. Evaluate Hybrid Classifier
        print("\n" + "=" * 70)
        print("EVALUATING HYBRID CLASSIFIER (threshold=0.8)")
        print("=" * 70)
        hybrid = HybridClassifier(mlp, llm, SupportedModel.MISTRAL_SMALL_3_2_Q4, 0.8)
        all_metrics["hybrid_0.8"] = evaluate_classifier(
            "hybrid_0.8",
            hybrid.classify_batch,
            test_dataset,
            results_dir,
        )

        # Print hybrid stats
        stats = hybrid.get_stats()
        print(f"\nHybrid Classifier Stats:")
        print(f"  Embedding-only calls: {stats['embedding_only_calls']}")
        print(f"  LLM fallback calls: {stats['llm_fallback_calls']}")
        print(f"  Embedding ratio: {stats['embedding_ratio']:.1%}")

        # Try different thresholds
        for threshold in [0.7, 0.9]:
            print(f"\n" + "=" * 70)
            print(f"EVALUATING HYBRID CLASSIFIER (threshold={threshold})")
            print("=" * 70)
            hybrid = HybridClassifier(mlp, llm, SupportedModel.MISTRAL_SMALL_3_2_Q4, threshold)
            all_metrics[f"hybrid_{threshold}"] = evaluate_classifier(
                f"hybrid_{threshold}",
                hybrid.classify_batch,
                test_dataset,
                results_dir,
            )
            stats = hybrid.get_stats()
            print(f"\nHybrid Stats: {stats['embedding_ratio']:.1%} embedding-only")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Classifier':<25} {'Accuracy':>10} {'Avg F1':>10} {'Latency':>12}")
    print("-" * 60)

    for name, metrics in sorted(all_metrics.items(), key=lambda x: -x[1].accuracy):
        avg_f1 = sum(metrics.per_class_f1.values()) / len(metrics.per_class_f1)
        print(
            f"{name:<25} {metrics.accuracy:>10.1%} {avg_f1:>10.2%} "
            f"{metrics.avg_latency_ms:>10.1f}ms"
        )

    # Save summary
    summary = {
        "classifiers": {
            name: {
                "accuracy": m.accuracy,
                "avg_f1": sum(m.per_class_f1.values()) / len(m.per_class_f1),
                "avg_latency_ms": m.avg_latency_ms,
            }
            for name, m in all_metrics.items()
        }
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
