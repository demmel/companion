"""
Evaluate improved hybrid classifier approaches based on confidence analysis.

Tests:
1. Lowered threshold (0.35) with existing classifier
2. Calibrated classifier with standard threshold (0.7)
3. Margin-based confidence with calibrated classifier
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from agent.embedding_service import EmbeddingService
from agent.experiments.query_classification.classifiers.llm_few_shot import LLMFewShotClassifier
from agent.llm import SupportedModel, create_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "output" / "results"
DATASET_DIR = EXPERIMENT_DIR / "output" / "dataset"


@dataclass
class HybridResult:
    """Results from hybrid classification."""

    accuracy: float
    embedding_ratio: float
    embedding_accuracy: float
    llm_accuracy: float
    avg_latency_ms: float
    total_queries: int


def load_dataset(split: str) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """Load queries and labels from dataset."""
    path = DATASET_DIR / f"queries_{split}.json"
    with open(path) as f:
        data = json.load(f)
    items = data.get("queries", data) if isinstance(data, dict) else data
    queries = [item["query"] for item in items]
    labels = [item["query_type"] for item in items]
    return items, queries, labels


class ImprovedHybridClassifier:
    """Hybrid classifier with configurable confidence strategy."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        threshold: float = 0.35,
        use_calibration: bool = False,
        calibration_method: str = "isotonic",
        use_margin: bool = False,
    ):
        self.embedding_service = embedding_service
        self.threshold = threshold
        self.use_calibration = use_calibration
        self.calibration_method = calibration_method
        self.use_margin = use_margin
        self.classifier: LogisticRegression | CalibratedClassifierCV | None = None
        self.label_names: list[str] = []
        self.llm = create_llm()
        self.llm_classifier = LLMFewShotClassifier(self.llm, SupportedModel.MISTRAL_SMALL_3_2_Q4)

    def train(self, queries: list[str], labels: list[str]) -> None:
        """Train the embedding classifier."""
        embeddings = self.embedding_service.encode_batch(queries)
        X = np.array(embeddings)

        self.label_names = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}
        y = np.array([label_to_idx[label] for label in labels])

        base_clf = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )

        if self.use_calibration:
            self.classifier = CalibratedClassifierCV(
                estimator=base_clf,
                method=self.calibration_method,
                cv=5,
            )
            self.classifier.fit(X, y)
        else:
            base_clf.fit(X, y)
            self.classifier = base_clf

    def _get_confidence(self, probs: np.ndarray) -> float:
        """Get confidence score based on strategy."""
        if self.use_margin:
            sorted_probs = np.sort(probs)[::-1]
            return float(sorted_probs[0] - sorted_probs[1])
        return float(np.max(probs))

    def classify(self, query: str) -> tuple[str, float, bool, float]:
        """
        Classify a query.

        Returns: (predicted_type, confidence, used_embedding, latency_ms)
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not trained")

        start = time.perf_counter()

        # Get embedding prediction
        embedding = np.array([self.embedding_service.encode(query)])
        probs = self.classifier.predict_proba(embedding)[0]
        confidence = self._get_confidence(probs)

        if confidence >= self.threshold:
            pred_idx = int(np.argmax(probs))
            pred_type = self.label_names[pred_idx]
            latency_ms = (time.perf_counter() - start) * 1000
            return pred_type, confidence, True, latency_ms

        # Fall back to LLM
        llm_result = self.llm_classifier.classify(query)
        latency_ms = (time.perf_counter() - start) * 1000
        return llm_result.predicted_type.value, llm_result.confidence, False, latency_ms


def evaluate_hybrid(
    name: str,
    classifier: ImprovedHybridClassifier,
    test_queries: list[str],
    test_labels: list[str],
) -> HybridResult:
    """Evaluate a hybrid classifier configuration."""
    correct = 0
    embedding_correct = 0
    embedding_count = 0
    llm_correct = 0
    llm_count = 0
    total_latency = 0.0

    for query, true_label in zip(test_queries, test_labels):
        pred_type, confidence, used_embedding, latency_ms = classifier.classify(query)
        total_latency += latency_ms

        is_correct = pred_type == true_label
        if is_correct:
            correct += 1

        if used_embedding:
            embedding_count += 1
            if is_correct:
                embedding_correct += 1
        else:
            llm_count += 1
            if is_correct:
                llm_correct += 1

    return HybridResult(
        accuracy=correct / len(test_queries),
        embedding_ratio=embedding_count / len(test_queries),
        embedding_accuracy=embedding_correct / embedding_count if embedding_count > 0 else 0.0,
        llm_accuracy=llm_correct / llm_count if llm_count > 0 else 0.0,
        avg_latency_ms=total_latency / len(test_queries),
        total_queries=len(test_queries),
    )


def main() -> None:
    """Run improved hybrid evaluation."""
    print("=" * 70)
    print("IMPROVED HYBRID CLASSIFIER EVALUATION")
    print("=" * 70)

    # Load data
    _, train_queries, train_labels = load_dataset("train")
    _, test_queries, test_labels = load_dataset("test")

    embedding_service = EmbeddingService()

    # Define configurations to test
    configs = [
        {
            "name": "baseline_0.7",
            "threshold": 0.7,
            "use_calibration": False,
            "use_margin": False,
        },
        {
            "name": "lowered_0.35",
            "threshold": 0.35,
            "use_calibration": False,
            "use_margin": False,
        },
        {
            "name": "lowered_0.40",
            "threshold": 0.40,
            "use_calibration": False,
            "use_margin": False,
        },
        {
            "name": "calibrated_isotonic_0.7",
            "threshold": 0.7,
            "use_calibration": True,
            "calibration_method": "isotonic",
            "use_margin": False,
        },
        {
            "name": "calibrated_isotonic_0.8",
            "threshold": 0.8,
            "use_calibration": True,
            "calibration_method": "isotonic",
            "use_margin": False,
        },
        {
            "name": "calibrated_sigmoid_0.5",
            "threshold": 0.5,
            "use_calibration": True,
            "calibration_method": "sigmoid",
            "use_margin": False,
        },
        {
            "name": "margin_isotonic_0.3",
            "threshold": 0.3,
            "use_calibration": True,
            "calibration_method": "isotonic",
            "use_margin": True,
        },
        {
            "name": "margin_isotonic_0.4",
            "threshold": 0.4,
            "use_calibration": True,
            "calibration_method": "isotonic",
            "use_margin": True,
        },
    ]

    results = {}

    for config in configs:
        name = config.pop("name")
        print(f"\n--- Evaluating: {name} ---")

        classifier = ImprovedHybridClassifier(embedding_service, **config)
        classifier.train(train_queries, train_labels)

        result = evaluate_hybrid(name, classifier, test_queries, test_labels)
        results[name] = result

        print(f"Overall Accuracy: {result.accuracy:.1%}")
        print(f"Embedding Ratio: {result.embedding_ratio:.1%}")
        print(f"Embedding Accuracy: {result.embedding_accuracy:.1%}")
        print(f"LLM Accuracy: {result.llm_accuracy:.1%}")
        print(f"Avg Latency: {result.avg_latency_ms:.1f}ms")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Config':<30} {'Accuracy':>10} {'Emb-Ratio':>10} {'Emb-Acc':>10} {'Latency':>10}")
    print("-" * 70)

    for name, result in sorted(results.items(), key=lambda x: -x[1].accuracy):
        print(
            f"{name:<30} "
            f"{result.accuracy:>10.1%} "
            f"{result.embedding_ratio:>10.1%} "
            f"{result.embedding_accuracy:>10.1%} "
            f"{result.avg_latency_ms:>9.0f}ms"
        )

    # Find best config
    best_name = max(results, key=lambda x: results[x].accuracy)
    best = results[best_name]

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"\nConfiguration: {best_name}")
    print(f"  Overall Accuracy: {best.accuracy:.1%}")
    print(f"  Embedding Ratio: {best.embedding_ratio:.1%}")
    print(f"  Embedding Accuracy: {best.embedding_accuracy:.1%}")
    print(f"  LLM Fallback Accuracy: {best.llm_accuracy:.1%}")
    print(f"  Average Latency: {best.avg_latency_ms:.1f}ms")

    # Calculate speed improvement
    llm_only_latency = 1866.8  # from previous experiment
    speed_improvement = (llm_only_latency - best.avg_latency_ms) / llm_only_latency * 100
    print(f"\nSpeed improvement vs LLM-only: {speed_improvement:.1f}%")

    # Save results
    output = {
        name: {
            "accuracy": r.accuracy,
            "embedding_ratio": r.embedding_ratio,
            "embedding_accuracy": r.embedding_accuracy,
            "llm_accuracy": r.llm_accuracy,
            "avg_latency_ms": r.avg_latency_ms,
        }
        for name, r in results.items()
    }
    output["best_config"] = best_name

    output_path = RESULTS_DIR / "improved_hybrid_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
