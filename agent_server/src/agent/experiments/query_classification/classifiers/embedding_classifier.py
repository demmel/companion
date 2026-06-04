"""Embedding-based query classifier using sklearn."""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from agent.embedding_service import EmbeddingService

from ..models import (
    ClassificationResult,
    Dataset,
    LabeledQuery,
    QueryType,
)

logger = logging.getLogger(__name__)


class EmbeddingClassifier:
    """Classifier using sentence embeddings and sklearn models."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        classifier_type: str = "logistic",  # "logistic" or "mlp"
    ):
        self.embedding_service = embedding_service
        self.classifier_type = classifier_type
        self.name = f"embedding_{classifier_type}"

        self.label_encoder = LabelEncoder()
        self.classifier: LogisticRegression | MLPClassifier | None = None
        self._is_trained = False

    def train(self, dataset: Dataset) -> None:
        """Train the classifier on a dataset."""
        logger.info(f"Training {self.name} classifier on {len(dataset.queries)} examples")

        # Get embeddings for all queries
        queries = [q.query for q in dataset.queries]
        labels = [q.query_type.value for q in dataset.queries]

        logger.info("Generating embeddings...")
        embeddings = self.embedding_service.encode_batch(queries)
        X = np.array(embeddings)

        # Encode labels
        self.label_encoder.fit([qt.value for qt in QueryType])
        y = self.label_encoder.transform(labels)

        # Train classifier
        if self.classifier_type == "logistic":
            self.classifier = LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                class_weight="balanced",
                random_state=42,
            )
        elif self.classifier_type == "mlp":
            self.classifier = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

        logger.info(f"Fitting {self.classifier_type} classifier...")
        self.classifier.fit(X, y)
        self._is_trained = True

        # Log training accuracy
        train_accuracy = self.classifier.score(X, y)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")

    def classify(self, query: str) -> ClassificationResult:
        """Classify a single query."""
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before classification")

        assert self.classifier is not None

        start_time = time.perf_counter()

        # Get embedding
        embedding = self.embedding_service.encode(query)
        X = np.array([embedding])

        # Predict
        y_pred = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(proba))

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Decode label
        query_type_str = str(self.label_encoder.inverse_transform([y_pred])[0])
        query_type = QueryType(query_type_str)

        return ClassificationResult(
            query=query,
            predicted_type=query_type,
            confidence=confidence,
            reasoning=f"Embedding classifier ({self.classifier_type}) prediction",
            latency_ms=latency_ms,
        )

    def classify_batch(self, queries: list[str]) -> list[ClassificationResult]:
        """Classify multiple queries efficiently."""
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before classification")

        assert self.classifier is not None

        start_time = time.perf_counter()

        # Get embeddings in batch
        embeddings = self.embedding_service.encode_batch(queries)
        X = np.array(embeddings)

        # Predict
        y_preds = self.classifier.predict(X)
        probas = self.classifier.predict_proba(X)

        batch_latency_ms = (time.perf_counter() - start_time) * 1000
        per_query_latency = batch_latency_ms / len(queries)

        results = []
        for i, query in enumerate(queries):
            query_type_str = str(self.label_encoder.inverse_transform([y_preds[i]])[0])
            query_type = QueryType(query_type_str)
            confidence = float(np.max(probas[i]))

            results.append(
                ClassificationResult(
                    query=query,
                    predicted_type=query_type,
                    confidence=confidence,
                    reasoning=f"Embedding classifier ({self.classifier_type}) prediction",
                    latency_ms=per_query_latency,
                )
            )

        return results

    def get_confidence_for_query(self, query: str) -> tuple[QueryType, float]:
        """Get predicted type and confidence for a query."""
        result = self.classify(query)
        return result.predicted_type, result.confidence

    def save(self, path: Path) -> None:
        """Save the trained classifier to disk."""
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before saving")

        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
            "classifier_type": self.classifier_type,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved classifier to {path}")

    def load(self, path: Path) -> None:
        """Load a trained classifier from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.classifier = data["classifier"]
        self.label_encoder = data["label_encoder"]
        self.classifier_type = data["classifier_type"]
        self._is_trained = True
        self.name = f"embedding_{self.classifier_type}"

        logger.info(f"Loaded classifier from {path}")

    @property
    def is_trained(self) -> bool:
        """Check if classifier is trained."""
        return self._is_trained
