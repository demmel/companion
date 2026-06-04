"""Train embedding classifiers on the dataset."""

import logging
from pathlib import Path

from agent.embedding_service import create_embedding_service

from .classifiers.embedding_classifier import EmbeddingClassifier
from .create_dataset import load_dataset

logger = logging.getLogger(__name__)


def main() -> None:
    """Train and save embedding classifiers."""
    logging.basicConfig(level=logging.INFO)

    # Set up paths
    experiment_dir = Path(__file__).parent
    dataset_dir = experiment_dir / "output" / "dataset"
    models_dir = experiment_dir / "output" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load training dataset
    train_path = dataset_dir / "queries_train.json"
    if not train_path.exists():
        logger.error(f"Training dataset not found at {train_path}")
        logger.error("Run create_dataset.py first to generate the dataset")
        return

    logger.info(f"Loading training dataset from {train_path}")
    train_dataset = load_dataset(train_path)
    logger.info(f"Loaded {len(train_dataset.queries)} training examples")

    # Create embedding service
    embedding_service = create_embedding_service()

    # Train logistic regression classifier
    logger.info("\n=== Training Logistic Regression Classifier ===")
    logistic_classifier = EmbeddingClassifier(
        embedding_service=embedding_service,
        classifier_type="logistic",
    )
    logistic_classifier.train(train_dataset)
    logistic_classifier.save(models_dir / "logistic_classifier.pkl")

    # Train MLP classifier
    logger.info("\n=== Training MLP Classifier ===")
    mlp_classifier = EmbeddingClassifier(
        embedding_service=embedding_service,
        classifier_type="mlp",
    )
    mlp_classifier.train(train_dataset)
    mlp_classifier.save(models_dir / "mlp_classifier.pkl")

    logger.info("\n=== Training Complete ===")
    logger.info(f"Models saved to {models_dir}")


if __name__ == "__main__":
    main()
