"""
Embedding service for memory similarity calculations.
"""

import logging
import time
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing text embeddings for memory similarity"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy load the sentence transformer model"""
        if self._model_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            start_time = time.time()

            self._model = SentenceTransformer(self.model_name)

            load_time = time.time() - start_time
            logger.info(f"Embedding model loaded in {load_time:.2f}s")
            self._model_loaded = True

        except ImportError:
            logger.error(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    def encode(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to encode

        Returns:
            List of floats representing the embedding vector
        """
        self._load_model()
        assert self._model is not None, "Model must be loaded before encoding"

        start_time = time.time()
        embedding = self._model.encode(text, convert_to_numpy=True)
        encode_time = time.time() - start_time

        logger.debug(
            f"Generated embedding for text ({len(text)} chars) in {encode_time*1000:.1f}ms"
        )

        # Convert to list for JSON serialization
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        self._load_model()
        assert self._model is not None, "Model must be loaded before encoding batch"

        start_time = time.time()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        encode_time = time.time() - start_time

        logger.info(
            f"Generated {len(texts)} embeddings in {encode_time:.3f}s ({encode_time/len(texts)*1000:.1f}ms per text)"
        )

        # Convert to list of lists for JSON serialization
        return [emb.tolist() for emb in embeddings]

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        # Convert to numpy arrays for efficient computation
        a = np.array(vec1)
        b = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self._model_loaded:
            return {"model_name": self.model_name, "loaded": False}

        return {
            "model_name": self.model_name,
            "loaded": True,
            "max_seq_length": getattr(self._model, "max_seq_length", "unknown"),
            "embedding_dimension": getattr(
                self._model, "get_sentence_embedding_dimension", lambda: "unknown"
            )(),
        }


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def create_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """Create a new embedding service with specified model"""
    return EmbeddingService(model_name=model_name)
