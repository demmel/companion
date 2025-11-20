"""
Retrieval strategies for finding relevant facts in knowledge graphs.

Different approaches to answer queries using structured knowledge.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from .interfaces import IRetriever, IKnowledgeGraph, Fact

logger = logging.getLogger(__name__)


class EmbeddingRetriever(IRetriever):
    """
    Retrieve facts using embedding similarity.

    Embeds facts and queries, returns top-k by cosine similarity.
    Baseline approach to compare against.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with sentence transformer model.

        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self._fact_cache: Dict[str, Tuple[Fact, np.ndarray]] = (
            {}
        )  # fact_id -> (fact, embedding)

    def retrieve(
        self, query: str, graph: IKnowledgeGraph, top_k: int = 10
    ) -> List[Fact]:
        """Retrieve facts by embedding similarity"""
        # Embed query
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Embed all facts (with caching)
        fact_embeddings = []
        facts = []

        for fact in graph.get_all_facts():
            if fact.id in self._fact_cache:
                cached_fact, cached_emb = self._fact_cache[fact.id]
                # Check if fact hasn't changed (by comparing id and predicate)
                if cached_fact.predicate == fact.predicate:
                    fact_embeddings.append(cached_emb)
                    facts.append(fact)
                    continue

            # Embed fact
            fact_text = self._fact_to_text(fact)
            embedding = self.model.encode(fact_text, convert_to_numpy=True)
            self._fact_cache[fact.id] = (fact, embedding)

            fact_embeddings.append(embedding)
            facts.append(fact)

        if not facts:
            return []

        # Compute similarities
        fact_embeddings = np.array(fact_embeddings)
        similarities = np.dot(fact_embeddings, query_embedding)

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [facts[i] for i in top_indices]

    def _fact_to_text(self, fact: Fact) -> str:
        """Convert fact to text for embedding"""
        entities_str = " ".join(
            [f"{role}:{entity}" for role, entity in fact.entities.items()]
        )
        # Include domain properties in text
        parts = [fact.predicate, entities_str]
        if fact.time_period:
            parts.append(fact.time_period)
        if fact.region:
            parts.append(fact.region)
        if fact.confidence:
            parts.append(fact.confidence)
        return " ".join(parts)


class KeywordRetriever(IRetriever):
    """
    Retrieve facts by keyword matching.

    Simple but fast: checks if query keywords appear in fact text.
    """

    def retrieve(
        self, query: str, graph: IKnowledgeGraph, top_k: int = 10
    ) -> List[Fact]:
        """Retrieve facts by keyword matching"""
        # Extract keywords from query (simple tokenization)
        keywords = set(query.lower().split())

        # Score each fact by keyword overlap
        scored_facts = []
        for fact in graph.get_all_facts():
            fact_text = self._fact_to_text(fact).lower()
            score = sum(1 for keyword in keywords if keyword in fact_text)

            if score > 0:
                scored_facts.append((score, fact))

        # Sort by score and return top-k
        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored_facts[:top_k]]

    def _fact_to_text(self, fact: Fact) -> str:
        """Convert fact to text for keyword matching"""
        parts = [fact.predicate]
        parts.extend(fact.entities.values())
        # Include domain properties
        if fact.time_period:
            parts.append(fact.time_period)
        if fact.region:
            parts.append(fact.region)
        if fact.confidence:
            parts.append(fact.confidence)
        return " ".join(parts)


class HybridRetriever(IRetriever):
    """
    Combines embedding similarity and keyword matching.

    Uses weighted combination of both scores.
    """

    def __init__(self, embedding_weight: float = 0.7, keyword_weight: float = 0.3):
        """
        Initialize hybrid retriever.

        Args:
            embedding_weight: Weight for embedding similarity
            keyword_weight: Weight for keyword match
        """
        self.embedding_retriever = EmbeddingRetriever()
        self.keyword_retriever = KeywordRetriever()
        self.embedding_weight = embedding_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self, query: str, graph: IKnowledgeGraph, top_k: int = 10
    ) -> List[Fact]:
        """Retrieve using hybrid scoring"""
        # Get candidates from both methods
        embedding_facts = self.embedding_retriever.retrieve(
            query, graph, top_k=top_k * 2
        )
        keyword_facts = self.keyword_retriever.retrieve(query, graph, top_k=top_k * 2)

        # Build unified scoring
        fact_scores: Dict[str, float] = {}

        # Add embedding scores (normalized)
        for i, fact in enumerate(embedding_facts):
            score = (len(embedding_facts) - i) / len(
                embedding_facts
            )  # Rank-based score
            fact_scores[fact.id] = self.embedding_weight * score

        # Add keyword scores (normalized)
        for i, fact in enumerate(keyword_facts):
            score = (len(keyword_facts) - i) / len(keyword_facts)
            fact_scores[fact.id] = (
                fact_scores.get(fact.id, 0) + self.keyword_weight * score
            )

        # Sort by combined score
        all_facts_by_id = {f.id: f for f in graph.get_all_facts()}
        scored_facts = [
            (score, all_facts_by_id[fid]) for fid, score in fact_scores.items()
        ]
        scored_facts.sort(key=lambda x: x[0], reverse=True)

        return [fact for _, fact in scored_facts[:top_k]]


class GraphTraversalRetriever(IRetriever):
    """
    Retrieves facts by graph traversal from query entities.

    Starts from entities mentioned in query, expands to connected facts.
    """

    def retrieve(
        self, query: str, graph: IKnowledgeGraph, top_k: int = 10
    ) -> List[Fact]:
        """Retrieve by traversing from query entities"""
        # Extract potential entity names from query
        query_tokens = set(query.split())

        # Find entities that match query tokens
        matching_entities = []
        for entity_id in graph.get_all_entities():
            # Simple matching: check if entity appears in query
            if any(token.lower() in entity_id.lower() for token in query_tokens):
                matching_entities.append(entity_id)

        if not matching_entities:
            # Fall back to keyword retrieval
            logger.debug("No matching entities, falling back to keyword retrieval")
            return KeywordRetriever().retrieve(query, graph, top_k)

        # Get facts involving these entities
        relevant_facts = []
        seen_ids = set()

        for entity in matching_entities:
            for fact in graph.find_facts_by_entity(entity):
                if fact.id not in seen_ids:
                    relevant_facts.append(fact)
                    seen_ids.add(fact.id)

        # If we have more than top_k, rank by number of matching entities
        if len(relevant_facts) > top_k:
            scored_facts = []
            for fact in relevant_facts:
                # Count how many query entities are in this fact
                score = sum(
                    1 for entity in matching_entities if fact.involves_entity(entity)
                )
                scored_facts.append((score, fact))

            scored_facts.sort(key=lambda x: x[0], reverse=True)
            return [fact for _, fact in scored_facts[:top_k]]

        return relevant_facts[:top_k]


# Registry of retrieval strategies
RETRIEVAL_STRATEGIES = {
    "embedding": EmbeddingRetriever,
    "keyword": KeywordRetriever,
    "hybrid": HybridRetriever,
    "graph_traversal": GraphTraversalRetriever,
}


def get_retriever(strategy_name: str) -> IRetriever:
    """Get a retriever by strategy name"""
    retriever_class = RETRIEVAL_STRATEGIES.get(strategy_name)
    if not retriever_class:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(RETRIEVAL_STRATEGIES.keys())}"
        )
    return retriever_class()
