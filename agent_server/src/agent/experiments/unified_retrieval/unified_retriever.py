"""Unified Retriever - Main retrieval pipeline.

This module implements the UnifiedRetriever class that orchestrates all
retrieval components based on query classification.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from agent.embedding_service import EmbeddingService
from agent.experiments.retrieval.knowledge_graph import (
    KnowledgeGraph,
    KGFact,
    retrieve_kg_aware,
)

from .models import (
    DetectedReference,
    EpisodeSummary,
    Fact,
    Memory,
    QueryType,
    RetrievalContext,
    TopicMatch,
    UnifiedRetrieverConfig,
)
from .query_classifier import LLMQueryClassifier, RuleBasedQueryClassifier

logger = logging.getLogger(__name__)


# =============================================================================
# Component Adapters
# =============================================================================


@dataclass
class SimpleMemoryIndex:
    """Simple in-memory index for similarity search.

    This is a basic implementation using brute-force cosine similarity.
    Production would use a vector database like ChromaDB or Pinecone.
    """

    memories: list[Memory] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    memory_id_to_idx: dict[str, int] = field(default_factory=dict)
    embedding_service: EmbeddingService | None = None

    def add(self, memory: Memory) -> None:
        """Add a memory to the index."""
        if memory.memory_id in self.memory_id_to_idx:
            return  # Already indexed

        idx = len(self.memories)
        self.memories.append(memory)
        self.memory_id_to_idx[memory.memory_id] = idx

        # Store or compute embedding
        if memory.embedding_vector:
            self.embeddings.append(memory.embedding_vector)
        elif self.embedding_service:
            embedding = self.embedding_service.encode(memory.content)
            self.embeddings.append(embedding)
            memory.embedding_vector = embedding
        else:
            self.embeddings.append([])

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_memory_ids: list[str] | None = None,
    ) -> list[Memory]:
        """Search for memories similar to query."""
        if not self.memories or not self.embedding_service:
            return []

        query_emb = np.array(self.embedding_service.encode(query))

        # Compute similarities
        scores: list[tuple[int, float]] = []
        for idx, emb in enumerate(self.embeddings):
            if not emb:
                continue

            memory = self.memories[idx]
            if filter_memory_ids and memory.memory_id not in filter_memory_ids:
                continue

            emb_arr = np.array(emb)
            score = float(np.dot(query_emb, emb_arr))
            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        results: list[Memory] = []
        for idx, score in scores[:top_k]:
            memory = self.memories[idx]
            # Create a copy with the similarity score
            result = Memory(
                memory_id=memory.memory_id,
                content=memory.content,
                timestamp=memory.timestamp,
                similarity_score=score,
                embedding_vector=memory.embedding_vector,
            )
            results.append(result)

        return results

    def get_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        """Get memories by their IDs."""
        results: list[Memory] = []
        for memory_id in memory_ids:
            if memory_id in self.memory_id_to_idx:
                idx = self.memory_id_to_idx[memory_id]
                results.append(self.memories[idx])
        return results


@dataclass
class SimpleEpisodeIndex:
    """Simple in-memory episode index.

    Stores episode summaries for temporal queries.
    """

    episodes: list[EpisodeSummary] = field(default_factory=list)
    embedding_service: EmbeddingService | None = None
    episode_embeddings: list[list[float]] = field(default_factory=list)

    def add(self, episode: EpisodeSummary) -> None:
        """Add an episode to the index."""
        self.episodes.append(episode)

        # Compute embedding for the summary
        if self.embedding_service:
            text = f"{episode.title}. {episode.summary}"
            embedding = self.embedding_service.encode(text)
            self.episode_embeddings.append(embedding)
        else:
            self.episode_embeddings.append([])

    def search_by_time(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 10,
    ) -> list[EpisodeSummary]:
        """Search episodes by time range."""
        results: list[EpisodeSummary] = []

        for episode in self.episodes:
            # Check time range
            if start_time and episode.end_time < start_time:
                continue
            if end_time and episode.start_time > end_time:
                continue
            results.append(episode)

        # Sort by recency (most recent first)
        results.sort(key=lambda e: e.end_time, reverse=True)

        return results[:limit]

    def search_by_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EpisodeSummary]:
        """Search episodes by semantic similarity."""
        if not self.episodes or not self.embedding_service:
            return []

        query_emb = np.array(self.embedding_service.encode(query))

        # Compute similarities
        scores: list[tuple[int, float]] = []
        for idx, emb in enumerate(self.episode_embeddings):
            if not emb:
                continue
            emb_arr = np.array(emb)
            score = float(np.dot(query_emb, emb_arr))
            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        return [self.episodes[idx] for idx, _ in scores[:top_k]]


@dataclass
class SimpleTopicClusters:
    """Simple topic clustering for continuity queries.

    Maps memory IDs to cluster IDs and provides cluster-based retrieval.
    """

    memory_to_cluster: dict[str, str] = field(default_factory=dict)
    cluster_to_memories: dict[str, list[str]] = field(default_factory=dict)
    cluster_names: dict[str, str] = field(default_factory=dict)
    cluster_centroids: dict[str, list[float]] = field(default_factory=dict)
    embedding_service: EmbeddingService | None = None

    def add_memory_to_cluster(
        self,
        memory_id: str,
        cluster_id: str,
        cluster_name: str = "",
    ) -> None:
        """Add a memory to a cluster."""
        self.memory_to_cluster[memory_id] = cluster_id

        if cluster_id not in self.cluster_to_memories:
            self.cluster_to_memories[cluster_id] = []
        self.cluster_to_memories[cluster_id].append(memory_id)

        if cluster_name:
            self.cluster_names[cluster_id] = cluster_name

    def set_cluster_centroid(
        self,
        cluster_id: str,
        centroid: list[float],
    ) -> None:
        """Set the centroid embedding for a cluster."""
        self.cluster_centroids[cluster_id] = centroid

    def find_cluster(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[TopicMatch]:
        """Find topic clusters matching query."""
        if not self.cluster_centroids or not self.embedding_service:
            return []

        query_emb = np.array(self.embedding_service.encode(query))

        # Compute similarities to cluster centroids
        scores: list[tuple[str, float]] = []
        for cluster_id, centroid in self.cluster_centroids.items():
            centroid_arr = np.array(centroid)
            score = float(np.dot(query_emb, centroid_arr))
            scores.append((cluster_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        results: list[TopicMatch] = []
        for cluster_id, score in scores[:top_k]:
            results.append(
                TopicMatch(
                    cluster_id=cluster_id,
                    cluster_name=self.cluster_names.get(cluster_id, cluster_id),
                    relevance_score=score,
                    memory_ids=self.cluster_to_memories.get(cluster_id, []),
                )
            )

        return results

    def get_recent_in_cluster(
        self,
        cluster_id: str,
        limit: int = 10,
    ) -> list[str]:
        """Get recent memory IDs in a cluster.

        Note: Without timestamp info, returns last N added.
        """
        memory_ids = self.cluster_to_memories.get(cluster_id, [])
        return memory_ids[-limit:]


# =============================================================================
# Unified Retriever
# =============================================================================


class UnifiedRetriever:
    """Main unified retrieval pipeline.

    Orchestrates query classification and routes to appropriate retrieval
    strategies based on query type.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        memory_index: SimpleMemoryIndex,
        episode_index: SimpleEpisodeIndex,
        topic_clusters: SimpleTopicClusters,
        embedding_service: EmbeddingService,
        classifier: LLMQueryClassifier | RuleBasedQueryClassifier | None = None,
        config: UnifiedRetrieverConfig | None = None,
    ):
        self.kg = kg
        self.memory_index = memory_index
        self.episode_index = episode_index
        self.topic_clusters = topic_clusters
        self.embedding_service = embedding_service
        self.classifier = classifier or RuleBasedQueryClassifier()
        self.config = config or UnifiedRetrieverConfig()

    def retrieve(
        self,
        user_input: str,
        conversation_context: list[str] | None = None,
        override_query_type: QueryType | None = None,
    ) -> RetrievalContext:
        """Run retrieval for a single turn.

        Args:
            user_input: The user's input
            conversation_context: Recent conversation history
            override_query_type: Force a specific query type (for ablation)

        Returns:
            RetrievalContext with retrieved information
        """
        start_time = time.time()
        context = conversation_context or []

        # 1. Detect references
        detected_refs: list[DetectedReference] = []
        if self.config.use_reference_detection and hasattr(self.classifier, "detect_references"):
            detected_refs = self.classifier.detect_references(user_input, context)

        # 2. Classify query
        if override_query_type:
            query_type = override_query_type
        elif self.config.use_query_classification:
            query_type = self.classifier.classify(user_input, context, detected_refs)
        else:
            query_type = QueryType.PROACTIVE_CONTEXT

        # 3. Route and retrieve
        result = self._route_and_retrieve(query_type, detected_refs, user_input, context)

        # Add timing and metadata
        result.latency_ms = (time.time() - start_time) * 1000
        result.detected_references = detected_refs

        return result

    def _route_and_retrieve(
        self,
        query_type: QueryType,
        refs: list[DetectedReference],
        user_input: str,
        context: list[str],
    ) -> RetrievalContext:
        """Route to appropriate strategy based on query type."""
        if query_type == QueryType.NO_RETRIEVAL:
            return RetrievalContext(
                query_type=query_type,
                strategy_used="skip",
            )

        if query_type == QueryType.CURRENT_STATE:
            return self._retrieve_current_state(user_input, refs)

        if query_type == QueryType.HISTORY:
            return self._retrieve_history(user_input, refs)

        if query_type == QueryType.ENTITY_OVERVIEW:
            return self._retrieve_entity_overview(user_input, refs)

        if query_type == QueryType.TEMPORAL:
            return self._retrieve_temporal(user_input, refs)

        if query_type == QueryType.CONTINUITY:
            return self._retrieve_continuity(user_input, refs, context)

        if query_type == QueryType.PROACTIVE_CONTEXT:
            return self._retrieve_proactive(user_input, refs, context)

        # Default fallback
        return self._retrieve_proactive(user_input, refs, context)

    def _retrieve_current_state(
        self,
        query: str,
        refs: list[DetectedReference],
    ) -> RetrievalContext:
        """Retrieve current state using KG.

        For queries like "What is X wearing?" - get most recent value.
        """
        # Use KG-aware retrieval
        memory_ids = retrieve_kg_aware(query, self.kg, top_k=self.config.kg_top_k)

        # Get the actual memories
        memories = self.memory_index.get_by_ids(memory_ids)

        # Get facts from KG
        facts = self._extract_facts_for_query(query)

        # Format context
        context_text = self._format_context(facts=facts, memories=memories)

        return RetrievalContext(
            query_type=QueryType.CURRENT_STATE,
            strategy_used="kg_current_state",
            facts=facts,
            memories=memories,
            context_text=context_text,
            num_candidates_searched=len(self.kg.facts),
        )

    def _retrieve_history(
        self,
        query: str,
        refs: list[DetectedReference],
    ) -> RetrievalContext:
        """Retrieve history using similarity search.

        For queries like "What has X worn?" - get all relevant memories.
        """
        memories = self.memory_index.search(query, top_k=self.config.similarity_top_k)

        # Format context
        context_text = self._format_context(memories=memories)

        return RetrievalContext(
            query_type=QueryType.HISTORY,
            strategy_used="similarity_search",
            memories=memories,
            context_text=context_text,
            num_candidates_searched=len(self.memory_index.memories),
        )

    def _retrieve_entity_overview(
        self,
        query: str,
        refs: list[DetectedReference],
    ) -> RetrievalContext:
        """Retrieve entity overview using KG aggregation.

        For queries like "What do I know about X?" - get all facts.
        """
        # Find entity in query
        entity_id = self.kg.resolve_entity_from_query(query)

        facts: list[Fact] = []
        memory_ids: list[str] = []

        if entity_id:
            # Get all current state facts for entity
            kg_facts = self.kg.get_current_state(entity_id)
            for kf in kg_facts:
                entity = self.kg.entity_resolver.entities.get(kf.entity_id)
                facts.append(
                    Fact(
                        entity_id=kf.entity_id,
                        entity_name=entity.canonical_name if entity else kf.entity_id,
                        attribute=kf.attribute_id.replace("attr_", ""),
                        value=kf.value,
                        source_memory_id=kf.source_memory_id,
                        timestamp=kf.timestamp,
                        is_current=True,
                    )
                )
                memory_ids.append(kf.source_memory_id)

        # Get the actual memories
        memories = self.memory_index.get_by_ids(list(set(memory_ids)))

        # Format context
        context_text = self._format_context(facts=facts, memories=memories)

        return RetrievalContext(
            query_type=QueryType.ENTITY_OVERVIEW,
            strategy_used="kg_entity_overview",
            facts=facts,
            memories=memories,
            context_text=context_text,
            num_candidates_searched=len(self.kg.facts),
        )

    def _retrieve_temporal(
        self,
        query: str,
        refs: list[DetectedReference],
    ) -> RetrievalContext:
        """Retrieve temporal context using episode index.

        For queries like "What happened yesterday?" - find relevant episodes.
        """
        # Search episodes by semantic similarity
        episodes = self.episode_index.search_by_query(
            query, top_k=self.config.episode_top_k
        )

        # Also search by time if we detect time references
        time_refs = [r for r in refs if r.reference_type == "time"]
        if time_refs:
            # Could parse time references here for more precise retrieval
            # For now, rely on semantic search
            pass

        # Get memories from top episodes
        memory_ids: list[str] = []
        for ep in episodes[:3]:  # Top 3 episodes
            memory_ids.extend(ep.memory_ids[:5])  # First 5 memories per episode

        memories = self.memory_index.get_by_ids(memory_ids)

        # Format context
        context_text = self._format_context(episodes=episodes, memories=memories)

        return RetrievalContext(
            query_type=QueryType.TEMPORAL,
            strategy_used="episode_search",
            episodes=episodes,
            memories=memories,
            context_text=context_text,
            num_candidates_searched=len(self.episode_index.episodes),
        )

    def _retrieve_continuity(
        self,
        query: str,
        refs: list[DetectedReference],
        context: list[str],
    ) -> RetrievalContext:
        """Retrieve continuity context using topics + recency.

        For queries like "How did the interview go?" - find recent + topical.
        """
        # Find relevant topic clusters
        topic_matches = self.topic_clusters.find_cluster(
            query, top_k=self.config.topic_top_k
        )

        # Get recent memories from matching clusters
        memory_ids: list[str] = []
        for match in topic_matches:
            recent_ids = self.topic_clusters.get_recent_in_cluster(
                match.cluster_id, limit=5
            )
            memory_ids.extend(recent_ids)

        # Also add similarity search results
        sim_memories = self.memory_index.search(
            query, top_k=self.config.similarity_top_k // 2
        )
        for m in sim_memories:
            if m.memory_id not in memory_ids:
                memory_ids.append(m.memory_id)

        memories = self.memory_index.get_by_ids(memory_ids)

        # Format context
        context_text = self._format_context(topics=topic_matches, memories=memories)

        return RetrievalContext(
            query_type=QueryType.CONTINUITY,
            strategy_used="topic_continuity",
            topics=topic_matches,
            memories=memories,
            context_text=context_text,
            num_candidates_searched=len(self.memory_index.memories),
        )

    def _retrieve_proactive(
        self,
        query: str,
        refs: list[DetectedReference],
        context: list[str],
    ) -> RetrievalContext:
        """Retrieve proactive context for detected references.

        When user mentions entities/topics, fetch relevant context.
        """
        facts: list[Fact] = []
        memories: list[Memory] = []
        episodes: list[EpisodeSummary] = []

        # For each reference, fetch appropriate context
        for ref in refs:
            if ref.reference_type == "entity":
                # KG lookup for entities
                entity_id = self.kg.entity_resolver.resolve(ref.text)
                if entity_id:
                    kg_facts = self.kg.get_current_state(entity_id)
                    for kf in kg_facts[:3]:  # Limit per entity
                        entity = self.kg.entity_resolver.entities.get(kf.entity_id)
                        facts.append(
                            Fact(
                                entity_id=kf.entity_id,
                                entity_name=entity.canonical_name if entity else kf.entity_id,
                                attribute=kf.attribute_id.replace("attr_", ""),
                                value=kf.value,
                                source_memory_id=kf.source_memory_id,
                                timestamp=kf.timestamp,
                                is_current=True,
                            )
                        )

            elif ref.reference_type == "event":
                # Episode search for events
                ep_results = self.episode_index.search_by_query(ref.text, top_k=2)
                episodes.extend(ep_results)

            elif ref.reference_type == "topic":
                # Topic cluster search
                clusters = self.topic_clusters.find_cluster(ref.text, top_k=1)
                for cluster in clusters:
                    mem_ids = self.topic_clusters.get_recent_in_cluster(
                        cluster.cluster_id, limit=3
                    )
                    memories.extend(self.memory_index.get_by_ids(mem_ids))

        # Also do general similarity search
        if not refs:
            memories = self.memory_index.search(
                query, top_k=self.config.similarity_top_k
            )

        # Deduplicate memories
        seen_ids: set[str] = set()
        unique_memories: list[Memory] = []
        for m in memories:
            if m.memory_id not in seen_ids:
                seen_ids.add(m.memory_id)
                unique_memories.append(m)

        # Format context
        context_text = self._format_context(
            facts=facts, episodes=episodes, memories=unique_memories
        )

        return RetrievalContext(
            query_type=QueryType.PROACTIVE_CONTEXT,
            strategy_used="multi_strategy",
            facts=facts,
            memories=unique_memories,
            episodes=episodes,
            context_text=context_text,
            num_candidates_searched=len(self.memory_index.memories) + len(self.kg.facts),
        )

    def _extract_facts_for_query(self, query: str) -> list[Fact]:
        """Extract relevant facts from KG for a query."""
        facts: list[Fact] = []

        # Find entity in query
        entity_id = self.kg.resolve_entity_from_query(query)
        if not entity_id:
            return facts

        # Get entity and find relevant attribute
        entity = self.kg.entity_resolver.entities.get(entity_id)
        if not entity:
            return facts

        # Try to find specific attribute
        attribute_id = self.kg.resolve_attribute_from_query(query, entity_id)

        if attribute_id:
            # Specific attribute query
            kg_facts = self.kg.get_attribute_facts(entity_id, attribute_id)
        else:
            # General state query
            kg_facts = self.kg.get_current_state(entity_id)

        for kf in kg_facts:
            facts.append(
                Fact(
                    entity_id=kf.entity_id,
                    entity_name=entity.canonical_name,
                    attribute=kf.attribute_id.replace("attr_", ""),
                    value=kf.value,
                    source_memory_id=kf.source_memory_id,
                    timestamp=kf.timestamp,
                    is_current=True,
                )
            )

        return facts

    def _format_context(
        self,
        facts: list[Fact] | None = None,
        memories: list[Memory] | None = None,
        episodes: list[EpisodeSummary] | None = None,
        topics: list[TopicMatch] | None = None,
    ) -> str:
        """Format retrieved context for LLM consumption."""
        parts: list[str] = []

        if facts:
            parts.append("## Known Facts")
            for fact in facts:
                parts.append(f"- {fact.entity_name}'s {fact.attribute}: {fact.value}")
            parts.append("")

        if episodes:
            parts.append("## Relevant Episodes")
            for ep in episodes:
                parts.append(f"### {ep.title}")
                parts.append(ep.summary)
                parts.append("")

        if topics:
            parts.append("## Related Topics")
            for topic in topics:
                parts.append(f"- {topic.cluster_name} (relevance: {topic.relevance_score:.2f})")
            parts.append("")

        if memories:
            parts.append("## Relevant Memories")
            for memory in memories[:10]:  # Limit to 10
                timestamp_str = memory.timestamp.strftime("%Y-%m-%d %H:%M")
                parts.append(f"[{timestamp_str}] {memory.content[:200]}...")
                if self.config.include_source_ids:
                    parts.append(f"  (id: {memory.memory_id})")
            parts.append("")

        return "\n".join(parts)

    # =========================================================================
    # Incremental Update Methods
    # =========================================================================

    def on_new_memory(self, memory: Memory) -> None:
        """Called when a new memory is created.

        Updates relevant indices incrementally.
        """
        # Add to memory index
        self.memory_index.add(memory)

        # TODO: Extract facts and update KG
        # TODO: Assign to topic cluster

    def on_episode_boundary(self, episode: EpisodeSummary) -> None:
        """Called when an episode boundary is detected.

        Updates the episode index.
        """
        self.episode_index.add(episode)
