"""
Semantic clustering for tier 4 memories.

Groups memories by topic/theme using embedding similarity, regardless of
when they occurred. Uses hierarchical clustering to find natural topic groups.
"""

import logging
import uuid
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
from pydantic import BaseModel, Field

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.memory.models import MemoryElement, MemoryGraph
from agent.embedding_service import get_embedding_service
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from .models import ConversationBoundary, SemanticCluster

logger = logging.getLogger(__name__)


# Clustering configuration
MIN_CLUSTER_SIZE = 2  # Minimum elements to form a cluster
SIMILARITY_THRESHOLD = 0.6  # Cosine similarity threshold for clustering
MAX_CLUSTERS = 20  # Maximum number of clusters to create


class ClusterTopicAnalysis(BaseModel):
    """LLM analysis of cluster topic and content."""

    topic: str = Field(description="Concise topic/theme name for this cluster")
    summary: str = Field(description="2-3 sentence summary of the cluster content")
    reasoning: str = Field(description="Why these items belong together")


class EmbeddingCluster:
    """Represents a cluster of similar embeddings."""

    def __init__(self, cluster_id: str):
        self.cluster_id = cluster_id
        self.member_ids: List[str] = []
        self.member_embeddings: List[List[float]] = []
        self.centroid: Optional[np.ndarray] = None

    def add_member(self, member_id: str, embedding: List[float]):
        """Add a member to the cluster."""
        self.member_ids.append(member_id)
        self.member_embeddings.append(embedding)
        self._update_centroid()

    def _update_centroid(self):
        """Recalculate cluster centroid."""
        if self.member_embeddings:
            embeddings_array = np.array(self.member_embeddings)
            self.centroid = np.mean(embeddings_array, axis=0)

    def similarity_to_centroid(self, embedding: List[float]) -> float:
        """Calculate cosine similarity to cluster centroid."""
        if self.centroid is None:
            return 0.0

        embedding_service = get_embedding_service()
        return embedding_service.cosine_similarity(embedding, self.centroid.tolist())

    def merge_with(self, other: "EmbeddingCluster"):
        """Merge another cluster into this one."""
        self.member_ids.extend(other.member_ids)
        self.member_embeddings.extend(other.member_embeddings)
        self._update_centroid()


def hierarchical_clustering(
    elements: List[Tuple[str, List[float]]],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    max_clusters: int = MAX_CLUSTERS,
) -> List[EmbeddingCluster]:
    """
    Perform agglomerative hierarchical clustering on embeddings.

    Args:
        elements: List of (id, embedding) tuples
        similarity_threshold: Minimum similarity to join a cluster
        max_clusters: Maximum number of clusters to create

    Returns:
        List of EmbeddingCluster objects
    """
    if not elements:
        return []

    logger.info(
        f"Clustering {len(elements)} elements with threshold {similarity_threshold}"
    )

    # Initialize: each element is its own cluster
    clusters = [EmbeddingCluster(f"cluster_{i}") for i in range(len(elements))]
    for cluster, (elem_id, embedding) in zip(clusters, elements):
        cluster.add_member(elem_id, embedding)

    embedding_service = get_embedding_service()

    # Agglomerative clustering
    while len(clusters) > max_clusters:
        # Find most similar pair of clusters
        best_similarity = -1
        best_pair = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                centroid_i = clusters[i].centroid
                centroid_j = clusters[j].centroid
                if centroid_i is None or centroid_j is None:
                    continue
                sim = embedding_service.cosine_similarity(
                    centroid_i.tolist(), centroid_j.tolist()
                )
                if sim > best_similarity:
                    best_similarity = sim
                    best_pair = (i, j)

        # Stop if no similar pairs found
        if best_similarity < similarity_threshold:
            break

        # Merge the most similar pair
        if best_pair:
            i, j = best_pair
            clusters[i].merge_with(clusters[j])
            del clusters[j]

            logger.debug(
                f"Merged clusters (similarity: {best_similarity:.3f}), "
                f"now have {len(clusters)} clusters"
            )

    # Filter out single-element clusters
    valid_clusters = [c for c in clusters if len(c.member_ids) >= MIN_CLUSTER_SIZE]

    logger.info(
        f"Created {len(valid_clusters)} valid clusters "
        f"(filtered {len(clusters) - len(valid_clusters)} too-small clusters)"
    )

    return valid_clusters


def create_semantic_clusters_from_conversations(
    conversations: List[ConversationBoundary],
    llm: LLM,
    model: SupportedModel,
    state,
    on_cluster_created=None,
) -> List[SemanticCluster]:
    """
    Create semantic clusters from tier 3 conversations.

    Args:
        conversations: List of conversation boundaries
        llm: LLM instance for topic analysis
        model: Model to use

    Returns:
        List of SemanticCluster objects (tier 4)
    """
    logger.info(f"Creating semantic clusters from {len(conversations)} conversations")

    # Prepare elements for clustering (using conversation embeddings)
    elements = [
        (conv.id, conv.embedding_vector)
        for conv in conversations
        if conv.embedding_vector
    ]

    if not elements:
        logger.warning("No conversations with embeddings found")
        return []

    # Perform clustering
    embedding_clusters = hierarchical_clustering(elements)

    # Convert to SemanticCluster objects with LLM-generated topics
    semantic_clusters = []
    embedding_service = get_embedding_service()

    for cluster in embedding_clusters:
        # Get conversation objects for this cluster
        conv_dict = {conv.id: conv for conv in conversations}
        cluster_conversations = [
            conv_dict[conv_id] for conv_id in cluster.member_ids if conv_id in conv_dict
        ]

        # Generate topic and summary via LLM
        topic_analysis = _analyze_cluster_topic(
            cluster_conversations, llm, model, state
        )

        # Create embedding for cluster summary
        cluster_embedding = embedding_service.encode(topic_analysis.summary)

        # Collect all lower-tier references
        all_trigger_entry_ids = []
        for conv in cluster_conversations:
            all_trigger_entry_ids.extend(conv.trigger_entry_ids)

        semantic_cluster = SemanticCluster(
            id=str(uuid.uuid4()),
            cluster_topic=topic_analysis.topic,
            summary=topic_analysis.summary,
            embedding_vector=cluster_embedding,
            conversation_ids=[conv.id for conv in cluster_conversations],
            trigger_entry_ids=all_trigger_entry_ids,
            memory_element_ids=[],  # Will be populated if we drill down further
            cluster_size=len(all_trigger_entry_ids),
        )

        semantic_clusters.append(semantic_cluster)

        logger.info(
            f"Created cluster '{topic_analysis.topic}' with "
            f"{len(cluster_conversations)} conversations, "
            f"{len(all_trigger_entry_ids)} trigger entries"
        )

        # Call callback if provided
        if on_cluster_created:
            on_cluster_created(semantic_cluster)

    logger.info(f"Created {len(semantic_clusters)} semantic clusters (tier 4)")

    return semantic_clusters


def create_semantic_clusters_from_trigger_entries(
    trigger_entries: List[TriggerHistoryEntry],
    llm: LLM,
    model: SupportedModel,
    state,
    on_cluster_created=None,
) -> List[SemanticCluster]:
    """
    Create semantic clusters directly from tier 2 trigger entries.

    Useful when you want to cluster without conversation boundaries.

    Args:
        trigger_entries: List of trigger history entries
        llm: LLM instance
        model: Model to use

    Returns:
        List of SemanticCluster objects
    """
    logger.info(
        f"Creating semantic clusters from {len(trigger_entries)} trigger entries"
    )

    # Prepare elements for clustering
    elements = [
        (entry.entry_id, entry.embedding_vector)
        for entry in trigger_entries
        if entry.embedding_vector
    ]

    if not elements:
        logger.warning("No trigger entries with embeddings found")
        return []

    # Perform clustering
    embedding_clusters = hierarchical_clustering(elements)

    # Convert to SemanticCluster objects
    semantic_clusters = []
    embedding_service = get_embedding_service()

    entry_dict = {entry.entry_id: entry for entry in trigger_entries}

    for cluster in embedding_clusters:
        # Get trigger entries for this cluster
        cluster_entries = [
            entry_dict[entry_id]
            for entry_id in cluster.member_ids
            if entry_id in entry_dict
        ]

        # Generate topic and summary
        topic_analysis = _analyze_trigger_cluster_topic(
            cluster_entries, llm, model, state
        )

        # Create embedding for cluster summary
        cluster_embedding = embedding_service.encode(topic_analysis.summary)

        semantic_cluster = SemanticCluster(
            id=str(uuid.uuid4()),
            cluster_topic=topic_analysis.topic,
            summary=topic_analysis.summary,
            embedding_vector=cluster_embedding,
            conversation_ids=[],
            trigger_entry_ids=[entry.entry_id for entry in cluster_entries],
            memory_element_ids=[],
            cluster_size=len(cluster_entries),
        )

        semantic_clusters.append(semantic_cluster)

        logger.info(
            f"Created cluster '{topic_analysis.topic}' with "
            f"{len(cluster_entries)} trigger entries"
        )

        # Call callback if provided
        if on_cluster_created:
            on_cluster_created(semantic_cluster)

    return semantic_clusters


def _analyze_cluster_topic(
    conversations: List[ConversationBoundary],
    llm: LLM,
    model: SupportedModel,
    state,
) -> ClusterTopicAnalysis:
    """Analyze a cluster of conversations to determine topic."""
    from agent.state import build_agent_state_description
    from agent.chain_of_action.prompts import format_section

    # Build context
    conv_summaries = []
    for i, conv in enumerate(
        conversations[:10]
    ):  # Limit to first 10 for token efficiency
        timestamp = conv.start_timestamp.strftime("%Y-%m-%d %H:%M")
        tags = ", ".join(conv.topic_tags)
        conv_summaries.append(f"{i+1}. [{timestamp}] Tags: {tags}\n   {conv.summary}")

    state_desc = build_agent_state_description(state)

    prompt = f"""I am {state.name}, {state.role}. I'm analyzing a cluster of my related conversations to identify the common topic or theme.

{state_desc}

{format_section(
    "CONVERSATIONS IN THIS CLUSTER",
    "\n".join(conv_summaries)
)}

My task:
1. Identify the main topic or theme that connects these conversations
2. Create a concise 2-3 sentence summary written in FIRST PERSON about what this cluster represents in my experience
3. Explain why these conversations belong together

The summary should describe what I explored, discussed, or experienced across these conversations. Use "I" or "we" language, not third-person."""

    try:
        analysis = direct_structured_llm_call(
            prompt=prompt,
            response_model=ClusterTopicAnalysis,
            model=model,
            llm=llm,
            caller="cluster_topic_analysis",
        )
        return analysis

    except Exception as e:
        logger.warning(f"Cluster topic analysis failed: {e}")
        # Fallback
        return ClusterTopicAnalysis(
            topic="General Discussion",
            summary=f"Cluster of {len(conversations)} related conversations",
            reasoning="Fallback due to analysis failure",
        )


def _analyze_trigger_cluster_topic(
    trigger_entries: List[TriggerHistoryEntry],
    llm: LLM,
    model: SupportedModel,
    state,
) -> ClusterTopicAnalysis:
    """Analyze a cluster of trigger entries to determine topic."""
    from agent.state import build_agent_state_description
    from agent.chain_of_action.prompts import format_section

    # Build context
    entry_summaries = []
    for i, entry in enumerate(trigger_entries[:10]):  # Limit for tokens
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M")
        summary = entry.compressed_summary or "No summary"
        entry_summaries.append(f"{i+1}. [{timestamp}] {summary}")

    state_desc = build_agent_state_description(state)

    prompt = f"""I am {state.name}, {state.role}. I'm analyzing a cluster of my related interactions to identify the common topic or theme.

{state_desc}

{format_section(
    "INTERACTIONS IN THIS CLUSTER",
    "\n".join(entry_summaries)
)}

My task:
1. Identify the main topic or theme that connects these interactions
2. Create a concise 2-3 sentence summary written in FIRST PERSON about what this cluster represents
3. Explain why these interactions belong together

The summary should describe what I experienced or engaged with. Use "I" or "we" language, not third-person."""

    try:
        analysis = direct_structured_llm_call(
            prompt=prompt,
            response_model=ClusterTopicAnalysis,
            model=model,
            llm=llm,
            caller="trigger_cluster_topic_analysis",
        )
        return analysis

    except Exception as e:
        logger.warning(f"Cluster topic analysis failed: {e}")
        return ClusterTopicAnalysis(
            topic="General Discussion",
            summary=f"Cluster of {len(trigger_entries)} related interactions",
            reasoning="Fallback due to analysis failure",
        )


def create_semantic_clusters_from_memory_elements(
    memory_graph: MemoryGraph,
    llm: LLM,
    model: SupportedModel,
    state,
    on_cluster_created=None,
) -> List[SemanticCluster]:
    """
    Create semantic clusters directly from tier 1 memory elements.

    Most fine-grained clustering option.

    Args:
        memory_graph: Memory graph with tier 1 elements
        llm: LLM instance
        model: Model to use

    Returns:
        List of SemanticCluster objects
    """
    logger.info(f"Creating semantic clusters from memory elements")

    # Prepare elements for clustering
    elements = [
        (mem.id, mem.embedding_vector)
        for mem in memory_graph.elements.values()
        if mem.embedding_vector
    ]

    if not elements:
        logger.warning("No memory elements with embeddings found")
        return []

    # Perform clustering
    embedding_clusters = hierarchical_clustering(elements)

    # Convert to SemanticCluster objects
    semantic_clusters = []
    embedding_service = get_embedding_service()

    for cluster in embedding_clusters:
        # Get memory elements for this cluster
        cluster_memories = [
            memory_graph.elements[mem_id]
            for mem_id in cluster.member_ids
            if mem_id in memory_graph.elements
        ]

        # Generate summary
        topic_analysis = _analyze_memory_cluster_topic(
            cluster_memories, llm, model, state
        )

        # Create embedding for cluster summary
        cluster_embedding = embedding_service.encode(topic_analysis.summary)

        # Group by container to get trigger entry IDs
        container_ids = set(mem.container_id for mem in cluster_memories)

        semantic_cluster = SemanticCluster(
            id=str(uuid.uuid4()),
            cluster_topic=topic_analysis.topic,
            summary=topic_analysis.summary,
            embedding_vector=cluster_embedding,
            conversation_ids=[],
            trigger_entry_ids=list(container_ids),
            memory_element_ids=[mem.id for mem in cluster_memories],
            cluster_size=len(cluster_memories),
        )

        semantic_clusters.append(semantic_cluster)

        logger.info(
            f"Created cluster '{topic_analysis.topic}' with "
            f"{len(cluster_memories)} memory elements"
        )

        # Call callback if provided
        if on_cluster_created:
            on_cluster_created(semantic_cluster)

    return semantic_clusters


def _analyze_memory_cluster_topic(
    memories: List[MemoryElement],
    llm: LLM,
    model: SupportedModel,
    state,
) -> ClusterTopicAnalysis:
    """Analyze a cluster of memory elements to determine topic."""
    from agent.state import build_agent_state_description
    from agent.chain_of_action.prompts import format_section

    # Build context
    memory_contents = []
    for i, mem in enumerate(memories[:15]):  # Limit for tokens
        timestamp = mem.timestamp.strftime("%Y-%m-%d %H:%M")
        content = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content
        memory_contents.append(f"{i+1}. [{timestamp}] {content}")

    state_desc = build_agent_state_description(state)

    prompt = f"""I am {state.name}, {state.role}. I'm analyzing a cluster of my related memories to identify the common topic or theme.

{state_desc}

{format_section(
    "MEMORIES IN THIS CLUSTER",
    "\n".join(memory_contents)
)}

My task:
1. Identify the main topic or theme that connects these memories
2. Create a concise 2-3 sentence summary written in FIRST PERSON about what this cluster represents
3. Explain why these memories belong together

The summary should describe what I remember or what these memories capture about my experience. Use "I" or "we" language, not third-person."""

    try:
        analysis = direct_structured_llm_call(
            prompt=prompt,
            response_model=ClusterTopicAnalysis,
            model=model,
            llm=llm,
            caller="memory_cluster_topic_analysis",
        )
        return analysis

    except Exception as e:
        logger.warning(f"Cluster topic analysis failed: {e}")
        return ClusterTopicAnalysis(
            topic="General Memories",
            summary=f"Cluster of {len(memories)} related memories",
            reasoning="Fallback due to analysis failure",
        )
