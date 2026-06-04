"""Build indices for the unified retrieval experiment.

This script loads conversation data and builds:
- Knowledge graph from extracted facts
- Memory index for similarity search
- Episode index for temporal queries
- Topic clusters for continuity queries

Usage:
    uv run python -m agent.experiments.unified_retrieval.build_indices --conversation <id>
"""

import argparse
import json
import logging
import pickle
import uuid
from datetime import datetime
from pathlib import Path

from agent.embedding_service import EmbeddingService, get_embedding_service
from agent.experiments.retrieval.attribute_retrieval import (
    extract_all_facts,
    load_memories as load_trigger_memories,
    TypedFact,
)
from agent.experiments.retrieval.knowledge_graph import KnowledgeGraph
from agent.experiments.episode_summaries.detection import detect_episodes_by_gap
from agent.experiments.topic_clustering.clustering import (
    cluster_cross_action_only,
    prepare_embeddings,
)
from agent.llm import LLM, SupportedModel, create_llm
from agent.memory.dag.models import MemoryElement, ConfidenceLevel
from agent.storage import create_trigger_history

from .models import EpisodeSummary, Memory
from .unified_retriever import (
    SimpleMemoryIndex,
    SimpleEpisodeIndex,
    SimpleTopicClusters,
    UnifiedRetriever,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path("conversations")
CACHE_DIR = Path(__file__).parent / "output" / "cache"


def load_conversation_memories(
    conversation_id: str,
    max_memories: int | None = None,
) -> tuple[list[MemoryElement], list[Memory]]:
    """Load memories from a conversation.

    Returns both MemoryElement (for clustering) and Memory (for our index).
    """
    # Try loading from trigger history JSON - check multiple locations
    possible_paths = [
        CONVERSATIONS_DIR / f"{conversation_id}_triggers.json",
        CONVERSATIONS_DIR / f"{conversation_id}.json",
        CONVERSATIONS_DIR / "archive" / conversation_id / f"{conversation_id}_triggers.json",
        CONVERSATIONS_DIR / "archive" / conversation_id / f"{conversation_id}.json",
    ]

    triggers_file = None
    for path in possible_paths:
        if path.exists():
            triggers_file = path
            break

    if triggers_file is None:
        raise FileNotFoundError(f"Conversation file not found. Tried: {possible_paths}")

    logger.info(f"Loading from {triggers_file}")

    with open(triggers_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    memory_elements: list[MemoryElement] = []
    memories: list[Memory] = []

    entries = data.get("entries", [])
    if max_memories:
        entries = entries[:max_memories]

    for i, entry_data in enumerate(entries):
        entry_id = entry_data.get("entry_id", f"entry_{i}")

        # Try to get timestamp
        timestamp_str = entry_data.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        # Get content - prefer compressed_summary, fall back to other fields
        content = entry_data.get("compressed_summary", "")
        if not content:
            content = entry_data.get("summary", "")
        if not content:
            content = entry_data.get("content", "")

        if not content or len(content) < 20:
            continue

        # Create MemoryElement
        embedding = entry_data.get("embedding_vector")

        memory_element = MemoryElement(
            id=entry_id,
            content=content,
            timestamp=timestamp,
            confidence_level=ConfidenceLevel.STRONG_INFERENCE,
            sequence_in_container=0,
            container_id=entry_id,
            embedding_vector=embedding,
        )
        memory_elements.append(memory_element)

        # Create Memory for our index
        memory = Memory(
            memory_id=entry_id,
            content=content,
            timestamp=timestamp,
            embedding_vector=embedding,
        )
        memories.append(memory)

    logger.info(f"Loaded {len(memories)} memories")
    return memory_elements, memories


def build_knowledge_graph(
    memories: list[Memory],
    embedding_service: EmbeddingService,
    llm: LLM,
    model: SupportedModel,
    cache_key: str,
) -> KnowledgeGraph:
    """Build knowledge graph from memories.

    First extracts facts using LLM, then populates KG.
    """
    # Check for cached facts
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}_facts.json"

    if cache_file.exists():
        logger.info(f"Loading cached facts from {cache_file}")
        with open(cache_file) as f:
            facts_data = json.load(f)
        all_facts = [TypedFact(**f) for f in facts_data]
    else:
        # Convert to MemorySample format for extraction
        from agent.experiments.retrieval.attribute_retrieval import MemorySample

        memory_samples = [
            MemorySample(
                memory_id=m.memory_id,
                content=m.content,
                timestamp=i,
            )
            for i, m in enumerate(memories)
        ]

        logger.info("Extracting facts from memories...")
        all_facts = extract_all_facts(memory_samples, llm, model)

        # Cache the facts
        facts_data = [
            {
                "entity": f.entity,
                "attribute": f.attribute,
                "attribute_type": f.attribute_type,
                "value": f.value,
                "source_memory_id": f.source_memory_id,
                "timestamp": f.timestamp,
            }
            for f in all_facts
        ]
        with open(cache_file, "w") as f:
            json.dump(facts_data, f, indent=2)
        logger.info(f"Cached {len(all_facts)} facts to {cache_file}")

    # Build KG
    kg = KnowledgeGraph(embedding_service)

    for fact in all_facts:
        kg.add_fact(
            raw_entity=fact.entity,
            raw_attribute=fact.attribute,
            value=fact.value,
            source_memory_id=fact.source_memory_id,
            timestamp=fact.timestamp,
        )

    logger.info(f"Built KG with {len(kg.facts)} facts, {len(kg.entity_resolver.entities)} entities")
    return kg


def build_memory_index(
    memories: list[Memory],
    embedding_service: EmbeddingService,
) -> SimpleMemoryIndex:
    """Build memory index for similarity search."""
    index = SimpleMemoryIndex(embedding_service=embedding_service)

    for memory in memories:
        # Compute embedding if not present
        if not memory.embedding_vector:
            memory.embedding_vector = embedding_service.encode(memory.content)
        index.add(memory)

    logger.info(f"Built memory index with {len(index.memories)} memories")
    return index


def build_episode_index(
    memory_elements: list[MemoryElement],
    embedding_service: EmbeddingService,
    gap_minutes: int = 60,
) -> SimpleEpisodeIndex:
    """Build episode index for temporal queries."""
    # Detect episodes using time gaps
    detection_result = detect_episodes_by_gap(memory_elements, gap_minutes)

    index = SimpleEpisodeIndex(embedding_service=embedding_service)

    for ep in detection_result.episodes:
        summary = EpisodeSummary(
            episode_id=ep.id,
            title=f"Episode {ep.start_time.strftime('%Y-%m-%d %H:%M')}",
            summary=f"Conversation from {ep.start_time.strftime('%Y-%m-%d %H:%M')} to {ep.end_time.strftime('%Y-%m-%d %H:%M')} with {ep.memory_count} memories",
            start_time=ep.start_time,
            end_time=ep.end_time,
            memory_ids=ep.memory_ids,
            key_events=[],
            topics=[],
        )
        index.add(summary)

    logger.info(f"Built episode index with {len(index.episodes)} episodes")
    return index


def build_topic_clusters(
    memory_elements: list[MemoryElement],
    embedding_service: EmbeddingService,
    n_clusters: int = 15,
) -> SimpleTopicClusters:
    """Build topic clusters for continuity queries.

    Note: This requires trigger_history for accurate action type detection.
    For simplicity, we'll use a simpler clustering approach here.
    """
    clusters = SimpleTopicClusters(embedding_service=embedding_service)

    # Filter memories with embeddings
    valid_memories = [m for m in memory_elements if m.embedding_vector]

    if len(valid_memories) < n_clusters:
        logger.warning(f"Not enough memories ({len(valid_memories)}) for {n_clusters} clusters")
        n_clusters = max(2, len(valid_memories) // 3)

    if len(valid_memories) < 3:
        logger.warning("Not enough memories for clustering")
        return clusters

    # Simple k-means clustering
    try:
        from sklearn.cluster import KMeans
        import numpy as np

        embeddings = np.array([m.embedding_vector for m in valid_memories])
        memory_ids = [m.id for m in valid_memories]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Build cluster mappings
        for idx, (memory_id, label) in enumerate(zip(memory_ids, labels)):
            cluster_id = f"cluster_{label}"
            clusters.add_memory_to_cluster(
                memory_id=memory_id,
                cluster_id=cluster_id,
                cluster_name=f"Topic {label}",
            )

        # Store cluster centroids
        for i, centroid in enumerate(kmeans.cluster_centers_):
            cluster_id = f"cluster_{i}"
            clusters.set_cluster_centroid(cluster_id, centroid.tolist())

        logger.info(f"Built {n_clusters} topic clusters")

    except ImportError:
        logger.warning("sklearn not available, skipping topic clustering")

    return clusters


def save_indices(
    kg: KnowledgeGraph,
    memory_index: SimpleMemoryIndex,
    episode_index: SimpleEpisodeIndex,
    topic_clusters: SimpleTopicClusters,
    output_dir: Path,
) -> None:
    """Save indices to disk for later use."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pickle for now (could use more robust serialization)
    with open(output_dir / "kg.pkl", "wb") as f:
        pickle.dump(kg, f)

    with open(output_dir / "memory_index.pkl", "wb") as f:
        pickle.dump(memory_index, f)

    with open(output_dir / "episode_index.pkl", "wb") as f:
        pickle.dump(episode_index, f)

    with open(output_dir / "topic_clusters.pkl", "wb") as f:
        pickle.dump(topic_clusters, f)

    logger.info(f"Saved indices to {output_dir}")


def load_indices(
    input_dir: Path,
    embedding_service: EmbeddingService,
) -> tuple[KnowledgeGraph, SimpleMemoryIndex, SimpleEpisodeIndex, SimpleTopicClusters]:
    """Load indices from disk."""
    with open(input_dir / "kg.pkl", "rb") as f:
        kg = pickle.load(f)

    with open(input_dir / "memory_index.pkl", "rb") as f:
        memory_index = pickle.load(f)

    with open(input_dir / "episode_index.pkl", "rb") as f:
        episode_index = pickle.load(f)

    with open(input_dir / "topic_clusters.pkl", "rb") as f:
        topic_clusters = pickle.load(f)

    # Re-attach embedding service
    memory_index.embedding_service = embedding_service
    episode_index.embedding_service = embedding_service
    topic_clusters.embedding_service = embedding_service

    logger.info(f"Loaded indices from {input_dir}")
    return kg, memory_index, episode_index, topic_clusters


def build_all_indices(
    conversation_id: str,
    max_memories: int | None = None,
    output_dir: Path | None = None,
) -> UnifiedRetriever:
    """Build all indices for a conversation.

    Args:
        conversation_id: ID of the conversation to load
        max_memories: Maximum number of memories to load
        output_dir: Directory to save indices (optional)

    Returns:
        Configured UnifiedRetriever instance
    """
    # Initialize services
    embedding_service = get_embedding_service()
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    # Load conversation
    memory_elements, memories = load_conversation_memories(
        conversation_id, max_memories
    )

    if not memories:
        raise ValueError(f"No memories loaded from conversation {conversation_id}")

    # Create cache key from conversation ID and memory count
    cache_key = f"{conversation_id}_{len(memories)}"

    # Build indices
    logger.info("Building knowledge graph...")
    kg = build_knowledge_graph(memories, embedding_service, llm, model, cache_key)

    logger.info("Building memory index...")
    memory_index = build_memory_index(memories, embedding_service)

    logger.info("Building episode index...")
    episode_index = build_episode_index(memory_elements, embedding_service)

    logger.info("Building topic clusters...")
    topic_clusters = build_topic_clusters(memory_elements, embedding_service)

    # Save if output directory specified
    if output_dir:
        save_indices(kg, memory_index, episode_index, topic_clusters, output_dir)

    # Create and return retriever
    retriever = UnifiedRetriever(
        kg=kg,
        memory_index=memory_index,
        episode_index=episode_index,
        topic_clusters=topic_clusters,
        embedding_service=embedding_service,
    )

    return retriever


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build indices for unified retrieval experiment"
    )
    parser.add_argument(
        "--conversation",
        type=str,
        required=True,
        help="Conversation ID to load",
    )
    parser.add_argument(
        "--max-memories",
        type=int,
        default=None,
        help="Maximum number of memories to load",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save indices",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else CACHE_DIR / args.conversation

    retriever = build_all_indices(
        conversation_id=args.conversation,
        max_memories=args.max_memories,
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Knowledge Graph: {len(retriever.kg.facts)} facts, {len(retriever.kg.entity_resolver.entities)} entities")
    print(f"Memory Index: {len(retriever.memory_index.memories)} memories")
    print(f"Episode Index: {len(retriever.episode_index.episodes)} episodes")
    print(f"Topic Clusters: {len(retriever.topic_clusters.cluster_to_memories)} clusters")
    print(f"\nIndices saved to: {output_dir}")


if __name__ == "__main__":
    main()
