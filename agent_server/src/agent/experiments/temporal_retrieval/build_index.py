"""
Build episode index from conversation data.

Loads memories, detects episodes, generates summaries,
extracts topics and moods, and builds the index.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

from agent.experiments.episode_summaries.detection import detect_episodes_llm_filtered
from agent.experiments.episode_summaries.models import Episode
from agent.experiments.episode_summaries.summarization import (
    generate_episode_summary,
    generate_episode_title,
)
from agent.experiments.temporal_retrieval.episode_index import EpisodeIndex
from agent.experiments.temporal_retrieval.models import IndexedEpisode
from agent.llm.models import SupportedModel
from agent.llm.router import LLM
from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.memory.dag.dag_memory_manager import DagMemoryManager
from agent.memory.dag.models import MemoryElement
from agent.storage import ITriggerHistory


class _StubTriggerHistory(ITriggerHistory):
    """Minimal trigger history stub for read-only memory loading."""

    def add_entry(self, entry: TriggerHistoryEntry) -> None:
        pass

    def update_entry(self, entry: TriggerHistoryEntry) -> None:
        pass

    def get_first_entry(self) -> TriggerHistoryEntry | None:
        return None

    def get_last_entry(self) -> TriggerHistoryEntry | None:
        return None

    def get_entry_by_id(self, entry_id: str) -> TriggerHistoryEntry:
        raise KeyError(entry_id)

    def get_entry_count(self) -> int:
        return 0

    def iter_entries(
        self, reverse: bool = False, start: int = 0
    ) -> list[TriggerHistoryEntry]:
        return []

    def get_entry_index(self, entry_id: str) -> int:
        raise KeyError(entry_id)

    def get_last_entry_by_trigger_type(
        self, trigger_type: str
    ) -> TriggerHistoryEntry | None:
        return None

    def __len__(self) -> int:
        return 0

    def close(self) -> None:
        pass


def load_memories(
    conversations_dir: Path,
    conversation_prefix: str,
) -> list[MemoryElement]:
    """Load memories from a conversation file.

    Supports both:
    - Archive format: {dir}/archive/{prefix}/{prefix}_dag.json
    - Standard format: {dir}/{prefix}/dag.json
    """
    # Try archive format first (for archived conversations)
    archive_dag_file = (
        conversations_dir / "archive" / conversation_prefix /
        f"{conversation_prefix}_dag.json"
    )

    if archive_dag_file.exists():
        # Load directly from archive format
        trigger_history = _StubTriggerHistory()
        dag = DagMemoryManager.load_from_file(
            str(archive_dag_file),
            trigger_history,
            use_individual_formatting=True,
        )
        memory_graph = dag.get_memory_graph()
        return list(memory_graph.elements.values())

    # Fall back to standard format
    standard_dag_file = conversations_dir / conversation_prefix / "dag.json"
    if standard_dag_file.exists():
        trigger_history = _StubTriggerHistory()
        dag = DagMemoryManager.load_from_file(
            str(standard_dag_file),
            trigger_history,
            use_individual_formatting=True,
        )
        memory_graph = dag.get_memory_graph()
        return list(memory_graph.elements.values())

    raise FileNotFoundError(
        f"Could not find DAG file for {conversation_prefix} in "
        f"{conversations_dir} (tried archive and standard formats)"
    )


def extract_topics_and_moods(
    episode: Episode,
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
) -> tuple[list[str], list[str], float]:
    """
    Extract topics and moods from an episode using LLM.

    Args:
        episode: The episode to analyze
        memories: All memories
        llm: LLM router
        model: Model to use

    Returns:
        Tuple of (topics, moods, emotional_intensity)
    """
    # Get episode memories
    episode_memory_ids = set(episode.memory_ids)
    episode_memories = [m for m in memories if m.id in episode_memory_ids]
    episode_memories.sort(key=lambda m: m.timestamp)

    # Build content for analysis
    content_lines = []
    for m in episode_memories[:20]:  # Limit to first 20 memories
        content_lines.append(m.content[:200])

    content = "\n".join(content_lines)

    prompt = f"""Analyze this conversation excerpt and extract:
1. Main topics discussed (2-5 topics, single words or short phrases)
2. Emotional moods present (1-3 moods like "stressed", "happy", "tired", etc.)
3. Emotional intensity (0.0 to 1.0)

Conversation:
{content}

Respond in JSON format:
{{"topics": ["topic1", "topic2"], "moods": ["mood1"], "intensity": 0.5}}

JSON:"""

    response = llm.generate(
        model=model,
        prompt=prompt,
        caller="topic_mood_extraction",
    )

    # Parse JSON response
    try:
        # Find JSON in response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            topics = data.get("topics", [])
            moods = data.get("moods", [])
            intensity = float(data.get("intensity", 0.5))
            return topics, moods, intensity
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract from content heuristically
    topics = []
    moods = []
    intensity = 0.5

    content_lower = content.lower()

    # Topic extraction heuristics
    topic_keywords = [
        "work",
        "project",
        "code",
        "meeting",
        "task",
        "bug",
        "feature",
        "test",
        "deploy",
        "review",
        "plan",
        "idea",
        "thought",
        "feeling",
        "morning",
        "evening",
        "night",
    ]
    for keyword in topic_keywords:
        if keyword in content_lower:
            topics.append(keyword)
            if len(topics) >= 5:
                break

    # Mood extraction heuristics
    mood_keywords = {
        "stressed": ["stressed", "anxious", "worried", "pressure"],
        "happy": ["happy", "excited", "great", "wonderful"],
        "tired": ["tired", "exhausted", "sleepy"],
        "calm": ["calm", "relaxed", "peaceful"],
        "focused": ["focused", "concentrated", "working on"],
    }
    for mood, keywords in mood_keywords.items():
        for keyword in keywords:
            if keyword in content_lower:
                moods.append(mood)
                break

    return topics[:5], moods[:3], intensity


def build_index(
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
    max_chunks: int | None = None,
    generate_summaries: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> EpisodeIndex:
    """
    Build an episode index from memories.

    Args:
        memories: List of memories
        llm: LLM router
        model: Model for summarization
        max_chunks: Maximum chunks to process for episode detection
        generate_summaries: Whether to generate summaries for each episode
        progress_callback: Optional callback for progress updates

    Returns:
        Built EpisodeIndex
    """
    if progress_callback:
        progress_callback("Detecting episodes...")

    # Detect episodes using LLM-filtered approach
    detection_result = detect_episodes_llm_filtered(
        memories=memories,
        llm=llm,
        model=model,
        chunk_size=50,
        overlap=10,
        max_chunks=max_chunks,
    )

    if progress_callback:
        progress_callback(f"Detected {len(detection_result.episodes)} episodes")

    index = EpisodeIndex()

    for i, episode in enumerate(detection_result.episodes):
        if progress_callback:
            progress_callback(
                f"Processing episode {i + 1}/{len(detection_result.episodes)}"
            )

        # Generate title and summary
        title = None
        summary = None

        if generate_summaries:
            try:
                title = generate_episode_title(episode, memories, llm, model)
                summary = generate_episode_summary(
                    episode, memories, llm, model, style="basic"
                )
            except Exception:
                # Skip summarization on error
                pass

        # Extract topics and moods
        try:
            topics, moods, intensity = extract_topics_and_moods(
                episode, memories, llm, model
            )
        except Exception:
            topics = []
            moods = []
            intensity = 0.5

        # Create indexed episode
        indexed_episode = IndexedEpisode(
            id=episode.id,
            start_time=episode.start_time,
            end_time=episode.end_time,
            duration_minutes=episode.duration_minutes,
            memory_ids=episode.memory_ids,
            memory_count=episode.memory_count,
            title=title,
            summary=summary,
            topics=topics,
            moods=moods,
            emotional_intensity=intensity,
            key_events=[],  # Could be extracted from structured summary
        )

        index.add_episode(indexed_episode)

    if progress_callback:
        progress_callback(f"Index built with {len(index)} episodes")

    return index


def build_index_from_conversation(
    conversations_dir: Path,
    conversation_prefix: str,
    llm: LLM,
    model: SupportedModel,
    output_path: Path | None = None,
    max_chunks: int | None = None,
    generate_summaries: bool = True,
) -> EpisodeIndex:
    """
    Build an episode index from a conversation.

    Args:
        conversations_dir: Directory containing conversations
        conversation_prefix: Conversation ID/prefix
        llm: LLM router
        model: Model for summarization
        output_path: Optional path to save the index
        max_chunks: Maximum chunks to process
        generate_summaries: Whether to generate summaries

    Returns:
        Built EpisodeIndex
    """
    # Load memories
    memories = load_memories(conversations_dir, conversation_prefix)

    def progress(msg: str) -> None:
        print(msg)

    # Build index
    index = build_index(
        memories=memories,
        llm=llm,
        model=model,
        max_chunks=max_chunks,
        generate_summaries=generate_summaries,
        progress_callback=progress,
    )

    # Save if output path provided
    if output_path:
        index.save(output_path)
        print(f"Index saved to {output_path}")

    return index
