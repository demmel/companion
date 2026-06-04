"""
Retrieval strategies for temporal queries.

Implements 4 strategies:
A. Episode Summary Only
B. Episode → Memories
C. Hybrid (summaries + top-K memories)
D. Direct Memory Search (Baseline)
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime

from agent.experiments.episode_summaries.detection import cosine_similarity
from agent.experiments.temporal_retrieval.episode_index import EpisodeIndex
from agent.experiments.temporal_retrieval.models import (
    IndexedEpisode,
    RetrievalResult,
    TimeReference,
)
from agent.experiments.temporal_retrieval.time_parser import parse_time_reference
from agent.llm.models import SupportedModel
from agent.llm.router import LLM
from agent.memory.dag.models import MemoryElement


class RetrievalStrategy(ABC):
    """Base class for retrieval strategies."""

    name: str

    @abstractmethod
    def retrieve(
        self,
        query: str,
        time_ref: TimeReference,
        index: EpisodeIndex,
        memories: list[MemoryElement],
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve relevant episodes/memories for a query.

        Args:
            query: The query text
            time_ref: Parsed time reference
            index: Episode index
            memories: All memories
            top_k: Maximum number of results

        Returns:
            RetrievalResult with matched episodes
        """
        pass


class StrategyA(RetrievalStrategy):
    """
    Strategy A: Episode Summary Only

    Parse time → Find episodes → Return summaries
    Fast and compressed, but may lose detail.
    """

    name = "episode_summary_only"

    def retrieve(
        self,
        query: str,
        time_ref: TimeReference,
        index: EpisodeIndex,
        memories: list[MemoryElement],
        top_k: int = 5,
    ) -> RetrievalResult:
        start_time = time.time()

        # Query index
        episodes = index.query(time_ref)

        # Limit to top_k
        episodes = episodes[:top_k]

        # Get summaries
        summaries = []
        episode_ids = []
        for ep in episodes:
            episode_ids.append(ep.id)
            if ep.summary:
                summaries.append(ep.summary)
            elif ep.title:
                summaries.append(f"Episode: {ep.title}")
            else:
                summaries.append(
                    f"Episode from {ep.start_time.strftime('%Y-%m-%d %H:%M')}"
                )

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query_text=query,
            time_ref=time_ref,
            retrieved_episode_ids=episode_ids,
            retrieved_summaries=summaries,
            strategy=self.name,
            latency_ms=latency_ms,
        )


class StrategyB(RetrievalStrategy):
    """
    Strategy B: Episode → Memories

    Parse time → Find episodes → Return raw memories
    Full detail, but more tokens and slower.
    """

    name = "episode_to_memories"

    def retrieve(
        self,
        query: str,
        time_ref: TimeReference,
        index: EpisodeIndex,
        memories: list[MemoryElement],
        top_k: int = 5,
    ) -> RetrievalResult:
        start_time = time.time()

        # Query index
        episodes = index.query(time_ref)

        # Limit to top_k episodes
        episodes = episodes[:top_k]

        # Get all memories from these episodes
        memory_by_id = {m.id: m for m in memories}
        episode_ids = []
        summaries = []

        for ep in episodes:
            episode_ids.append(ep.id)

            # Get episode memories
            ep_memories = []
            for mem_id in ep.memory_ids:
                mem = memory_by_id.get(mem_id)
                if mem:
                    ep_memories.append(mem)

            # Sort by timestamp
            ep_memories.sort(key=lambda m: m.timestamp)

            # Format as content
            memory_text = "\n".join(
                f"[{m.timestamp.strftime('%H:%M')}] {m.content[:200]}"
                for m in ep_memories[:10]  # Limit memories per episode
            )
            summaries.append(memory_text)

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query_text=query,
            time_ref=time_ref,
            retrieved_episode_ids=episode_ids,
            retrieved_summaries=summaries,
            strategy=self.name,
            latency_ms=latency_ms,
        )


class StrategyC(RetrievalStrategy):
    """
    Strategy C: Hybrid

    Parse time → Find episodes → Return summaries + top-K relevant memories
    Balance of context and detail.
    """

    name = "hybrid"

    def __init__(self, memories_per_episode: int = 3):
        self.memories_per_episode = memories_per_episode

    def retrieve(
        self,
        query: str,
        time_ref: TimeReference,
        index: EpisodeIndex,
        memories: list[MemoryElement],
        top_k: int = 5,
    ) -> RetrievalResult:
        start_time = time.time()

        # Query index
        episodes = index.query(time_ref)

        # Limit to top_k episodes
        episodes = episodes[:top_k]

        # Get query embedding for similarity search within episodes
        query_embedding: list[float] | None = None

        # Try to find a similar memory to use its embedding pattern
        # (In production, we'd embed the query directly)
        for mem in memories[:100]:
            if mem.embedding_vector:
                query_embedding = mem.embedding_vector
                break

        memory_by_id = {m.id: m for m in memories}
        episode_ids = []
        summaries = []

        for ep in episodes:
            episode_ids.append(ep.id)

            # Start with summary
            parts = []
            if ep.summary:
                parts.append(f"Summary: {ep.summary}")
            elif ep.title:
                parts.append(f"Episode: {ep.title}")

            # Get episode memories
            ep_memories = []
            for mem_id in ep.memory_ids:
                mem = memory_by_id.get(mem_id)
                if mem:
                    ep_memories.append(mem)

            # Find most relevant memories using similarity
            if query_embedding and ep_memories:
                scored_memories: list[tuple[float, MemoryElement]] = []
                for mem in ep_memories:
                    if mem.embedding_vector:
                        sim = cosine_similarity(query_embedding, mem.embedding_vector)
                        scored_memories.append((sim, mem))
                    else:
                        scored_memories.append((0.0, mem))

                # Sort by similarity
                scored_memories.sort(key=lambda x: x[0], reverse=True)
                top_memories = [m for _, m in scored_memories[: self.memories_per_episode]]
            else:
                # Fallback: take first N memories
                ep_memories.sort(key=lambda m: m.timestamp)
                top_memories = ep_memories[: self.memories_per_episode]

            # Add top memories
            if top_memories:
                parts.append("\nKey memories:")
                for mem in top_memories:
                    parts.append(
                        f"  [{mem.timestamp.strftime('%H:%M')}] {mem.content[:150]}"
                    )

            summaries.append("\n".join(parts))

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query_text=query,
            time_ref=time_ref,
            retrieved_episode_ids=episode_ids,
            retrieved_summaries=summaries,
            strategy=self.name,
            latency_ms=latency_ms,
        )


class StrategyD(RetrievalStrategy):
    """
    Strategy D: Direct Memory Search (Baseline)

    Filter memories by time range → Similarity search
    Simple but doesn't use episode structure.
    """

    name = "direct_memory_search"

    def retrieve(
        self,
        query: str,
        time_ref: TimeReference,
        index: EpisodeIndex,
        memories: list[MemoryElement],
        top_k: int = 5,
    ) -> RetrievalResult:
        start_time = time.time()

        # Filter memories by time range
        filtered_memories = []
        episode_ids: list[str] = []

        if time_ref.start_time and time_ref.end_time:
            for mem in memories:
                if time_ref.start_time <= mem.timestamp <= time_ref.end_time:
                    filtered_memories.append(mem)
            # Query the index by time range to get overlapping episodes
            episodes = index.query_by_time_range(time_ref.start_time, time_ref.end_time)
            episode_ids = [ep.id for ep in episodes[:top_k]]
        else:
            # For emotional/topic queries, search all memories
            filtered_memories = memories
            # Use index.query() to find relevant episodes
            episodes = index.query(time_ref)
            episode_ids = [ep.id for ep in episodes[:top_k]]

        # Get query embedding
        query_embedding: list[float] | None = None
        for mem in memories[:100]:
            if mem.embedding_vector:
                query_embedding = mem.embedding_vector
                break

        # Score and rank memories
        if query_embedding:
            scored_memories: list[tuple[float, MemoryElement]] = []
            for mem in filtered_memories:
                if mem.embedding_vector:
                    sim = cosine_similarity(query_embedding, mem.embedding_vector)
                    scored_memories.append((sim, mem))
                else:
                    scored_memories.append((0.0, mem))

            # Sort by similarity
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            top_memories = [m for _, m in scored_memories[:top_k]]
        else:
            # Fallback: return most recent
            filtered_memories.sort(key=lambda m: m.timestamp, reverse=True)
            top_memories = filtered_memories[:top_k]

        # Format results
        summaries = []
        for mem in top_memories:
            summaries.append(
                f"[{mem.timestamp.strftime('%Y-%m-%d %H:%M')}] {mem.content[:300]}"
            )

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query_text=query,
            time_ref=time_ref,
            retrieved_episode_ids=episode_ids,
            retrieved_summaries=summaries,
            strategy=self.name,
            latency_ms=latency_ms,
        )


# Registry of all strategies
STRATEGIES: dict[str, RetrievalStrategy] = {
    "A": StrategyA(),
    "B": StrategyB(),
    "C": StrategyC(),
    "D": StrategyD(),
}


def retrieve_with_strategy(
    query: str,
    strategy_name: str,
    index: EpisodeIndex,
    memories: list[MemoryElement],
    now: datetime | None = None,
    top_k: int = 5,
    llm: LLM | None = None,
    model: SupportedModel | None = None,
) -> RetrievalResult:
    """
    Retrieve results using a named strategy.

    Args:
        query: Query text
        strategy_name: Strategy name (A, B, C, or D)
        index: Episode index
        memories: All memories
        now: Current time for parsing
        top_k: Maximum results
        llm: LLM instance for time parsing (optional)
        model: Model for time parsing (optional)

    Returns:
        RetrievalResult
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Parse time reference (uses LLM if provided, otherwise heuristics)
    time_ref = parse_time_reference(query, now, llm, model)
    if not time_ref:
        # Default to recent if no time reference found
        if now is None:
            now = datetime.now()
        time_ref = TimeReference(
            raw_text="",
            ref_type="relative",
            start_time=now,
            end_time=now,
        )

    strategy = STRATEGIES[strategy_name]
    return strategy.retrieve(query, time_ref, index, memories, top_k)


def compare_strategies(
    query: str,
    index: EpisodeIndex,
    memories: list[MemoryElement],
    now: datetime | None = None,
    top_k: int = 5,
    llm: LLM | None = None,
    model: SupportedModel | None = None,
) -> dict[str, RetrievalResult]:
    """
    Run all strategies on a query and compare results.

    Args:
        query: Query text
        index: Episode index
        memories: All memories
        now: Current time for parsing
        top_k: Maximum results
        llm: LLM instance for time parsing (optional)
        model: Model for time parsing (optional)

    Returns:
        Dict mapping strategy name to result
    """
    results = {}
    for name in STRATEGIES:
        results[name] = retrieve_with_strategy(
            query=query,
            strategy_name=name,
            index=index,
            memories=memories,
            now=now,
            top_k=top_k,
            llm=llm,
            model=model,
        )
    return results
