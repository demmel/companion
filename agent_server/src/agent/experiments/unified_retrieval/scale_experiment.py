"""Scale experiment for unified retrieval.

Tests the retrieval system with simulated decade-scale data to measure:
- Latency at various scales (1K, 10K, 100K, 1M memories)
- Memory usage
- Index build time
- Query throughput

Usage:
    uv run python -m agent.experiments.unified_retrieval.scale_experiment
"""

import argparse
import gc
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from agent.embedding_service import get_embedding_service

from .models import Memory, EpisodeSummary
from .unified_retriever import (
    SimpleMemoryIndex,
    SimpleEpisodeIndex,
    SimpleTopicClusters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScaleResult:
    """Results for a single scale level."""

    num_memories: int
    build_time_seconds: float
    memory_usage_mb: float
    avg_query_latency_ms: float
    p95_query_latency_ms: float
    p99_query_latency_ms: float
    queries_per_second: float


@dataclass
class ScaleExperimentResult:
    """Complete scale experiment results."""

    results: list[ScaleResult]
    summary: str


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def generate_synthetic_memories(
    num_memories: int,
    embedding_dim: int = 384,  # Default to MiniLM dimension
    seed: int = 42,
) -> list[Memory]:
    """Generate synthetic memories with random embeddings.

    Creates memories with:
    - Random content (lorem ipsum style)
    - Random embeddings
    - Timestamps spanning 10 years
    """
    np.random.seed(seed)

    # Sample topics for content generation
    topics = [
        "meeting with team",
        "project update",
        "had coffee with friend",
        "worked on presentation",
        "went to the gym",
        "read a book",
        "watched movie",
        "cooked dinner",
        "had conversation about",
        "felt happy about",
        "discussed plans for",
        "celebrated birthday",
        "started new project",
        "finished task",
        "learned about",
    ]

    entities = [
        "David",
        "Sarah",
        "Alex",
        "mom",
        "dad",
        "boss",
        "colleague",
        "friend",
        "partner",
    ]

    memories: list[Memory] = []

    # Spread timestamps over 10 years
    base_time = datetime(2015, 1, 1)
    time_span = timedelta(days=3650)  # 10 years

    for i in range(num_memories):
        # Generate random content
        topic = np.random.choice(topics)
        entity = np.random.choice(entities)
        content = f"{topic} {entity}. This is memory {i} with some additional context about life events."

        # Generate random timestamp
        random_days = np.random.uniform(0, time_span.days)
        timestamp = base_time + timedelta(days=random_days)

        # Generate random embedding (normalized)
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        memory = Memory(
            memory_id=f"mem_{i:08d}",
            content=content,
            timestamp=timestamp,
            embedding_vector=embedding.tolist(),
        )
        memories.append(memory)

    return memories


def generate_synthetic_episodes(
    memories: list[Memory],
    episodes_per_year: int = 52,  # Weekly episodes
) -> list[EpisodeSummary]:
    """Generate synthetic episodes from memories."""
    if not memories:
        return []

    # Sort memories by timestamp
    sorted_memories = sorted(memories, key=lambda m: m.timestamp)

    # Group into episodes (roughly weekly)
    episodes: list[EpisodeSummary] = []

    min_time = sorted_memories[0].timestamp
    max_time = sorted_memories[-1].timestamp
    time_span = (max_time - min_time).days

    episode_length_days = max(1, time_span // (episodes_per_year * 10))  # 10 years

    current_start = min_time
    episode_memories: list[str] = []

    for memory in sorted_memories:
        if (memory.timestamp - current_start).days > episode_length_days:
            # Create episode
            if episode_memories:
                episodes.append(
                    EpisodeSummary(
                        episode_id=str(uuid.uuid4()),
                        title=f"Episode starting {current_start.strftime('%Y-%m-%d')}",
                        summary=f"Episode with {len(episode_memories)} memories",
                        start_time=current_start,
                        end_time=memory.timestamp,
                        memory_ids=episode_memories[:50],  # Limit per episode
                    )
                )

            # Start new episode
            current_start = memory.timestamp
            episode_memories = []

        episode_memories.append(memory.memory_id)

    # Add final episode
    if episode_memories:
        episodes.append(
            EpisodeSummary(
                episode_id=str(uuid.uuid4()),
                title=f"Episode starting {current_start.strftime('%Y-%m-%d')}",
                summary=f"Episode with {len(episode_memories)} memories",
                start_time=current_start,
                end_time=max_time,
                memory_ids=episode_memories[:50],
            )
        )

    return episodes


def run_scale_test(
    num_memories: int,
    num_queries: int = 100,
    embedding_dim: int = 384,  # Default to MiniLM dimension
) -> ScaleResult:
    """Run scale test at a specific memory count."""
    print(f"\n{'='*60}")
    print(f"SCALE TEST: {num_memories:,} memories")
    print(f"{'='*60}")

    gc.collect()
    initial_memory = get_memory_usage_mb()

    # Generate synthetic data
    print("Generating synthetic memories...")
    start_time = time.time()
    memories = generate_synthetic_memories(num_memories, embedding_dim)
    generation_time = time.time() - start_time
    print(f"  Generated {len(memories):,} memories in {generation_time:.1f}s")

    # Build memory index
    print("Building memory index...")
    build_start = time.time()

    embedding_service = get_embedding_service()
    memory_index = SimpleMemoryIndex(embedding_service=embedding_service)

    for memory in memories:
        memory_index.add(memory)

    # Build episode index
    episodes = generate_synthetic_episodes(memories)
    episode_index = SimpleEpisodeIndex(embedding_service=embedding_service)
    for episode in episodes:
        episode_index.add(episode)

    # Build topic clusters
    topic_clusters = SimpleTopicClusters(embedding_service=embedding_service)
    num_clusters = min(50, num_memories // 100)
    for i, memory in enumerate(memories):
        cluster_id = f"cluster_{i % num_clusters}"
        topic_clusters.add_memory_to_cluster(
            memory.memory_id,
            cluster_id,
            f"Topic {i % num_clusters}",
        )

    build_time = time.time() - build_start
    print(f"  Built indices in {build_time:.1f}s")

    # Measure memory usage
    gc.collect()
    final_memory = get_memory_usage_mb()
    memory_usage = final_memory - initial_memory
    print(f"  Memory usage: {memory_usage:.1f}MB")

    # Run query benchmarks
    print(f"Running {num_queries} queries...")
    test_queries = [
        "What is David wearing?",
        "Tell me about Sarah",
        "What happened yesterday?",
        "How did the meeting go?",
        "What do I know about the project?",
    ] * (num_queries // 5 + 1)
    test_queries = test_queries[:num_queries]

    latencies: list[float] = []

    for query in test_queries:
        start = time.time()
        memory_index.search(query, top_k=10)
        latencies.append((time.time() - start) * 1000)

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))
    qps = 1000 / avg_latency if avg_latency > 0 else 0

    print(f"  Avg latency: {avg_latency:.2f}ms")
    print(f"  P95 latency: {p95_latency:.2f}ms")
    print(f"  P99 latency: {p99_latency:.2f}ms")
    print(f"  Throughput: {qps:.1f} queries/sec")

    return ScaleResult(
        num_memories=num_memories,
        build_time_seconds=build_time,
        memory_usage_mb=memory_usage,
        avg_query_latency_ms=avg_latency,
        p95_query_latency_ms=p95_latency,
        p99_query_latency_ms=p99_latency,
        queries_per_second=qps,
    )


def run_scale_experiment(
    scale_levels: list[int] | None = None,
    num_queries: int = 100,
) -> ScaleExperimentResult:
    """Run scale experiment at multiple levels."""
    if scale_levels is None:
        scale_levels = [1000, 5000, 10000]  # Start conservative

    results: list[ScaleResult] = []

    for num_memories in scale_levels:
        try:
            result = run_scale_test(num_memories, num_queries)
            results.append(result)
        except MemoryError:
            print(f"MemoryError at {num_memories:,} memories - stopping")
            break
        except Exception as e:
            print(f"Error at {num_memories:,} memories: {e}")
            break

        # Force garbage collection between tests
        gc.collect()

    # Generate summary
    summary_lines = [
        "",
        "=" * 70,
        "SCALE EXPERIMENT SUMMARY",
        "=" * 70,
        "",
        f"{'Memories':>12} {'Build(s)':>10} {'Memory(MB)':>12} {'Avg(ms)':>10} {'P95(ms)':>10} {'QPS':>10}",
        "-" * 70,
    ]

    for r in results:
        summary_lines.append(
            f"{r.num_memories:>12,} {r.build_time_seconds:>10.1f} {r.memory_usage_mb:>12.1f} "
            f"{r.avg_query_latency_ms:>10.2f} {r.p95_query_latency_ms:>10.2f} {r.queries_per_second:>10.1f}"
        )

    summary_lines.extend([
        "",
        "=" * 70,
        "FINDINGS",
        "=" * 70,
    ])

    if results:
        # Check if latency stays under target
        max_p95 = max(r.p95_query_latency_ms for r in results)
        if max_p95 < 200:
            summary_lines.append(f"  Latency target met at all scales (max P95: {max_p95:.1f}ms)")
        else:
            summary_lines.append(f"  Latency exceeds target at some scales (max P95: {max_p95:.1f}ms)")

        # Memory scaling
        if len(results) >= 2:
            mem_ratio = results[-1].memory_usage_mb / results[0].memory_usage_mb
            scale_ratio = results[-1].num_memories / results[0].num_memories
            summary_lines.append(f"  Memory scales {mem_ratio:.1f}x for {scale_ratio:.0f}x more data")

    summary = "\n".join(summary_lines)
    print(summary)

    return ScaleExperimentResult(results=results, summary=summary)


def save_results(result: ScaleExperimentResult, output_file: Path) -> None:
    """Save results to JSON."""
    data = {
        "results": [asdict(r) for r in result.results],
        "summary": result.summary,
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run scale experiment for unified retrieval"
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="1000,5000,10000",
        help="Comma-separated list of scale levels",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of queries to run per scale level",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results JSON",
    )

    args = parser.parse_args()

    scale_levels = [int(x.strip()) for x in args.scales.split(",")]

    result = run_scale_experiment(
        scale_levels=scale_levels,
        num_queries=args.queries,
    )

    if args.output:
        save_results(result, Path(args.output))
    else:
        output_file = Path(__file__).parent / "output" / "cache" / "scale_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_results(result, output_file)


if __name__ == "__main__":
    main()
