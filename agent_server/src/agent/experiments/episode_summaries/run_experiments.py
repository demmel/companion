"""
Episode Summaries Experiment Runner

CLI for running episode detection and summarization experiments.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click

from agent.conversation_persistence import ConversationPersistence
from agent.memory.dag.dag_memory_manager import DagMemoryManager
from agent.experiments.episode_summaries.detection import (
    detect_episodes_by_gap,
    analyze_gap_distribution,
    run_gap_threshold_sweep,
    analyze_similarity_distribution,
    detect_episodes_by_topic,
    run_topic_threshold_sweep,
    detect_episodes_windowed,
    run_windowed_sweep,
    detect_episodes_llm,
    detect_episodes_llm_filtered,
    detect_episodes_llm_chunk,
    detect_episodes_llm_json,
    detect_episodes_llm_chunk_json,
    EpisodeBoundary,
)
from agent.experiments.episode_summaries.summarization import (
    generate_episode_summary,
    generate_episode_title,
    generate_summary_at_detail_level,
    SUMMARY_STYLES,
)
from agent.llm.router import create_llm
from agent.llm.models import SupportedModel


# Default configuration
DEFAULT_CONVERSATION_PREFIX = "conversation_20251024_083630_306692"
DEFAULT_CONVERSATIONS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "conversations"
)
RESULTS_DIR = Path(__file__).parent / "results"


def load_memories(
    conversations_dir: Path,
    conversation_prefix: str,
):
    """Load memories from a conversation file."""
    persistence = ConversationPersistence(str(conversations_dir))
    agent_data = persistence.load_agent_data(
        conversation_prefix,
        use_individual_formatting=True,
    )
    if not isinstance(agent_data.memory, DagMemoryManager):
        raise ValueError("Episode summaries experiment requires DAG memory type")
    memory_graph = agent_data.memory.get_memory_graph()
    memories = list(memory_graph.elements.values())
    return memories


def save_results(filename: str, data: dict) -> Path:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    filepath = RESULTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return filepath


@click.group()
def cli():
    """Episode Summaries Experiment CLI"""
    pass


@cli.command()
@click.option(
    "--conversation",
    default=DEFAULT_CONVERSATION_PREFIX,
    help="Conversation prefix to load",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Directory containing conversation files",
)
@click.option(
    "--thresholds",
    default="15,30,60,120,240",
    help="Comma-separated gap thresholds in minutes",
)
def experiment1(conversation: str, conversations_dir: str, thresholds: str):
    """
    Experiment 1: Gap Threshold Sweep

    Tests multiple gap thresholds to find optimal episode boundaries.
    """
    click.echo(f"Loading memories from {conversation}...")
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    # Parse thresholds
    threshold_list = [int(t.strip()) for t in thresholds.split(",")]

    # Analyze gap distribution first
    click.echo("\n--- Gap Distribution Analysis ---")
    gap_stats = analyze_gap_distribution(memories)
    click.echo(f"Total memories: {gap_stats['count']}")
    click.echo(f"Total gaps: {gap_stats['total_gaps']}")
    click.echo(f"Min gap: {gap_stats['min_gap_minutes']:.2f} minutes")
    click.echo(f"Max gap: {gap_stats['max_gap_minutes']:.2f} minutes")
    click.echo(f"Avg gap: {gap_stats['avg_gap_minutes']:.2f} minutes")
    click.echo(f"Median gap: {gap_stats['median_gap_minutes']:.2f} minutes")
    click.echo(f"Large gaps (>30 min): {len(gap_stats['large_gaps'])}")

    # Run threshold sweep
    click.echo("\n--- Threshold Sweep Results ---")
    sweep_result = run_gap_threshold_sweep(memories, threshold_list)

    for stats in sweep_result.thresholds:
        click.echo(f"\nThreshold: {stats.gap_minutes} minutes")
        click.echo(f"  Episodes: {stats.episode_count}")
        click.echo(
            f"  Sizes: min={stats.sizes['min']:.0f}, "
            f"max={stats.sizes['max']:.0f}, "
            f"avg={stats.sizes['avg']:.1f}"
        )
        click.echo(
            f"  Durations: min={stats.durations['min_minutes']:.1f}min, "
            f"max={stats.durations['max_minutes']:.1f}min, "
            f"avg={stats.durations['avg_minutes']:.1f}min"
        )

    click.echo(f"\n--- Recommendation ---")
    click.echo(sweep_result.recommendation)

    # Save results
    results = {
        "experiment": "gap_threshold_sweep",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "gap_distribution": gap_stats,
        "threshold_sweep": {
            "thresholds": [asdict(s) for s in sweep_result.thresholds],
            "total_memories": sweep_result.total_memories,
            "recommendation": sweep_result.recommendation,
        },
    }
    filepath = save_results("experiment1_gap_sweep.json", results)
    click.echo(f"\nResults saved to: {filepath}")


@cli.command()
@click.option(
    "--conversation",
    default=DEFAULT_CONVERSATION_PREFIX,
    help="Conversation prefix to load",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Directory containing conversation files",
)
@click.option("--gap-threshold", default=30, help="Gap threshold in minutes")
@click.option("--num-episodes", default=5, help="Number of episodes to summarize")
@click.option(
    "--model",
    default="claude-sonnet-4-5-20250929",
    help="Model to use for summarization",
)
def experiment3(
    conversation: str,
    conversations_dir: str,
    gap_threshold: int,
    num_episodes: int,
    model: str,
):
    """
    Experiment 3: Summary Approach Comparison

    Generates summaries using different styles and compares results.
    """
    click.echo(f"Loading memories from {conversation}...")
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    # Detect episodes
    click.echo(f"\nDetecting episodes with {gap_threshold}-minute threshold...")
    detection_result = detect_episodes_by_gap(memories, gap_threshold)
    click.echo(f"Found {len(detection_result.episodes)} episodes")

    if not detection_result.episodes:
        click.echo("No episodes found. Try a smaller gap threshold.")
        return

    # Select episodes to summarize (diverse selection)
    episodes = detection_result.episodes[:num_episodes]
    click.echo(f"\nSummarizing {len(episodes)} episodes...")

    # Initialize LLM
    llm = create_llm()
    model_enum = SupportedModel(model)

    results = {
        "experiment": "summary_approach_comparison",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "gap_threshold": gap_threshold,
        "model": model,
        "episodes": [],
    }

    styles = list(SUMMARY_STYLES.keys())

    for i, episode in enumerate(episodes):
        click.echo(f"\n--- Episode {i+1}: {episode.start_time} ---")
        click.echo(
            f"Duration: {episode.duration_minutes:.1f} min, Memories: {episode.memory_count}"
        )

        # Generate title
        click.echo("Generating title...")
        title = generate_episode_title(episode, memories, llm, model_enum)
        click.echo(f"Title: {title}")

        episode_result = {
            "episode_id": episode.id,
            "start_time": episode.start_time.isoformat(),
            "end_time": episode.end_time.isoformat(),
            "duration_minutes": episode.duration_minutes,
            "memory_count": episode.memory_count,
            "title": title,
            "summaries": {},
        }

        # Generate summaries in each style
        for style in styles:
            click.echo(f"Generating {style} summary...")
            summary = generate_episode_summary(
                episode, memories, llm, model_enum, style
            )
            episode_result["summaries"][style] = summary
            click.echo(f"\n{style.upper()} SUMMARY:")
            click.echo(summary[:500] + "..." if len(summary) > 500 else summary)

        results["episodes"].append(episode_result)

    # Save results
    filepath = save_results("experiment3_summaries.json", results)
    click.echo(f"\n\nResults saved to: {filepath}")


@cli.command()
@click.option(
    "--conversation",
    default=DEFAULT_CONVERSATION_PREFIX,
    help="Conversation prefix to load",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Directory containing conversation files",
)
@click.option("--gap-threshold", default=30, help="Gap threshold in minutes")
@click.option("--num-episodes", default=3, help="Number of episodes to analyze")
@click.option(
    "--model",
    default="claude-sonnet-4-5-20250929",
    help="Model to use for summarization",
)
def experiment4(
    conversation: str,
    conversations_dir: str,
    gap_threshold: int,
    num_episodes: int,
    model: str,
):
    """
    Experiment 4: Summary Detail Level

    Compares short, medium, and detailed summaries for compression analysis.
    """
    click.echo(f"Loading memories from {conversation}...")
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    # Detect episodes
    click.echo(f"\nDetecting episodes with {gap_threshold}-minute threshold...")
    detection_result = detect_episodes_by_gap(memories, gap_threshold)
    click.echo(f"Found {len(detection_result.episodes)} episodes")

    if not detection_result.episodes:
        click.echo("No episodes found. Try a smaller gap threshold.")
        return

    # Select episodes
    episodes = detection_result.episodes[:num_episodes]
    click.echo(f"\nAnalyzing {len(episodes)} episodes...")

    # Initialize LLM
    llm = create_llm()
    model_enum = SupportedModel(model)

    results = {
        "experiment": "summary_detail_level",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "gap_threshold": gap_threshold,
        "model": model,
        "episodes": [],
    }

    detail_levels = ["short", "medium", "detailed"]

    for i, episode in enumerate(episodes):
        click.echo(f"\n--- Episode {i+1}: {episode.start_time} ---")
        click.echo(
            f"Duration: {episode.duration_minutes:.1f} min, Memories: {episode.memory_count}"
        )

        # Calculate raw content size
        episode_memory_ids = set(episode.memory_ids)
        episode_memories = [m for m in memories if m.id in episode_memory_ids]
        raw_content = "\n".join(m.content for m in episode_memories)
        raw_tokens = len(raw_content) / 3.4  # Approximate token count

        episode_result = {
            "episode_id": episode.id,
            "start_time": episode.start_time.isoformat(),
            "end_time": episode.end_time.isoformat(),
            "duration_minutes": episode.duration_minutes,
            "memory_count": episode.memory_count,
            "raw_content_chars": len(raw_content),
            "raw_tokens_approx": raw_tokens,
            "summaries": {},
        }

        # Generate summaries at each detail level
        for level in detail_levels:
            click.echo(f"Generating {level} summary...")
            summary = generate_summary_at_detail_level(
                episode, memories, llm, model_enum, level
            )
            summary_tokens = len(summary) / 3.4

            compression_ratio = raw_tokens / summary_tokens if summary_tokens > 0 else 0

            episode_result["summaries"][level] = {
                "text": summary,
                "chars": len(summary),
                "tokens_approx": summary_tokens,
                "compression_ratio": compression_ratio,
            }

            click.echo(
                f"\n{level.upper()} ({summary_tokens:.0f} tokens, {compression_ratio:.1f}x compression):"
            )
            click.echo(summary[:300] + "..." if len(summary) > 300 else summary)

        results["episodes"].append(episode_result)

    # Calculate aggregate statistics
    click.echo("\n\n--- Aggregate Statistics ---")
    for level in detail_levels:
        compressions = [
            ep["summaries"][level]["compression_ratio"] for ep in results["episodes"]
        ]
        avg_compression = sum(compressions) / len(compressions)
        click.echo(f"{level}: avg compression {avg_compression:.1f}x")

    # Save results
    filepath = save_results("experiment4_detail_levels.json", results)
    click.echo(f"\nResults saved to: {filepath}")


@cli.command()
@click.option(
    "--conversation",
    default=DEFAULT_CONVERSATION_PREFIX,
    help="Conversation prefix to load",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Directory containing conversation files",
)
@click.option("--gap-threshold", default=30, help="Gap threshold in minutes")
def show_episodes(conversation: str, conversations_dir: str, gap_threshold: int):
    """
    Show detected episodes without generating summaries.
    """
    click.echo(f"Loading memories from {conversation}...")
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    # Detect episodes
    click.echo(f"\nDetecting episodes with {gap_threshold}-minute threshold...")
    detection_result = detect_episodes_by_gap(memories, gap_threshold)
    click.echo(f"Found {len(detection_result.episodes)} episodes\n")

    for i, episode in enumerate(detection_result.episodes):
        click.echo(f"Episode {i+1}:")
        click.echo(f"  Start: {episode.start_time}")
        click.echo(f"  End: {episode.end_time}")
        click.echo(f"  Duration: {episode.duration_minutes:.1f} minutes")
        click.echo(f"  Memories: {episode.memory_count}")

        # Show first and last memory content
        episode_memory_ids = set(episode.memory_ids)
        episode_memories = sorted(
            [m for m in memories if m.id in episode_memory_ids],
            key=lambda m: m.timestamp,
        )
        if episode_memories:
            first = episode_memories[0]
            last = episode_memories[-1]
            click.echo(f"  First: {first.content[:80]}...")
            click.echo(f"  Last: {last.content[:80]}...")
        click.echo()


@cli.command()
@click.option(
    "--conversation",
    default=DEFAULT_CONVERSATION_PREFIX,
    help="Conversation prefix to load",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Directory containing conversation files",
)
@click.option(
    "--thresholds",
    default="0.5,0.6,0.7,0.8",
    help="Comma-separated similarity thresholds to test",
)
def experiment_topic(conversation: str, conversations_dir: str, thresholds: str):
    """
    Experiment: Topic-Based Detection

    Detects episode boundaries using embedding similarity instead of time gaps.
    Tests multiple similarity thresholds to find optimal value.
    """
    click.echo(f"Loading memories from {conversation}...")
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    # Check for embeddings
    with_embeddings = sum(1 for m in memories if m.embedding_vector)
    click.echo(f"Memories with embeddings: {with_embeddings}/{len(memories)}")

    if with_embeddings < len(memories) * 0.5:
        click.echo("WARNING: Less than 50% of memories have embeddings!")

    # Analyze similarity distribution
    click.echo("\n--- Similarity Distribution Analysis ---")
    sim_stats = analyze_similarity_distribution(memories)
    click.echo(f"Min similarity: {sim_stats.min_similarity:.3f}")
    click.echo(f"Max similarity: {sim_stats.max_similarity:.3f}")
    click.echo(f"Avg similarity: {sim_stats.avg_similarity:.3f}")
    click.echo(f"Median similarity: {sim_stats.median_similarity:.3f}")
    click.echo(f"Std deviation: {sim_stats.std_similarity:.3f}")
    click.echo("\nLow similarity counts:")
    for threshold, count in sim_stats.low_similarity_count.items():
        click.echo(f"  {threshold}: {count}")

    # Parse thresholds
    threshold_list = [float(t.strip()) for t in thresholds.split(",")]

    # Run threshold sweep
    click.echo("\n--- Topic Threshold Sweep Results ---")
    sweep_result = run_topic_threshold_sweep(memories, threshold_list)

    for result in sweep_result["thresholds"]:
        click.echo(f"\nSimilarity threshold: {result['similarity_threshold']}")
        click.echo(f"  Episodes: {result['episode_count']}")
        click.echo(f"  Topic shifts: {result['topic_shifts_count']}")
        click.echo(
            f"  Sizes: min={result['sizes']['min']:.0f}, "
            f"max={result['sizes']['max']:.0f}, "
            f"avg={result['sizes']['avg']:.1f}"
        )
        click.echo(
            f"  Durations: min={result['durations']['min_minutes']:.1f}min, "
            f"max={result['durations']['max_minutes']:.1f}min, "
            f"avg={result['durations']['avg_minutes']:.1f}min"
        )

    # Save results
    results = {
        "experiment": "topic_detection",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "similarity_distribution": {
            "min": sim_stats.min_similarity,
            "max": sim_stats.max_similarity,
            "avg": sim_stats.avg_similarity,
            "median": sim_stats.median_similarity,
            "std": sim_stats.std_similarity,
            "low_counts": sim_stats.low_similarity_count,
        },
        "threshold_sweep": sweep_result,
    }
    filepath = save_results("experiment_topic_detection.json", results)
    click.echo(f"\nResults saved to: {filepath}")


@cli.command()
@click.option(
    "--conversation",
    default=DEFAULT_CONVERSATION_PREFIX,
    help="Conversation prefix to load",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Directory containing conversation files",
)
@click.option("--similarity-threshold", default=0.7, help="Similarity threshold")
@click.option("--num-episodes", default=5, help="Number of episodes to show")
def show_topic_episodes(
    conversation: str,
    conversations_dir: str,
    similarity_threshold: float,
    num_episodes: int,
):
    """
    Show episodes detected by topic shift (embedding similarity).
    """
    click.echo(f"Loading memories from {conversation}...")
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    # Detect episodes
    click.echo(
        f"\nDetecting episodes with similarity threshold {similarity_threshold}..."
    )
    detection_result = detect_episodes_by_topic(memories, similarity_threshold)
    click.echo(f"Found {len(detection_result.episodes)} episodes")
    click.echo(f"Topic shifts detected: {len(detection_result.topic_shifts)}")

    # Show episodes
    for i, episode in enumerate(detection_result.episodes[:num_episodes]):
        click.echo(f"\n--- Episode {i+1} ---")
        click.echo(f"Start: {episode.start_time}")
        click.echo(f"End: {episode.end_time}")
        click.echo(f"Duration: {episode.duration_minutes:.1f} minutes")
        click.echo(f"Memories: {episode.memory_count}")

        # Show first and last memory
        episode_memory_ids = set(episode.memory_ids)
        episode_memories = sorted(
            [m for m in memories if m.id in episode_memory_ids],
            key=lambda m: m.timestamp,
        )
        if episode_memories:
            first = episode_memories[0]
            last = episode_memories[-1]
            click.echo(f"First: {first.content[:100]}...")
            if len(episode_memories) > 1:
                click.echo(f"Last: {last.content[:100]}...")


@cli.command()
@click.option(
    "--conversation",
    default=DEFAULT_CONVERSATION_PREFIX,
    help="Conversation prefix to load",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Directory containing conversation files",
)
@click.option(
    "--window-sizes",
    default="3,5,10",
    help="Comma-separated window sizes to test",
)
@click.option(
    "--thresholds",
    default="0.2,0.3,0.4",
    help="Comma-separated similarity thresholds to test",
)
@click.option(
    "--min-episode-size",
    default=3,
    help="Minimum episode size (smaller merged into neighbors)",
)
def experiment_windowed(
    conversation: str,
    conversations_dir: str,
    window_sizes: str,
    thresholds: str,
    min_episode_size: int,
):
    """
    Experiment: Windowed Topic Detection

    Detects episodes by comparing each memory to the centroid of the last N memories.
    This reduces fragmentation compared to pairwise comparison.
    """
    click.echo(f"Loading memories from {conversation}...")
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    # Parse parameters
    window_size_list = [int(w.strip()) for w in window_sizes.split(",")]
    threshold_list = [float(t.strip()) for t in thresholds.split(",")]

    click.echo(f"\nWindow sizes: {window_size_list}")
    click.echo(f"Thresholds: {threshold_list}")
    click.echo(f"Min episode size: {min_episode_size}")

    # Run sweep
    click.echo("\n--- Windowed Detection Sweep ---")
    sweep_result = run_windowed_sweep(
        memories,
        window_sizes=window_size_list,
        thresholds=threshold_list,
        min_episode_size=min_episode_size,
    )

    # Display results
    for result in sweep_result["results"]:
        click.echo(
            f"\nWindow={result['window_size']}, Threshold={result['similarity_threshold']}"
        )
        click.echo(f"  Episodes: {result['episode_count']}")
        click.echo(f"  Topic shifts: {result['topic_shifts_count']}")
        click.echo(
            f"  Sizes: min={result['sizes']['min']:.0f}, "
            f"max={result['sizes']['max']:.0f}, "
            f"avg={result['sizes']['avg']:.1f}"
        )
        click.echo(
            f"  Durations: min={result['durations']['min_minutes']:.1f}min, "
            f"max={result['durations']['max_minutes']:.1f}min, "
            f"avg={result['durations']['avg_minutes']:.1f}min"
        )

    # Comparison summary
    click.echo("\n--- Comparison Summary ---")
    click.echo("Phase 1 (time-based, 30min): 64 episodes, avg 104 memories")
    click.echo("Phase 2 (topic, 0.5 threshold): 4532 episodes, avg 1.5 memories")
    click.echo("Phase 3 (windowed):")
    for result in sweep_result["results"]:
        if result["episode_count"] > 0:
            click.echo(
                f"  w={result['window_size']}, t={result['similarity_threshold']}: "
                f"{result['episode_count']} episodes, avg {result['sizes']['avg']:.1f} memories"
            )

    # Save results
    results = {
        "experiment": "windowed_detection",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "parameters": {
            "window_sizes": window_size_list,
            "thresholds": threshold_list,
            "min_episode_size": min_episode_size,
        },
        "sweep_results": sweep_result,
    }
    filepath = save_results("experiment_windowed_detection.json", results)
    click.echo(f"\nResults saved to: {filepath}")


@cli.command()
@click.option(
    "--conversation",
    default=DEFAULT_CONVERSATION_PREFIX,
    help="Conversation prefix to load",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Directory containing conversation files",
)
@click.option(
    "--chunk-size",
    default=50,
    help="Number of memories per chunk",
)
@click.option(
    "--max-chunks",
    default=3,
    help="Maximum chunks to process (for testing)",
)
@click.option(
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model to use for boundary detection",
)
def experiment_llm(
    conversation: str,
    conversations_dir: str,
    chunk_size: int,
    max_chunks: int,
    model: str,
):
    """
    Experiment: LLM-Based Boundary Detection (Approach D)

    Uses LLM to identify natural conversation breaks by understanding
    context and meaning, rather than relying on embeddings or time gaps.
    """
    click.echo(f"Loading memories from {conversation}...")
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    click.echo(f"\nChunk size: {chunk_size}")
    click.echo(f"Max chunks: {max_chunks}")
    click.echo(f"Model: {model}")

    # Calculate what portion we're processing
    total_chunks = (len(memories) + chunk_size - 1) // chunk_size
    memories_to_process = min(max_chunks * chunk_size, len(memories))
    click.echo(
        f"Processing ~{memories_to_process} memories ({max_chunks}/{total_chunks} chunks)"
    )

    # Initialize LLM
    llm = create_llm()
    model_enum = SupportedModel(model)

    # Run detection
    click.echo("\n--- Running LLM Boundary Detection ---")
    result = detect_episodes_llm(
        memories,
        llm=llm,
        model=model_enum,
        chunk_size=chunk_size,
        overlap=10,
        max_chunks=max_chunks,
    )

    click.echo(f"\nEpisodes detected: {len(result.episodes)}")
    click.echo(f"Boundaries found: {len(result.topic_shifts) + 1}")

    # Show episode statistics
    if result.episodes:
        sizes = [ep.memory_count for ep in result.episodes]
        durations = [ep.duration_minutes for ep in result.episodes]
        click.echo(
            f"Episode sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}"
        )
        click.echo(
            f"Episode durations: min={min(durations):.1f}min, max={max(durations):.1f}min, avg={sum(durations)/len(durations):.1f}min"
        )

    # Show detected boundaries with context
    click.echo("\n--- Detected Boundaries ---")
    sorted_memories = sorted(memories, key=lambda m: m.timestamp)
    memory_by_id = {m.id: m for m in memories}

    for i, shift in enumerate(result.topic_shifts[:10]):  # Show first 10
        before = memory_by_id.get(shift.before_memory_id)
        after = memory_by_id.get(shift.after_memory_id)
        if before and after:
            click.echo(f"\nBoundary {i+1} (gap: {shift.time_gap_minutes:.1f} min):")
            before_text = (
                before.content[:100].encode("ascii", errors="replace").decode()
            )
            after_text = after.content[:100].encode("ascii", errors="replace").decode()
            click.echo(f"  BEFORE: {before_text}...")
            click.echo(f"  AFTER:  {after_text}...")

    # Save results
    results = {
        "experiment": "llm_boundary_detection",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "parameters": {
            "chunk_size": chunk_size,
            "max_chunks": max_chunks,
            "model": model,
        },
        "results": {
            "episode_count": len(result.episodes),
            "boundary_count": len(result.topic_shifts),
            "episodes": [
                {
                    "id": ep.id,
                    "start_time": ep.start_time.isoformat(),
                    "end_time": ep.end_time.isoformat(),
                    "duration_minutes": ep.duration_minutes,
                    "memory_count": ep.memory_count,
                }
                for ep in result.episodes
            ],
            "boundaries": [
                {
                    "index": shift.index,
                    "time_gap_minutes": shift.time_gap_minutes,
                }
                for shift in result.topic_shifts
            ],
        },
    }
    filepath = save_results("experiment_llm_detection.json", results)
    click.echo(f"\nResults saved to: {filepath}")


@cli.command()
@click.option(
    "--conversation",
    default="conversation_20251024_083630_306692",
    help="Conversation ID to analyze",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Path to conversations directory",
)
@click.option("--chunk-size", default=50, help="Memories per LLM chunk")
@click.option("--max-chunks", default=20, help="Max chunks to process (None for all)")
@click.option(
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model to use for boundary detection",
)
def experiment_llm_filtered(
    conversation: str,
    conversations_dir: str,
    chunk_size: int,
    max_chunks: int,
    model: str,
) -> None:
    """Experiment: LLM + Rule-Based Filtering (Hybrid Approach)

    Uses LLM to detect boundaries, then filters out action type changes.
    """
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")
    click.echo(f"\nChunk size: {chunk_size}")
    click.echo(f"Max chunks: {max_chunks}")
    click.echo(f"Model: {model}")
    click.echo(
        f"Processing ~{chunk_size * max_chunks} memories ({max_chunks}/134 chunks)"
    )

    llm = create_llm()
    model_enum = SupportedModel(model)

    click.echo("\n--- Running LLM + Filtered Boundary Detection ---")
    result = detect_episodes_llm_filtered(
        memories=memories,
        llm=llm,
        model=model_enum,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )

    click.echo(f"\nEpisodes detected: {len(result.episodes)}")
    click.echo(f"Boundaries found: {len(result.topic_shifts)}")

    if result.episodes:
        sizes = [ep.memory_count for ep in result.episodes]
        durations = [ep.duration_minutes for ep in result.episodes]
        click.echo(
            f"Episode sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}"
        )
        click.echo(
            f"Episode durations: min={min(durations):.1f}min, max={max(durations):.1f}min, avg={sum(durations)/len(durations):.1f}min"
        )

    # Show detected boundaries with context
    click.echo("\n--- Filtered Boundaries ---")
    memory_by_id = {m.id: m for m in memories}

    for i, shift in enumerate(result.topic_shifts[:15]):  # Show first 15
        before = memory_by_id.get(shift.before_memory_id)
        after = memory_by_id.get(shift.after_memory_id)
        if before and after:
            click.echo(f"\nBoundary {i+1} (gap: {shift.time_gap_minutes:.1f} min):")
            before_text = (
                before.content[:100].encode("ascii", errors="replace").decode()
            )
            after_text = after.content[:100].encode("ascii", errors="replace").decode()
            click.echo(f"  BEFORE: {before_text}...")
            click.echo(f"  AFTER:  {after_text}...")

    # Save results
    results = {
        "experiment": "llm_filtered_boundary_detection",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "parameters": {
            "chunk_size": chunk_size,
            "max_chunks": max_chunks,
            "model": model,
        },
        "results": {
            "episode_count": len(result.episodes),
            "boundary_count": len(result.topic_shifts),
            "episodes": [
                {
                    "id": ep.id,
                    "start_time": ep.start_time.isoformat(),
                    "end_time": ep.end_time.isoformat(),
                    "duration_minutes": ep.duration_minutes,
                    "memory_count": ep.memory_count,
                }
                for ep in result.episodes
            ],
            "boundaries": [
                {
                    "index": shift.index,
                    "time_gap_minutes": shift.time_gap_minutes,
                }
                for shift in result.topic_shifts
            ],
        },
    }
    filepath = save_results("experiment_llm_filtered.json", results)
    click.echo(f"\nResults saved to: {filepath}")


@cli.command()
@click.option(
    "--conversation",
    default="conversation_20251024_083630_306692",
    help="Conversation ID to analyze",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Path to conversations directory",
)
@click.option("--num-episodes", default=5, help="Number of episodes to summarize")
@click.option(
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model to use for summarization",
)
@click.option(
    "--style",
    default="structured",
    type=click.Choice(["basic", "structured", "narrative", "question"]),
    help="Summary style",
)
def summarize_filtered(
    conversation: str,
    conversations_dir: str,
    num_episodes: int,
    model: str,
    style: str,
) -> None:
    """Summarize episodes from LLM-filtered detection.

    Loads the filtered detection results and generates summaries for
    representative episodes of varying sizes.
    """
    # Load the filtered detection results
    results_file = RESULTS_DIR / "experiment_llm_filtered.json"
    if not results_file.exists():
        click.echo("Error: Run experiment-llm-filtered first to generate episodes.")
        return

    with open(results_file) as f:
        detection_results = json.load(f)

    episode_data = detection_results["results"]["episodes"]
    click.echo(f"Loaded {len(episode_data)} episodes from filtered detection")

    # Load memories
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    # Sort episodes by size and pick representative ones (small, medium, large)
    sorted_episodes = sorted(episode_data, key=lambda e: e["memory_count"])

    # Pick episodes at different size percentiles
    indices = []
    n = len(sorted_episodes)
    if num_episodes >= 3:
        # Small, medium, large + extras
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1][:num_episodes]
    else:
        # Just evenly spaced
        step = n // (num_episodes + 1)
        indices = [step * (i + 1) for i in range(num_episodes)]

    selected_episodes = [sorted_episodes[i] for i in indices]

    click.echo(f"\nSelected {len(selected_episodes)} episodes for summarization:")
    for ep in selected_episodes:
        click.echo(
            f"  - {ep['memory_count']} memories, {ep['duration_minutes']:.1f} min"
        )

    # Initialize LLM
    llm = create_llm()
    model_enum = SupportedModel(model)

    # Generate summaries
    click.echo(f"\n--- Generating {style} summaries ---")
    summaries = []

    for i, ep_data in enumerate(selected_episodes):
        click.echo(
            f"\nEpisode {i+1}/{len(selected_episodes)} ({ep_data['memory_count']} memories):"
        )

        # Reconstruct Episode object
        from agent.experiments.episode_summaries.models import Episode

        episode = Episode(
            id=ep_data["id"],
            start_time=datetime.fromisoformat(ep_data["start_time"]),
            end_time=datetime.fromisoformat(ep_data["end_time"]),
            duration_minutes=ep_data["duration_minutes"],
            memory_ids=[],  # Will be filled by getting memories in time range
            memory_count=ep_data["memory_count"],
        )

        # Get memories for this episode by time range
        episode_memories = [
            m for m in memories if episode.start_time <= m.timestamp <= episode.end_time
        ]
        episode.memory_ids = [m.id for m in episode_memories]

        # Generate title first
        title = generate_episode_title(episode, memories, llm, model_enum)
        click.echo(f"  Title: {title}")

        # Generate summary
        summary = generate_episode_summary(
            episode, memories, llm, model_enum, style=style
        )

        # Encode for display
        summary_display = summary.encode("ascii", errors="replace").decode()
        click.echo(f"  Summary:\n{summary_display[:500]}...")

        summaries.append(
            {
                "episode_id": episode.id,
                "memory_count": episode.memory_count,
                "duration_minutes": episode.duration_minutes,
                "start_time": episode.start_time.isoformat(),
                "title": title,
                "summary": summary,
            }
        )

    # Save results
    results = {
        "experiment": "episode_summarization",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "parameters": {
            "num_episodes": num_episodes,
            "model": model,
            "style": style,
        },
        "summaries": summaries,
    }
    filepath = save_results("experiment_summarization.json", results)
    click.echo(f"\nResults saved to: {filepath}")


def count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3 is rough estimate for English)."""
    words = len(text.split())
    return int(words * 1.3)


@cli.command()
@click.option(
    "--conversation",
    default="conversation_20251024_083630_306692",
    help="Conversation ID to analyze",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Path to conversations directory",
)
@click.option("--num-episodes", default=5, help="Number of episodes to compare")
@click.option(
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model to use for summarization",
)
def experiment_detail_levels(
    conversation: str,
    conversations_dir: str,
    num_episodes: int,
    model: str,
) -> None:
    """Experiment 4: Compare summary detail levels (short/medium/detailed).

    Generates summaries at 3 detail levels for representative episodes
    and compares token counts and compression ratios.
    """
    # Load the filtered detection results
    results_file = RESULTS_DIR / "experiment_llm_filtered.json"
    if not results_file.exists():
        click.echo("Error: Run experiment-llm-filtered first to generate episodes.")
        return

    with open(results_file) as f:
        detection_results = json.load(f)

    episode_data = detection_results["results"]["episodes"]
    click.echo(f"Loaded {len(episode_data)} episodes from filtered detection")

    # Load memories
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    # Sort episodes by size and pick representative ones
    sorted_episodes = sorted(episode_data, key=lambda e: e["memory_count"])
    n = len(sorted_episodes)

    # Pick episodes at different size percentiles
    if num_episodes >= 5:
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1][:num_episodes]
    else:
        step = n // (num_episodes + 1)
        indices = [step * (i + 1) for i in range(num_episodes)]

    selected_episodes = [sorted_episodes[i] for i in indices]

    click.echo(f"\nSelected {len(selected_episodes)} episodes for comparison:")
    for ep in selected_episodes:
        click.echo(
            f"  - {ep['memory_count']} memories, {ep['duration_minutes']:.1f} min"
        )

    # Initialize LLM
    llm = create_llm()
    model_enum = SupportedModel(model)

    # Generate summaries at each detail level
    click.echo("\n--- Generating summaries at all detail levels ---")
    detail_levels = ["short", "medium", "detailed"]
    all_results = []

    for i, ep_data in enumerate(selected_episodes):
        click.echo(f"\n{'='*60}")
        click.echo(
            f"Episode {i+1}/{len(selected_episodes)} ({ep_data['memory_count']} memories)"
        )
        click.echo(f"{'='*60}")

        # Reconstruct Episode object
        from agent.experiments.episode_summaries.models import Episode

        episode = Episode(
            id=ep_data["id"],
            start_time=datetime.fromisoformat(ep_data["start_time"]),
            end_time=datetime.fromisoformat(ep_data["end_time"]),
            duration_minutes=ep_data["duration_minutes"],
            memory_ids=[],
            memory_count=ep_data["memory_count"],
        )

        # Get memories for this episode by time range
        episode_memories = [
            m for m in memories if episode.start_time <= m.timestamp <= episode.end_time
        ]
        episode.memory_ids = [m.id for m in episode_memories]

        # Count raw tokens
        raw_content = "\n".join(m.content for m in episode_memories)
        raw_tokens = count_tokens_approx(raw_content)
        click.echo(f"Raw content: ~{raw_tokens} tokens")

        episode_result = {
            "episode_id": episode.id,
            "memory_count": episode.memory_count,
            "duration_minutes": episode.duration_minutes,
            "raw_tokens": raw_tokens,
            "summaries": {},
        }

        for level in detail_levels:
            click.echo(f"\n  {level.upper()}:")
            summary = generate_summary_at_detail_level(
                episode, memories, llm, model_enum, level
            )
            summary_tokens = count_tokens_approx(summary)
            compression = raw_tokens / summary_tokens if summary_tokens > 0 else 0

            # Display summary (truncated for readability)
            summary_display = summary.encode("ascii", errors="replace").decode()
            if len(summary_display) > 300:
                summary_display = summary_display[:300] + "..."
            click.echo(f"    Tokens: {summary_tokens}, Compression: {compression:.1f}x")
            click.echo(f"    {summary_display}")

            episode_result["summaries"][level] = {
                "text": summary,
                "tokens": summary_tokens,
                "compression_ratio": compression,
            }

        all_results.append(episode_result)

    # Summary statistics
    click.echo(f"\n{'='*60}")
    click.echo("SUMMARY STATISTICS")
    click.echo(f"{'='*60}")

    for level in detail_levels:
        tokens = [r["summaries"][level]["tokens"] for r in all_results]
        compressions = [r["summaries"][level]["compression_ratio"] for r in all_results]
        avg_tokens = sum(tokens) / len(tokens)
        avg_compression = sum(compressions) / len(compressions)
        click.echo(
            f"  {level.upper():10s}: avg {avg_tokens:.0f} tokens, avg {avg_compression:.1f}x compression"
        )

    # Save results
    results = {
        "experiment": "detail_level_comparison",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "parameters": {
            "num_episodes": num_episodes,
            "model": model,
        },
        "episodes": all_results,
        "summary_stats": {
            level: {
                "avg_tokens": sum(r["summaries"][level]["tokens"] for r in all_results)
                / len(all_results),
                "avg_compression": sum(
                    r["summaries"][level]["compression_ratio"] for r in all_results
                )
                / len(all_results),
            }
            for level in detail_levels
        },
    }
    filepath = save_results("experiment_detail_levels.json", results)
    click.echo(f"\nResults saved to: {filepath}")


@cli.command()
@click.option(
    "--conversation",
    default="conversation_20251024_083630_306692",
    help="Conversation ID to analyze",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Path to conversations directory",
)
@click.option("--chunk-size", default=50, help="Memories per LLM chunk")
@click.option("--max-chunks", default=10, help="Max chunks to process")
@click.option(
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model to use for boundary detection",
)
@click.option(
    "--context-size", default=3, help="Memories to show before/after boundary"
)
def analyze_boundaries(
    conversation: str,
    conversations_dir: str,
    chunk_size: int,
    max_chunks: int,
    model: str,
    context_size: int,
) -> None:
    """Analyze boundary quality by comparing raw LLM vs filtered boundaries.

    This command runs both raw and filtered detection on the same data,
    then shows which boundaries were kept vs removed, with full semantic
    content around each boundary.
    """
    from agent.experiments.episode_summaries.detection import (
        detect_episodes_llm,
        filter_llm_boundaries,
        is_bad_boundary,
        is_good_boundary,
    )

    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    sorted_memories = sorted(memories, key=lambda m: m.timestamp)
    click.echo(f"\nProcessing {min(chunk_size * max_chunks, len(memories))} memories")
    click.echo(f"Model: {model}")

    llm = create_llm()
    model_enum = SupportedModel(model)

    # Run raw LLM detection
    click.echo("\n--- Running Raw LLM Detection ---")
    raw_result = detect_episodes_llm(
        memories=memories,
        llm=llm,
        model=model_enum,
        chunk_size=chunk_size,
        overlap=10,
        max_chunks=max_chunks,
    )

    raw_boundaries = [0] + [shift.index for shift in raw_result.topic_shifts]
    click.echo(f"Raw boundaries found: {len(raw_boundaries)}")

    # Apply filter
    filtered_boundaries = filter_llm_boundaries(raw_boundaries, sorted_memories)
    click.echo(f"Filtered boundaries: {len(filtered_boundaries)}")

    # Identify removed boundaries
    removed_boundaries = set(raw_boundaries) - set(filtered_boundaries)
    click.echo(f"Boundaries removed by filter: {len(removed_boundaries)}")

    def show_boundary_context(idx: int, label: str) -> None:
        """Show memories around a boundary with full content."""
        click.echo(f"\n{'='*70}")
        click.echo(f"{label} - Index {idx}")
        click.echo(f"{'='*70}")

        start = max(0, idx - context_size)
        end = min(len(sorted_memories), idx + context_size + 1)

        for i in range(start, end):
            m = sorted_memories[i]
            marker = ">>> " if i == idx else "    "
            time_str = m.timestamp.strftime("%H:%M:%S")

            # Calculate time gap from previous
            if i > 0:
                gap_seconds = (
                    m.timestamp - sorted_memories[i - 1].timestamp
                ).total_seconds()
                if gap_seconds >= 3600:
                    gap_str = f"(+{gap_seconds/3600:.1f}h)"
                elif gap_seconds >= 60:
                    gap_str = f"(+{gap_seconds/60:.1f}m)"
                else:
                    gap_str = f"(+{gap_seconds:.0f}s)"
            else:
                gap_str = ""

            # Show full content (truncated at 200 chars)
            content = m.content[:200]
            if len(m.content) > 200:
                content += "..."
            content = content.encode("ascii", errors="replace").decode()
            content = content.replace("\n", " ")

            click.echo(f"{marker}[{i}] {time_str} {gap_str}")
            click.echo(f"     {content}")

        # Show why it was classified
        after_content = sorted_memories[idx].content
        if is_bad_boundary(after_content):
            click.echo(f"\n  Filter verdict: REMOVED (matches BAD pattern)")
        elif is_good_boundary(after_content):
            click.echo(f"\n  Filter verdict: KEPT (matches GOOD pattern)")
        else:
            click.echo(f"\n  Filter verdict: KEPT (no pattern match)")

    # Show sample of REMOVED boundaries
    click.echo("\n" + "=" * 70)
    click.echo("BOUNDARIES REMOVED BY FILTER (sample)")
    click.echo("=" * 70)

    removed_list = sorted(removed_boundaries)[:10]
    for idx in removed_list:
        show_boundary_context(idx, "REMOVED")

    # Show sample of KEPT boundaries (non-zero)
    click.echo("\n" + "=" * 70)
    click.echo("BOUNDARIES KEPT BY FILTER (sample)")
    click.echo("=" * 70)

    kept_list = [b for b in filtered_boundaries if b != 0][:10]
    for idx in kept_list:
        show_boundary_context(idx, "KEPT")

    # Statistics on boundary content patterns
    click.echo("\n" + "=" * 70)
    click.echo("BOUNDARY CONTENT ANALYSIS")
    click.echo("=" * 70)

    def categorize_boundary(idx: int) -> str:
        """Categorize what type of content starts at this boundary."""
        content = sorted_memories[idx].content.lower()
        if "david said" in content:
            return "user_input"
        elif "i continue to exist" in content:
            return "idle"
        elif "i responded" in content:
            return "response"
        elif "i thought" in content:
            return "thought"
        elif "my mood" in content:
            return "mood_change"
        elif "i updated my appearance" in content:
            return "appearance"
        else:
            return "other"

    raw_categories: dict[str, int] = {}
    for idx in raw_boundaries:
        if idx >= len(sorted_memories):
            continue
        cat = categorize_boundary(idx)
        raw_categories[cat] = raw_categories.get(cat, 0) + 1

    filtered_categories: dict[str, int] = {}
    for idx in filtered_boundaries:
        if idx >= len(sorted_memories):
            continue
        cat = categorize_boundary(idx)
        filtered_categories[cat] = filtered_categories.get(cat, 0) + 1

    click.echo("\nRaw LLM boundaries by content type:")
    for cat, count in sorted(raw_categories.items(), key=lambda x: -x[1]):
        click.echo(f"  {cat}: {count}")

    click.echo("\nFiltered boundaries by content type:")
    for cat, count in sorted(filtered_categories.items(), key=lambda x: -x[1]):
        click.echo(f"  {cat}: {count}")

    # Save detailed results
    results = {
        "experiment": "boundary_analysis",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "chunk_size": chunk_size,
            "max_chunks": max_chunks,
            "model": model,
        },
        "raw_boundary_count": len(raw_boundaries),
        "filtered_boundary_count": len(filtered_boundaries),
        "removed_count": len(removed_boundaries),
        "raw_categories": raw_categories,
        "filtered_categories": filtered_categories,
        "raw_boundaries": list(raw_boundaries),
        "filtered_boundaries": list(filtered_boundaries),
        "removed_boundaries": list(removed_boundaries),
    }
    filepath = save_results("experiment_boundary_analysis.json", results)
    click.echo(f"\nResults saved to: {filepath}")


@cli.command()
@click.option(
    "--conversation",
    default="conversation_20251024_083630_306692",
    help="Conversation ID to analyze",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Path to conversations directory",
)
@click.option("--chunk-size", default=50, help="Memories per LLM chunk")
@click.option(
    "--max-chunks", default=None, type=int, help="Max chunks to process (None for all)"
)
@click.option(
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model to use for boundary detection",
)
def experiment_described_episodes(
    conversation: str,
    conversations_dir: str,
    chunk_size: int,
    max_chunks: int | None,
    model: str,
) -> None:
    """Experiment: Episode Detection with Descriptions.

    Uses a description-first prompt format that forces the LLM to describe
    what each episode is "about" before identifying the boundary index.
    This should produce more coherent episodes than previous approaches.

    Output format: "Description of episode" index
    Example: "Morning greeting and plans" 33
    """
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    click.echo(f"\nChunk size: {chunk_size}")
    click.echo(f"Max chunks: {max_chunks if max_chunks else 'all'}")
    click.echo(f"Model: {model}")

    # Calculate total chunks
    total_chunks = (len(memories) + chunk_size - 1) // chunk_size
    chunks_to_process = max_chunks if max_chunks else total_chunks
    memories_to_process = min(chunks_to_process * chunk_size, len(memories))
    click.echo(
        f"Processing ~{memories_to_process} memories ({chunks_to_process}/{total_chunks} chunks)"
    )

    llm = create_llm()
    model_enum = SupportedModel(model)

    click.echo("\n--- Running Episode Detection with Descriptions ---")
    result = detect_episodes_llm(
        memories=memories,
        llm=llm,
        model=model_enum,
        chunk_size=chunk_size,
        overlap=10,
        max_chunks=max_chunks,
    )

    click.echo(f"\nEpisodes detected: {len(result.episodes)}")

    # Show episode statistics
    if result.episodes:
        sizes = [ep.memory_count for ep in result.episodes]
        durations = [ep.duration_minutes for ep in result.episodes]
        click.echo(
            f"Episode sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}"
        )
        click.echo(
            f"Episode durations: min={min(durations):.1f}min, max={max(durations):.1f}min, "
            f"avg={sum(durations)/len(durations):.1f}min"
        )

    # Get episode descriptions from the all_boundaries dict
    # We need to access the boundaries to get the descriptions
    sorted_memories = sorted(memories, key=lambda m: m.timestamp)

    # Re-run to collect boundaries with descriptions
    all_boundaries: dict[int, str] = {0: "Start"}
    chunk_count = 0
    start = 0
    overlap = 10
    while start < len(sorted_memories):
        if max_chunks is not None and chunk_count >= max_chunks:
            break

        end = min(start + chunk_size, len(sorted_memories))
        chunk = sorted_memories[start:end]

        boundaries = detect_episodes_llm_chunk(
            chunk, llm, model_enum, start_index=start
        )
        for b in boundaries:
            if b.starts_at not in all_boundaries:
                all_boundaries[b.starts_at] = b.about

        start = end - overlap if end < len(sorted_memories) else end
        chunk_count += 1

    # Calculate actual memories processed (accounting for overlap)
    # The last chunk ends at this index
    actual_memories_processed = (
        min(
            chunk_size + (chunk_count - 1) * (chunk_size - overlap),
            len(sorted_memories),
        )
        if chunk_count > 0
        else 0
    )

    # Display episodes with their descriptions
    click.echo("\n--- Episodes with Descriptions ---")
    sorted_boundary_indices = sorted(all_boundaries.keys())

    episode_data_list = []
    for i, boundary_start in enumerate(sorted_boundary_indices):
        # Skip boundaries beyond what we processed
        if boundary_start >= actual_memories_processed:
            continue

        if i + 1 < len(sorted_boundary_indices):
            boundary_end = min(
                sorted_boundary_indices[i + 1], actual_memories_processed
            )
        else:
            boundary_end = actual_memories_processed

        episode_memories = sorted_memories[boundary_start:boundary_end]
        if not episode_memories:
            continue

        description = all_boundaries[boundary_start]
        memory_count = len(episode_memories)
        start_time = episode_memories[0].timestamp
        end_time = episode_memories[-1].timestamp
        duration_minutes = (end_time - start_time).total_seconds() / 60

        click.echo(f'\nEpisode {i+1}: "{description}"')
        click.echo(
            f"  Index: {boundary_start}, Memories: {memory_count}, Duration: {duration_minutes:.1f} min"
        )
        click.echo(
            f"  Time: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}"
        )

        # Show first memory as context
        first_content = episode_memories[0].content[:150]
        first_content = (
            first_content.encode("ascii", errors="replace").decode().replace("\n", " ")
        )
        click.echo(f"  First: {first_content}...")

        episode_data_list.append(
            {
                "index": boundary_start,
                "description": description,
                "memory_count": memory_count,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration_minutes,
            }
        )

    # Summary statistics
    click.echo("\n--- Summary Statistics ---")
    click.echo(f"Total episodes: {len(episode_data_list)}")

    if episode_data_list:
        sizes = [ep["memory_count"] for ep in episode_data_list]
        durations = [ep["duration_minutes"] for ep in episode_data_list]
        click.echo(
            f"Episode sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}"
        )
        click.echo(
            f"Episode durations: min={min(durations):.1f}min, max={max(durations):.1f}min, "
            f"avg={sum(durations)/len(durations):.1f}min"
        )

        # Size distribution
        small = sum(1 for s in sizes if s <= 10)
        medium = sum(1 for s in sizes if 10 < s <= 50)
        large = sum(1 for s in sizes if s > 50)
        click.echo(
            f"Size distribution: small(<=10)={small}, medium(11-50)={medium}, large(>50)={large}"
        )

    # Compare to previous results if available
    click.echo("\n--- Comparison to Previous Results ---")
    click.echo("Previous LLM detection (JSON format): 226 episodes")
    click.echo(f"Current description-first format: {len(episode_data_list)} episodes")

    # Save results
    results = {
        "experiment": "described_episodes",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "parameters": {
            "chunk_size": chunk_size,
            "max_chunks": max_chunks,
            "model": model,
            "overlap": overlap,
        },
        "results": {
            "episode_count": len(episode_data_list),
            "total_memories_processed": memories_to_process,
            "episodes": episode_data_list,
        },
        "statistics": {
            "sizes": {
                "min": min(sizes) if episode_data_list else 0,
                "max": max(sizes) if episode_data_list else 0,
                "avg": sum(sizes) / len(sizes) if episode_data_list else 0,
            },
            "durations": {
                "min_minutes": min(durations) if episode_data_list else 0,
                "max_minutes": max(durations) if episode_data_list else 0,
                "avg_minutes": (
                    sum(durations) / len(durations) if episode_data_list else 0
                ),
            },
        },
    }
    filepath = save_results("experiment_described_episodes.json", results)
    click.echo(f"\nResults saved to: {filepath}")


@cli.command()
@click.option(
    "--conversation",
    default="conversation_20251024_083630_306692",
    help="Conversation ID to analyze",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Path to conversations directory",
)
@click.option("--chunk-size", default=50, help="Memories per LLM chunk")
@click.option(
    "--max-chunks", default=None, type=int, help="Max chunks to process (None for all)"
)
@click.option(
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model to use for boundary detection",
)
def experiment_baseline_json(
    conversation: str,
    conversations_dir: str,
    chunk_size: int,
    max_chunks: int | None,
    model: str,
) -> None:
    """Experiment: JSON-Format Baseline Detection.

    Runs episode detection using the JSON output format as a baseline
    for comparison with description-first format.

    This helps understand if fragmentation is caused by format differences.
    """
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    click.echo(f"\nChunk size: {chunk_size}")
    click.echo(f"Max chunks: {max_chunks if max_chunks else 'all'}")
    click.echo(f"Model: {model}")

    # Calculate total chunks
    total_chunks = (len(memories) + chunk_size - 1) // chunk_size
    chunks_to_process = max_chunks if max_chunks else total_chunks
    memories_to_process = min(chunks_to_process * chunk_size, len(memories))
    click.echo(
        f"Processing ~{memories_to_process} memories ({chunks_to_process}/{total_chunks} chunks)"
    )

    llm = create_llm()
    model_enum = SupportedModel(model)

    click.echo("\n--- Running JSON-Format Episode Detection ---")
    result = detect_episodes_llm_json(
        memories=memories,
        llm=llm,
        model=model_enum,
        chunk_size=chunk_size,
        overlap=10,
        max_chunks=max_chunks,
    )

    click.echo(f"\nEpisodes detected: {len(result.episodes)}")
    click.echo(f"Boundaries found: {len(result.topic_shifts) + 1}")

    # Show episode statistics
    if result.episodes:
        sizes = [ep.memory_count for ep in result.episodes]
        durations = [ep.duration_minutes for ep in result.episodes]
        click.echo(
            f"Episode sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}"
        )
        click.echo(
            f"Episode durations: min={min(durations):.1f}min, max={max(durations):.1f}min, "
            f"avg={sum(durations)/len(durations):.1f}min"
        )

        # Size distribution
        small = sum(1 for s in sizes if s <= 10)
        medium = sum(1 for s in sizes if 10 < s <= 50)
        large = sum(1 for s in sizes if s > 50)
        click.echo(
            f"Size distribution: small(<=10)={small}, medium(11-50)={medium}, large(>50)={large}"
        )

    # Show detected boundaries with context
    click.echo("\n--- Detected Boundaries (first 15) ---")
    sorted_memories = sorted(memories, key=lambda m: m.timestamp)
    memory_by_id = {m.id: m for m in memories}

    for i, shift in enumerate(result.topic_shifts[:15]):
        before = memory_by_id.get(shift.before_memory_id)
        after = memory_by_id.get(shift.after_memory_id)
        if before and after:
            click.echo(
                f"\nBoundary {i+1} (index {shift.index}, gap: {shift.time_gap_minutes:.1f} min):"
            )
            before_text = (
                before.content[:100].encode("ascii", errors="replace").decode()
            )
            after_text = after.content[:100].encode("ascii", errors="replace").decode()
            click.echo(f"  BEFORE: {before_text}...")
            click.echo(f"  AFTER:  {after_text}...")

    # Save results
    results = {
        "experiment": "baseline_json_format",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "parameters": {
            "chunk_size": chunk_size,
            "max_chunks": max_chunks,
            "model": model,
            "overlap": 10,
            "format": "json",
        },
        "results": {
            "episode_count": len(result.episodes),
            "boundary_count": len(result.topic_shifts),
            "episodes": [
                {
                    "id": ep.id,
                    "start_time": ep.start_time.isoformat(),
                    "end_time": ep.end_time.isoformat(),
                    "duration_minutes": ep.duration_minutes,
                    "memory_count": ep.memory_count,
                }
                for ep in result.episodes
            ],
            "boundaries": [
                {
                    "index": shift.index,
                    "time_gap_minutes": shift.time_gap_minutes,
                }
                for shift in result.topic_shifts
            ],
        },
        "statistics": {
            "sizes": {
                "min": min(sizes) if result.episodes else 0,
                "max": max(sizes) if result.episodes else 0,
                "avg": sum(sizes) / len(sizes) if result.episodes else 0,
            },
            "durations": {
                "min_minutes": min(durations) if result.episodes else 0,
                "max_minutes": max(durations) if result.episodes else 0,
                "avg_minutes": (
                    sum(durations) / len(durations) if result.episodes else 0
                ),
            },
            "size_distribution": {
                "small_le_10": small if result.episodes else 0,
                "medium_11_50": medium if result.episodes else 0,
                "large_gt_50": large if result.episodes else 0,
            },
        },
    }
    filepath = save_results("experiment_baseline_json.json", results)
    click.echo(f"\nResults saved to: {filepath}")

    # Comparison note
    click.echo("\n--- Comparison Notes ---")
    click.echo("To compare with description-first format, also run:")
    click.echo(
        "  uv run python -m agent.experiments.episode_summaries.run_experiments experiment-described-episodes"
    )


@cli.command()
@click.option(
    "--conversation",
    default="conversation_20251024_083630_306692",
    help="Conversation ID to analyze",
)
@click.option(
    "--conversations-dir",
    default=str(DEFAULT_CONVERSATIONS_DIR),
    help="Path to conversations directory",
)
@click.option("--chunk-size", default=50, help="Memories per LLM chunk")
@click.option(
    "--max-chunks", default=None, type=int, help="Max chunks to process (None for all)"
)
@click.option(
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model to use for boundary detection",
)
def experiment_format_comparison(
    conversation: str,
    conversations_dir: str,
    chunk_size: int,
    max_chunks: int | None,
    model: str,
) -> None:
    """Experiment: Compare JSON vs Description-First Formats.

    Runs both formats on the SAME data in a single run to understand
    whether format differences cause fragmentation.

    This is Phase 1 of the fragmentation investigation.
    """
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    click.echo(f"\nChunk size: {chunk_size}")
    click.echo(f"Max chunks: {max_chunks if max_chunks else 'all'}")
    click.echo(f"Model: {model}")

    total_chunks = (len(memories) + chunk_size - 1) // chunk_size
    chunks_to_process = max_chunks if max_chunks else total_chunks
    memories_to_process = min(chunks_to_process * chunk_size, len(memories))
    click.echo(
        f"Processing ~{memories_to_process} memories ({chunks_to_process}/{total_chunks} chunks)"
    )

    llm = create_llm()
    model_enum = SupportedModel(model)

    sorted_memories = sorted(memories, key=lambda m: m.timestamp)

    # Run JSON format
    click.echo("\n" + "=" * 60)
    click.echo("PHASE 1: JSON Format")
    click.echo("=" * 60)

    result_json = detect_episodes_llm_json(
        memories=memories,
        llm=llm,
        model=model_enum,
        chunk_size=chunk_size,
        overlap=10,
        max_chunks=max_chunks,
    )

    json_episode_count = len(result_json.episodes)
    json_sizes = (
        [ep.memory_count for ep in result_json.episodes] if result_json.episodes else []
    )
    json_avg_size = sum(json_sizes) / len(json_sizes) if json_sizes else 0

    click.echo(f"Episodes: {json_episode_count}")
    click.echo(f"Avg size: {json_avg_size:.1f} memories")
    if json_sizes:
        json_small = sum(1 for s in json_sizes if s <= 10)
        json_medium = sum(1 for s in json_sizes if 10 < s <= 50)
        json_large = sum(1 for s in json_sizes if s > 50)
        click.echo(
            f"Distribution: small={json_small}, medium={json_medium}, large={json_large}"
        )

    # Run description-first format
    click.echo("\n" + "=" * 60)
    click.echo("PHASE 2: Description-First Format")
    click.echo("=" * 60)

    result_desc = detect_episodes_llm(
        memories=memories,
        llm=llm,
        model=model_enum,
        chunk_size=chunk_size,
        overlap=10,
        max_chunks=max_chunks,
    )

    desc_episode_count = len(result_desc.episodes)
    desc_sizes = (
        [ep.memory_count for ep in result_desc.episodes] if result_desc.episodes else []
    )
    desc_avg_size = sum(desc_sizes) / len(desc_sizes) if desc_sizes else 0

    click.echo(f"Episodes: {desc_episode_count}")
    click.echo(f"Avg size: {desc_avg_size:.1f} memories")
    if desc_sizes:
        desc_small = sum(1 for s in desc_sizes if s <= 10)
        desc_medium = sum(1 for s in desc_sizes if 10 < s <= 50)
        desc_large = sum(1 for s in desc_sizes if s > 50)
        click.echo(
            f"Distribution: small={desc_small}, medium={desc_medium}, large={desc_large}"
        )

    # Comparison
    click.echo("\n" + "=" * 60)
    click.echo("COMPARISON")
    click.echo("=" * 60)

    click.echo(
        f"\n{'Format':<20} {'Episodes':<12} {'Avg Size':<12} {'Small':<8} {'Medium':<8} {'Large':<8}"
    )
    click.echo("-" * 68)
    click.echo(
        f"{'JSON':<20} {json_episode_count:<12} {json_avg_size:<12.1f} {json_small if json_sizes else 0:<8} {json_medium if json_sizes else 0:<8} {json_large if json_sizes else 0:<8}"
    )
    click.echo(
        f"{'Description-First':<20} {desc_episode_count:<12} {desc_avg_size:<12.1f} {desc_small if desc_sizes else 0:<8} {desc_medium if desc_sizes else 0:<8} {desc_large if desc_sizes else 0:<8}"
    )

    if json_episode_count > 0:
        ratio = desc_episode_count / json_episode_count
        click.echo(
            f"\nFragmentation ratio: {ratio:.2f}x (desc-first produces {ratio:.2f}x more episodes)"
        )

    # Analyze per-chunk boundary counts
    click.echo("\n" + "=" * 60)
    click.echo("PER-CHUNK ANALYSIS")
    click.echo("=" * 60)

    # Get boundary indices
    json_boundaries = set([0] + [s.index for s in result_json.topic_shifts])
    desc_boundaries = set([0] + [s.index for s in result_desc.topic_shifts])

    # Analyze which boundaries are shared vs unique
    shared = json_boundaries & desc_boundaries
    json_only = json_boundaries - desc_boundaries
    desc_only = desc_boundaries - json_boundaries

    click.echo(f"\nBoundary overlap analysis:")
    click.echo(f"  Shared boundaries: {len(shared)}")
    click.echo(f"  JSON-only boundaries: {len(json_only)}")
    click.echo(f"  Description-first-only boundaries: {len(desc_only)}")

    # Sample boundaries unique to description-first
    if desc_only:
        click.echo(f"\nSample boundaries unique to description-first (first 10):")
        for idx in sorted(desc_only)[:10]:
            if idx < len(sorted_memories):
                mem = sorted_memories[idx]
                content = (
                    mem.content[:80]
                    .encode("ascii", errors="replace")
                    .decode()
                    .replace("\n", " ")
                )
                click.echo(f"  [{idx}] {content}...")

    # Save comparison results
    results = {
        "experiment": "format_comparison",
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "parameters": {
            "chunk_size": chunk_size,
            "max_chunks": max_chunks,
            "model": model,
            "overlap": 10,
        },
        "json_format": {
            "episode_count": json_episode_count,
            "avg_size": json_avg_size,
            "sizes": json_sizes,
            "boundary_indices": sorted(json_boundaries),
        },
        "description_first": {
            "episode_count": desc_episode_count,
            "avg_size": desc_avg_size,
            "sizes": desc_sizes,
            "boundary_indices": sorted(desc_boundaries),
        },
        "comparison": {
            "fragmentation_ratio": (
                desc_episode_count / json_episode_count if json_episode_count > 0 else 0
            ),
            "shared_boundaries": len(shared),
            "json_only_boundaries": len(json_only),
            "desc_only_boundaries": len(desc_only),
            "shared_boundary_indices": sorted(shared),
            "json_only_indices": sorted(json_only),
            "desc_only_indices": sorted(desc_only),
        },
    }
    filepath = save_results("experiment_format_comparison.json", results)
    click.echo(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    cli()
