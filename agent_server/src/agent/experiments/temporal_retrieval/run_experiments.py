"""
Temporal Retrieval Experiment Runner

CLI for running temporal retrieval experiments.
"""

import json
from datetime import datetime
from pathlib import Path

import click

from agent.conversation_persistence import ConversationPersistence
from agent.experiments.temporal_retrieval.build_index import (
    build_index_from_conversation,
    load_memories,
)
from agent.experiments.temporal_retrieval.emotional_time import (
    EmotionalTimeResolver,
    evaluate_emotional_time_approaches,
)
from agent.experiments.temporal_retrieval.episode_index import EpisodeIndex
from agent.experiments.temporal_retrieval.evaluate import (
    generate_findings_report,
    print_evaluation_summary,
    run_full_evaluation,
    save_evaluation_results,
)
from agent.experiments.temporal_retrieval.strategies import (
    compare_strategies,
    retrieve_with_strategy,
)
from agent.experiments.temporal_retrieval.test_data import (
    generate_test_dataset,
    load_test_dataset,
    save_test_dataset,
)
from agent.experiments.temporal_retrieval.time_parser import (
    parse_time_reference,
    test_time_parser,
)
from agent.llm.models import SupportedModel
from agent.llm.router import create_llm
from agent.memory.dag.dag_memory_manager import DagMemoryManager


# Default configuration
DEFAULT_CONVERSATION_PREFIX = "conversation_20251024_083630_306692"
DEFAULT_CONVERSATIONS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "conversations"
)
RESULTS_DIR = Path(__file__).parent / "output" / "results"


def ensure_results_dir() -> Path:
    """Ensure results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


@click.group()
def cli() -> None:
    """Temporal Retrieval Experiment CLI"""
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
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model to use for summarization",
)
@click.option(
    "--max-chunks",
    default=None,
    type=int,
    help="Maximum chunks to process (None for all)",
)
@click.option(
    "--no-summaries",
    is_flag=True,
    help="Skip summary generation (faster)",
)
def build_index(
    conversation: str,
    conversations_dir: str,
    model: str,
    max_chunks: int | None,
    no_summaries: bool,
) -> None:
    """Build episode index from conversation."""
    click.echo(f"Building episode index from {conversation}...")

    llm = create_llm()
    model_enum = SupportedModel(model)

    output_path = ensure_results_dir() / "episode_index.json"

    index = build_index_from_conversation(
        conversations_dir=Path(conversations_dir),
        conversation_prefix=conversation,
        llm=llm,
        model=model_enum,
        output_path=output_path,
        max_chunks=max_chunks,
        generate_summaries=not no_summaries,
    )

    click.echo(f"\nIndex built with {len(index)} episodes")
    click.echo(f"Topics: {len(index.get_all_topics())}")
    click.echo(f"Moods: {len(index.get_all_moods())}")
    click.echo(f"\nSaved to: {output_path}")


@cli.command()
def test_parser() -> None:
    """Test time expression parsing."""
    click.echo("Testing time parser...\n")

    results = test_time_parser()

    for result in results:
        click.echo(f"Input: {result['input']}")
        click.echo(f"  Type: {result['ref_type']}")
        if result["start_time"]:
            click.echo(f"  Start: {result['start_time']}")
            click.echo(f"  End: {result['end_time']}")
        if result["mood_filter"]:
            click.echo(f"  Mood: {result['mood_filter']}")
        if result["topic_filter"]:
            click.echo(f"  Topic: {result['topic_filter']}")
        click.echo()


@cli.command()
@click.option(
    "--query",
    "-q",
    required=True,
    help="Query to parse",
)
def parse_query(query: str) -> None:
    """Parse a single time query."""
    now = datetime.now()
    result = parse_time_reference(query, now)

    if result:
        click.echo(f"Query: {query}")
        click.echo(f"Type: {result.ref_type}")
        if result.start_time:
            click.echo(f"Start: {result.start_time}")
            click.echo(f"End: {result.end_time}")
        if result.mood_filter:
            click.echo(f"Mood filter: {result.mood_filter}")
        if result.topic_filter:
            click.echo(f"Topic filter: {result.topic_filter}")
    else:
        click.echo(f"Could not parse time reference from: {query}")


@cli.command()
@click.option(
    "--relative-count",
    default=40,
    help="Number of relative/time-of-day queries",
)
@click.option(
    "--absolute-count",
    default=30,
    help="Number of date-based queries",
)
@click.option(
    "--emotional-count",
    default=30,
    help="Number of emotional/topic queries",
)
def generate_queries(
    relative_count: int,
    absolute_count: int,
    emotional_count: int,
) -> None:
    """Generate test queries dataset based on actual episode data."""
    click.echo("Loading episode index...")

    index_path = RESULTS_DIR / "episode_index.json"
    if not index_path.exists():
        click.echo("Error: Episode index not found. Run 'build-index' first.")
        return

    index = EpisodeIndex.load(index_path)
    click.echo(f"Loaded index with {len(index)} episodes")

    click.echo("\nGenerating test queries from actual episode data...")
    queries = generate_test_dataset(
        index=index,
        relative_count=relative_count,
        absolute_count=absolute_count,
        emotional_count=emotional_count,
    )

    output_path = ensure_results_dir() / "test_queries.json"
    save_test_dataset(queries, output_path)

    click.echo(f"\nGenerated {len(queries)} queries:")
    click.echo(f"  - Relative time: {relative_count}")
    click.echo(f"  - Absolute time: {absolute_count}")
    click.echo(f"  - Emotional time: {emotional_count}")
    click.echo(f"\nSaved to: {output_path}")


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
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model for relevance evaluation",
)
@click.option(
    "--eval-relevance",
    is_flag=True,
    help="Evaluate content relevance (slower, requires LLM)",
)
def evaluate(
    conversation: str,
    conversations_dir: str,
    model: str,
    eval_relevance: bool,
) -> None:
    """Run full evaluation of retrieval strategies."""
    click.echo("Loading data...")

    # Load index
    index_path = RESULTS_DIR / "episode_index.json"
    if not index_path.exists():
        click.echo("Error: Episode index not found. Run 'build-index' first.")
        return

    index = EpisodeIndex.load(index_path)
    click.echo(f"Loaded index with {len(index)} episodes")

    # Load test queries
    queries_path = RESULTS_DIR / "test_queries.json"
    if not queries_path.exists():
        click.echo("Error: Test queries not found. Run 'generate-queries' first.")
        return

    queries = load_test_dataset(queries_path)
    click.echo(f"Loaded {len(queries)} test queries")

    # Load memories
    memories = load_memories(Path(conversations_dir), conversation)
    click.echo(f"Loaded {len(memories)} memories")

    # Initialize LLM for time parsing and optional relevance evaluation
    llm = create_llm()
    model_enum = SupportedModel(model)

    # Run evaluation
    click.echo("\nRunning evaluation (using LLM for time parsing)...")
    results = run_full_evaluation(
        queries=queries,
        index=index,
        memories=memories,
        llm=llm,
        model=model_enum,
        evaluate_relevance=eval_relevance,
    )

    # Print summary
    print_evaluation_summary(results)

    # Save results
    results_path = ensure_results_dir() / "evaluation_results.json"
    save_evaluation_results(results, results_path)
    click.echo(f"\nResults saved to: {results_path}")

    # Generate findings report (tracked at experiment root, not under output/)
    findings_path = Path(__file__).parent / "FINDINGS.md"
    generate_findings_report(results, findings_path)
    click.echo(f"Findings saved to: {findings_path}")


@cli.command()
@click.option(
    "--query",
    "-q",
    required=True,
    help="Query to test",
)
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
def test_query(
    query: str,
    conversation: str,
    conversations_dir: str,
) -> None:
    """Test retrieval with a single query."""
    click.echo(f"Testing query: {query}\n")

    # Load index
    index_path = RESULTS_DIR / "episode_index.json"
    if not index_path.exists():
        click.echo("Error: Episode index not found. Run 'build-index' first.")
        return

    index = EpisodeIndex.load(index_path)

    # Load memories
    memories = load_memories(Path(conversations_dir), conversation)

    # Compare strategies
    results = compare_strategies(
        query=query,
        index=index,
        memories=memories,
        top_k=3,
    )

    for name, result in results.items():
        click.echo(f"\n{'='*60}")
        click.echo(f"Strategy {name}: {result.strategy}")
        click.echo(f"Latency: {result.latency_ms:.1f}ms")
        click.echo(f"Episodes retrieved: {len(result.retrieved_episode_ids)}")

        if result.retrieved_summaries:
            click.echo("\nRetrieved content:")
            for i, summary in enumerate(result.retrieved_summaries[:2]):
                # Truncate for display
                display = summary[:300] if len(summary) > 300 else summary
                display = display.encode("ascii", errors="replace").decode()
                click.echo(f"\n[{i+1}] {display}...")


@cli.command()
@click.option(
    "--model",
    default="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    help="Model for LLM filtering",
)
def emotional_time(model: str) -> None:
    """Test emotional time handling approaches."""
    click.echo("Testing emotional time approaches...\n")

    # Load index
    index_path = RESULTS_DIR / "episode_index.json"
    if not index_path.exists():
        click.echo("Error: Episode index not found. Run 'build-index' first.")
        return

    index = EpisodeIndex.load(index_path)
    click.echo(f"Loaded index with {len(index)} episodes")

    llm = create_llm()
    model_enum = SupportedModel(model)

    resolver = EmotionalTimeResolver(index, llm)

    # Test queries
    test_queries = [
        {"query": "when I was stressed", "mood_filter": "stressed"},
        {"query": "when I felt happy", "mood_filter": "happy"},
        {"query": "during the project", "topic_filter": "project"},
        {"query": "that rough period", "mood_filter": "stressed"},
    ]

    click.echo("Testing different approaches:\n")

    for approach in ["metadata", "semantic", "llm"]:
        click.echo(f"\n{'='*60}")
        click.echo(f"Approach: {approach.upper()}")
        click.echo(f"{'='*60}")

        for test in test_queries:
            query = str(test["query"])
            mood = test.get("mood_filter")
            topic = test.get("topic_filter")

            click.echo(f"\nQuery: {query}")

            try:
                if approach == "llm":
                    episodes = resolver.resolve(
                        query=query,
                        mood_filter=str(mood) if mood else None,
                        topic_filter=str(topic) if topic else None,
                        approach=approach,
                        model=model_enum,
                    )
                else:
                    episodes = resolver.resolve(
                        query=query,
                        mood_filter=str(mood) if mood else None,
                        topic_filter=str(topic) if topic else None,
                        approach=approach,
                    )

                click.echo(f"  Found {len(episodes)} episodes")
                for ep in episodes[:2]:
                    title = ep.title or "Untitled"
                    click.echo(f"    - {title} (moods: {ep.moods}, topics: {ep.topics})")

            except Exception as e:
                click.echo(f"  Error: {e}")


@cli.command()
def show_index() -> None:
    """Show summary of the episode index."""
    index_path = RESULTS_DIR / "episode_index.json"
    if not index_path.exists():
        click.echo("Error: Episode index not found. Run 'build-index' first.")
        return

    index = EpisodeIndex.load(index_path)

    click.echo(f"Episode Index Summary")
    click.echo(f"{'='*60}")
    click.echo(f"Total episodes: {len(index)}")
    click.echo(f"Unique topics: {len(index.get_all_topics())}")
    click.echo(f"Unique moods: {len(index.get_all_moods())}")

    # Time range
    episodes = index.get_all_episodes()
    if episodes:
        min_time = min(ep.start_time for ep in episodes)
        max_time = max(ep.end_time for ep in episodes)
        click.echo(f"Time range: {min_time} to {max_time}")

    # Top topics
    click.echo(f"\nTop topics: {', '.join(index.get_all_topics()[:10])}")

    # Top moods
    click.echo(f"Moods: {', '.join(index.get_all_moods())}")

    # Episode size statistics
    sizes = [ep.memory_count for ep in episodes]
    if sizes:
        click.echo(f"\nEpisode sizes:")
        click.echo(f"  Min: {min(sizes)}")
        click.echo(f"  Max: {max(sizes)}")
        click.echo(f"  Avg: {sum(sizes)/len(sizes):.1f}")

    # Sample episodes
    click.echo(f"\nSample episodes:")
    for ep in episodes[:5]:
        title = ep.title or "Untitled"
        click.echo(
            f"  - {title} ({ep.memory_count} memories, {ep.duration_minutes:.0f} min)"
        )


if __name__ == "__main__":
    cli()
