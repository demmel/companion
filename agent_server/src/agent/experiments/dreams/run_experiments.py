"""
Main experiment runner for dreams prototype.

Runs experiments from PLAN.md to explore different traversal strategies,
narrative styles, dream depths, and seed selection methods.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from agent.conversation_persistence import ConversationPersistence
from agent.llm import create_llm, SupportedModel
from agent.memory.dag.dag_memory_manager import DagMemoryManager
from agent.ui_output import ui_print

from .models import (
    Dream,
    DreamConfig,
    DreamEvaluation,
    DreamMode,
    ExperimentResult,
    TraversalStrategy,
    NarrativeStyle,
    SeedSelection,
)
from .dreamer import Dreamer
from .seed_selection import select_seed
from .traversal import traverse

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default conversation to use
DEFAULT_CONVERSATION = "conversation_20251024_083630_306692"


def save_dream(dream: Dream, output_dir: Path, prefix: str, index: int) -> None:
    """Save a dream to a text file."""
    filename = f"{prefix}_dream_{index + 1}.txt"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Dream #{index + 1}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Config: {dream.config}\n")
        f.write(f"Seed Memory: {dream.seed_memory_id}\n")
        f.write(f"Memories Visited: {dream.duration_memories}\n")
        f.write(f"Themes: {', '.join(dream.themes_emerged)}\n")
        f.write(f"Created: {dream.created_at}\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write("NARRATIVE:\n")
        f.write("-" * 60 + "\n\n")
        f.write(dream.narrative)
        f.write("\n\n" + "-" * 60 + "\n")
        f.write("TRAVERSAL PATH:\n")
        f.write("-" * 60 + "\n")
        for i, mem_id in enumerate(dream.traversal_path):
            f.write(f"  {i + 1}. {mem_id}\n")


def save_experiment_summary(
    experiment_name: str, dreams: list[Dream], output_dir: Path
) -> None:
    """Save experiment summary to JSON."""
    summary = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "total_dreams": len(dreams),
        "dreams": [
            {
                "seed_memory_id": d.seed_memory_id,
                "depth": d.duration_memories,
                "themes": d.themes_emerged,
                "config": {
                    "seed_selection": d.config.seed_selection.value,
                    "traversal_strategy": d.config.traversal_strategy.value,
                    "depth": d.config.depth,
                    "narrative_style": d.config.narrative_style.value,
                },
            }
            for d in dreams
        ],
    }

    filepath = output_dir / f"{experiment_name}_summary.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_experiment_1_traversal_comparison(
    dreamer: Dreamer, output_dir: Path, dreams_per_strategy: int = 3
) -> list[Dream]:
    """
    Experiment 1: Traversal Strategy Comparison

    Generate dreams using different traversal strategies with the same
    seed selection and depth.
    """
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT 1: Traversal Strategy Comparison")
    ui_print("=" * 60)

    strategies = [
        TraversalStrategy.RANDOM_JUMP,
        TraversalStrategy.RECENCY_WEIGHTED,
        TraversalStrategy.SEMANTIC_DRIFT,
        TraversalStrategy.CONTRAST_SEEKING,
        TraversalStrategy.EDGE_FOLLOWING,
    ]

    all_dreams: list[Dream] = []
    exp_dir = output_dir / "exp1_traversal"
    exp_dir.mkdir(exist_ok=True)

    for strategy in strategies:
        ui_print(f"\n--- Strategy: {strategy.value} ---")

        for i in range(dreams_per_strategy):
            config = DreamConfig(
                seed_selection=SeedSelection.RANDOM,
                traversal_strategy=strategy,
                depth=5,
                narrative_style=NarrativeStyle.FRAGMENT,
            )

            try:
                dream = dreamer.dream(config)
                all_dreams.append(dream)
                save_dream(dream, exp_dir, strategy.value, i)
                ui_print(
                    f"  Dream {i + 1}: {len(dream.narrative)} chars, themes: {dream.themes_emerged[:3]}"
                )
            except Exception as e:
                ui_print(f"  Dream {i + 1}: ERROR - {e}")
                logger.exception(f"Failed to generate dream: {e}")

    save_experiment_summary("exp1_traversal", all_dreams, exp_dir)
    ui_print(f"\nGenerated {len(all_dreams)} dreams for Experiment 1")
    return all_dreams


def run_experiment_2_narrative_comparison(
    dreamer: Dreamer, output_dir: Path
) -> list[Dream]:
    """
    Experiment 2: Narrative Style Comparison

    Use the same traversal path to generate narratives in different styles.
    """
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT 2: Narrative Style Comparison")
    ui_print("=" * 60)

    styles = [
        NarrativeStyle.FRAGMENT,
        NarrativeStyle.STREAM,
        NarrativeStyle.POETIC,
        NarrativeStyle.SENSORY,
    ]

    all_dreams: list[Dream] = []
    exp_dir = output_dir / "exp2_narrative"
    exp_dir.mkdir(exist_ok=True)

    # First, generate a fixed traversal path
    seed_id = select_seed(dreamer.memory_graph, "emotional")
    if seed_id is None:
        ui_print("ERROR: Could not select seed")
        return all_dreams

    traversal_path, _ = traverse(
        dreamer.memory_graph, seed_id, depth=5, strategy="random_jump"
    )

    ui_print(f"Fixed traversal path: {len(traversal_path)} memories")

    # Generate narrative in each style for the same path
    for style in styles:
        ui_print(f"\n--- Style: {style.value} ---")

        try:
            dream = dreamer.dream_with_fixed_path(traversal_path, style)
            all_dreams.append(dream)
            save_dream(dream, exp_dir, style.value, 0)
            ui_print(f"  Generated: {len(dream.narrative)} chars")
        except Exception as e:
            ui_print(f"  ERROR - {e}")
            logger.exception(f"Failed to generate dream: {e}")

    save_experiment_summary("exp2_narrative", all_dreams, exp_dir)
    ui_print(f"\nGenerated {len(all_dreams)} dreams for Experiment 2")
    return all_dreams


def run_experiment_3_depth_analysis(dreamer: Dreamer, output_dir: Path) -> list[Dream]:
    """
    Experiment 3: Dream Length Analysis

    Generate dreams at different depths to find optimal length.
    """
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT 3: Dream Length Analysis")
    ui_print("=" * 60)

    depths = [3, 5, 7, 10, 15]
    all_dreams: list[Dream] = []
    exp_dir = output_dir / "exp3_depth"
    exp_dir.mkdir(exist_ok=True)

    for depth in depths:
        ui_print(f"\n--- Depth: {depth} ---")

        config = DreamConfig(
            seed_selection=SeedSelection.EMOTIONAL,
            traversal_strategy=TraversalStrategy.RANDOM_JUMP,
            depth=depth,
            narrative_style=NarrativeStyle.FRAGMENT,
        )

        try:
            dream = dreamer.dream(config)
            all_dreams.append(dream)
            save_dream(dream, exp_dir, f"depth_{depth}", 0)
            ui_print(
                f"  Generated: {len(dream.narrative)} chars, themes: {dream.themes_emerged[:3]}"
            )
        except Exception as e:
            ui_print(f"  ERROR - {e}")
            logger.exception(f"Failed to generate dream: {e}")

    save_experiment_summary("exp3_depth", all_dreams, exp_dir)
    ui_print(f"\nGenerated {len(all_dreams)} dreams for Experiment 3")
    return all_dreams


def run_experiment_4_seed_selection(
    dreamer: Dreamer, output_dir: Path, dreams_per_seed: int = 2
) -> list[Dream]:
    """
    Experiment 4: Seed Selection Impact

    Compare different seed selection strategies.
    """
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT 4: Seed Selection Impact")
    ui_print("=" * 60)

    seeds = [
        SeedSelection.RANDOM,
        SeedSelection.RECENT,
        SeedSelection.EMOTIONAL,
        SeedSelection.UNPROCESSED,
    ]

    all_dreams: list[Dream] = []
    exp_dir = output_dir / "exp4_seed"
    exp_dir.mkdir(exist_ok=True)

    for seed_type in seeds:
        ui_print(f"\n--- Seed: {seed_type.value} ---")

        for i in range(dreams_per_seed):
            config = DreamConfig(
                seed_selection=seed_type,
                traversal_strategy=TraversalStrategy.RANDOM_JUMP,
                depth=5,
                narrative_style=NarrativeStyle.FRAGMENT,
            )

            try:
                dream = dreamer.dream(config)
                all_dreams.append(dream)
                save_dream(dream, exp_dir, seed_type.value, i)

                # Show the seed memory content
                seed_mem = dreamer.memory_graph.elements.get(dream.seed_memory_id)
                seed_preview = seed_mem.content[:50] + "..." if seed_mem else "N/A"
                ui_print(f'  Dream {i + 1}: seed="{seed_preview}"')
            except Exception as e:
                ui_print(f"  Dream {i + 1}: ERROR - {e}")
                logger.exception(f"Failed to generate dream: {e}")

    save_experiment_summary("exp4_seed", all_dreams, exp_dir)
    ui_print(f"\nGenerated {len(all_dreams)} dreams for Experiment 4")
    return all_dreams


def run_experiment_5_graph_structure(
    dreamer: Dreamer, output_dir: Path, dreams_per_mode: int = 3
) -> list[Dream]:
    """
    Experiment 5: Graph Structure Impact

    Compare edge-following vs edge-ignoring traversal.
    """
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT 5: Graph Structure Impact")
    ui_print("=" * 60)

    # First, analyze graph structure
    graph = dreamer.memory_graph
    ui_print(f"\nGraph analysis:")
    ui_print(f"  Nodes (memories): {len(graph.elements)}")
    ui_print(f"  Edges: {len(graph.edges)}")

    # Count connected memories
    connected = set()
    for edge in graph.edges.values():
        connected.add(edge.source_id)
        connected.add(edge.target_id)
    connectivity = len(connected) / len(graph.elements) * 100 if graph.elements else 0
    ui_print(f"  Connected memories: {len(connected)} ({connectivity:.1f}%)")

    all_dreams: list[Dream] = []
    exp_dir = output_dir / "exp5_graph"
    exp_dir.mkdir(exist_ok=True)

    # Edge-following dreams
    ui_print(f"\n--- Edge-Following ---")
    for i in range(dreams_per_mode):
        config = DreamConfig(
            seed_selection=SeedSelection.RANDOM,
            traversal_strategy=TraversalStrategy.EDGE_FOLLOWING,
            depth=5,
            narrative_style=NarrativeStyle.FRAGMENT,
        )

        try:
            dream = dreamer.dream(config)
            all_dreams.append(dream)
            save_dream(dream, exp_dir, "edge_following", i)
            ui_print(f"  Dream {i + 1}: {len(dream.edges_used)} edges used")
        except Exception as e:
            ui_print(f"  Dream {i + 1}: ERROR - {e}")

    # Random (edge-ignoring) dreams
    ui_print(f"\n--- Edge-Ignoring (Random) ---")
    for i in range(dreams_per_mode):
        config = DreamConfig(
            seed_selection=SeedSelection.RANDOM,
            traversal_strategy=TraversalStrategy.RANDOM_JUMP,
            depth=5,
            narrative_style=NarrativeStyle.FRAGMENT,
        )

        try:
            dream = dreamer.dream(config)
            all_dreams.append(dream)
            save_dream(dream, exp_dir, "edge_ignoring", i)
            ui_print(f"  Dream {i + 1}: random traversal")
        except Exception as e:
            ui_print(f"  Dream {i + 1}: ERROR - {e}")

    save_experiment_summary("exp5_graph", all_dreams, exp_dir)
    ui_print(f"\nGenerated {len(all_dreams)} dreams for Experiment 5")
    return all_dreams


def analyze_graph_connectivity(dreamer: Dreamer) -> tuple[list[str], list[str]]:
    """
    Analyze graph to find hub (highly connected) and peripheral (isolated) memories.

    Returns:
        Tuple of (hub_ids, peripheral_ids) - 5 each
    """
    graph = dreamer.memory_graph

    # Count edges per memory
    edge_counts: dict[str, int] = {mem_id: 0 for mem_id in graph.elements.keys()}

    for edge in graph.edges.values():
        if edge.source_id in edge_counts:
            edge_counts[edge.source_id] += 1
        if edge.target_id in edge_counts:
            edge_counts[edge.target_id] += 1

    # Sort by edge count
    sorted_memories = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)

    # Get top 5 hubs (most connected)
    hubs = [mem_id for mem_id, count in sorted_memories[:5]]

    # Get bottom 5 peripherals (least connected)
    peripherals = [mem_id for mem_id, count in sorted_memories[-5:]]

    return hubs, peripherals


def run_experiment_7_seed_influence(
    dreamer: Dreamer, output_dir: Path, dreams_per_seed: int = 10
) -> list[Dream]:
    """
    Experiment 7: Seed Influence

    Test whether the same seed produces similar dreams across multiple runs.
    Pick 3 diverse seed memories and generate multiple dreams from each.
    """
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT 7: Seed Influence")
    ui_print("=" * 60)

    all_dreams: list[Dream] = []
    exp_dir = output_dir / "exp7_seed_influence"
    exp_dir.mkdir(exist_ok=True)

    graph = dreamer.memory_graph

    # Pick 3 diverse seeds manually by looking at memory content
    # Get an emotional, factual, and random seed
    from .seed_selection import select_emotional_seed, select_random_seed

    seed_ids = []

    # Emotional seed
    emotional_seed = select_emotional_seed(graph)
    if emotional_seed:
        seed_ids.append(("emotional", emotional_seed))

    # Get a different random seed
    random_seed = select_random_seed(graph)
    if random_seed and random_seed != emotional_seed:
        seed_ids.append(("random", random_seed))

    # Get another random seed
    for _ in range(10):  # Try up to 10 times to get a unique one
        another_seed = select_random_seed(graph)
        if another_seed and another_seed not in [s[1] for s in seed_ids]:
            seed_ids.append(("random2", another_seed))
            break

    ui_print(f"Selected {len(seed_ids)} seed memories")

    for seed_name, seed_id in seed_ids:
        ui_print(f"\n--- Seed: {seed_name} ({seed_id[:8]}...) ---")

        # Show seed content
        seed_mem = graph.elements.get(seed_id)
        if seed_mem:
            # Sanitize for Windows console
            content_preview = (
                seed_mem.content[:80].encode("ascii", "replace").decode("ascii")
            )
            ui_print(f"  Content: {content_preview}...")

        seed_dreams: list[Dream] = []
        seed_themes: list[list[str]] = []
        seed_paths: list[set[str]] = []

        for i in range(dreams_per_seed):
            config = DreamConfig(
                seed_selection=SeedSelection.RANDOM,  # Not used since we pass fixed_seed
                traversal_strategy=TraversalStrategy.RANDOM_JUMP,
                depth=5,
                narrative_style=NarrativeStyle.FRAGMENT,
            )

            try:
                dream = dreamer.dream(config, fixed_seed_id=seed_id)
                all_dreams.append(dream)
                seed_dreams.append(dream)
                seed_themes.append(dream.themes_emerged)
                seed_paths.append(set(dream.traversal_path))
                save_dream(dream, exp_dir, f"{seed_name}_seed", i)
                ui_print(f"  Dream {i + 1}: themes={dream.themes_emerged[:2]}")
            except Exception as e:
                ui_print(f"  Dream {i + 1}: ERROR - {e}")
                logger.exception(f"Failed to generate dream: {e}")

        # Analyze overlap for this seed
        if len(seed_paths) >= 2:
            # Calculate average path overlap
            overlaps = []
            for j in range(len(seed_paths)):
                for k in range(j + 1, len(seed_paths)):
                    overlap = len(seed_paths[j] & seed_paths[k]) / 5.0  # depth is 5
                    overlaps.append(overlap)
            avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
            ui_print(f"  Path overlap: {avg_overlap:.1%}")

    save_experiment_summary("exp7_seed_influence", all_dreams, exp_dir)
    ui_print(f"\nGenerated {len(all_dreams)} dreams for Experiment 7")
    return all_dreams


def run_experiment_8_topology(
    dreamer: Dreamer, output_dir: Path, dreams_per_memory: int = 3
) -> list[Dream]:
    """
    Experiment 8: Graph Topology Effects

    Compare dreams seeded from hub memories (highly connected) vs
    peripheral memories (isolated/few connections).
    """
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT 8: Graph Topology Effects")
    ui_print("=" * 60)

    all_dreams: list[Dream] = []
    exp_dir = output_dir / "exp8_topology"
    exp_dir.mkdir(exist_ok=True)

    graph = dreamer.memory_graph

    # Analyze connectivity
    hubs, peripherals = analyze_graph_connectivity(dreamer)

    ui_print(f"\nGraph topology:")
    ui_print(f"  Total memories: {len(graph.elements)}")
    ui_print(f"  Total edges: {len(graph.edges)}")

    # Count edges for hubs
    edge_counts: dict[str, int] = {}
    for edge in graph.edges.values():
        edge_counts[edge.source_id] = edge_counts.get(edge.source_id, 0) + 1
        edge_counts[edge.target_id] = edge_counts.get(edge.target_id, 0) + 1

    ui_print(f"\nHub memories (most connected):")
    for hub_id in hubs:
        count = edge_counts.get(hub_id, 0)
        mem = graph.elements.get(hub_id)
        preview = (
            mem.content[:50].encode("ascii", "replace").decode("ascii") + "..."
            if mem
            else "N/A"
        )
        ui_print(f"  {hub_id[:8]}: {count} edges - {preview}")

    ui_print(f"\nPeripheral memories (least connected):")
    for per_id in peripherals:
        count = edge_counts.get(per_id, 0)
        mem = graph.elements.get(per_id)
        preview = (
            mem.content[:50].encode("ascii", "replace").decode("ascii") + "..."
            if mem
            else "N/A"
        )
        ui_print(f"  {per_id[:8]}: {count} edges - {preview}")

    # Generate dreams from hubs
    ui_print(f"\n--- Dreams from HUB memories ---")
    hub_dreams: list[Dream] = []
    for hub_id in hubs:
        for i in range(dreams_per_memory):
            config = DreamConfig(
                seed_selection=SeedSelection.RANDOM,
                traversal_strategy=TraversalStrategy.RANDOM_JUMP,
                depth=5,
                narrative_style=NarrativeStyle.FRAGMENT,
            )

            try:
                dream = dreamer.dream(config, fixed_seed_id=hub_id)
                all_dreams.append(dream)
                hub_dreams.append(dream)
                save_dream(dream, exp_dir, f"hub_{hub_id[:8]}", i)
                ui_print(
                    f"  Hub {hub_id[:8]} dream {i + 1}: {len(dream.themes_emerged)} themes"
                )
            except Exception as e:
                ui_print(f"  Hub {hub_id[:8]} dream {i + 1}: ERROR - {e}")

    # Generate dreams from peripherals
    ui_print(f"\n--- Dreams from PERIPHERAL memories ---")
    peripheral_dreams: list[Dream] = []
    for per_id in peripherals:
        for i in range(dreams_per_memory):
            config = DreamConfig(
                seed_selection=SeedSelection.RANDOM,
                traversal_strategy=TraversalStrategy.RANDOM_JUMP,
                depth=5,
                narrative_style=NarrativeStyle.FRAGMENT,
            )

            try:
                dream = dreamer.dream(config, fixed_seed_id=per_id)
                all_dreams.append(dream)
                peripheral_dreams.append(dream)
                save_dream(dream, exp_dir, f"peripheral_{per_id[:8]}", i)
                ui_print(
                    f"  Peripheral {per_id[:8]} dream {i + 1}: {len(dream.themes_emerged)} themes"
                )
            except Exception as e:
                ui_print(f"  Peripheral {per_id[:8]} dream {i + 1}: ERROR - {e}")

    # Summary comparison
    ui_print(f"\n--- Summary ---")
    if hub_dreams:
        avg_hub_themes = sum(len(d.themes_emerged) for d in hub_dreams) / len(
            hub_dreams
        )
        ui_print(
            f"  Hub dreams: {len(hub_dreams)} generated, avg {avg_hub_themes:.1f} themes"
        )
    if peripheral_dreams:
        avg_per_themes = sum(len(d.themes_emerged) for d in peripheral_dreams) / len(
            peripheral_dreams
        )
        ui_print(
            f"  Peripheral dreams: {len(peripheral_dreams)} generated, avg {avg_per_themes:.1f} themes"
        )

    save_experiment_summary("exp8_topology", all_dreams, exp_dir)
    ui_print(f"\nGenerated {len(all_dreams)} dreams for Experiment 8")
    return all_dreams


def run_experiment_9_ordering(
    dreamer: Dreamer,
    output_dir: Path,
) -> list[Dream]:
    """
    Experiment 9: Memory Ordering Effects

    Test whether the order of memories in the prompt affects the dream.
    Uses the same 5 memories in different orderings to see if the first
    memory dominates the narrative themes.
    """
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT 9: Memory Ordering Effects")
    ui_print("=" * 60)

    all_dreams: list[Dream] = []
    exp_dir = output_dir / "exp9_ordering"
    exp_dir.mkdir(exist_ok=True)

    graph = dreamer.memory_graph

    # Pick 5 random memories to use as our fixed set
    import random as rand

    all_ids = list(graph.elements.keys())
    rand.shuffle(all_ids)
    fixed_memories = all_ids[:5]

    ui_print(f"\nFixed memory set (5 memories):")
    for i, mem_id in enumerate(fixed_memories):
        mem = graph.elements.get(mem_id)
        if mem:
            preview = mem.content[:60].encode("ascii", "replace").decode("ascii")
            ui_print(f"  {i+1}. {mem_id[:8]}: {preview}...")

    # Generate dreams with each memory as the "first" position
    ui_print(f"\n--- Testing each memory as first position ---")

    results: list[tuple[str, list[str]]] = []  # (first_mem_id, themes)

    for rotation in range(5):
        # Rotate the list so a different memory is first
        rotated = fixed_memories[rotation:] + fixed_memories[:rotation]
        first_mem_id = rotated[0]

        ui_print(f"\nRotation {rotation + 1}: First memory = {first_mem_id[:8]}...")

        try:
            # Use dream_with_fixed_path to generate narrative for this ordering
            dream = dreamer.dream_with_fixed_path(rotated, NarrativeStyle.FRAGMENT)
            all_dreams.append(dream)
            save_dream(dream, exp_dir, f"rotation_{rotation}", 0)

            results.append((first_mem_id, dream.themes_emerged))
            ui_print(f"  Themes: {dream.themes_emerged[:3]}")
        except Exception as e:
            ui_print(f"  ERROR - {e}")
            logger.exception(f"Failed to generate dream: {e}")

    # Analysis: Do themes correlate with which memory is first?
    ui_print(f"\n--- Analysis ---")
    ui_print("Same 5 memories, different order. Do themes follow the first memory?")

    for first_id, themes in results:
        mem = graph.elements.get(first_id)
        preview = (
            mem.content[:40].encode("ascii", "replace").decode("ascii")
            if mem
            else "N/A"
        )
        ui_print(f"  First: {preview}...")
        ui_print(f"    -> Themes: {themes[:2]}")

    save_experiment_summary("exp9_ordering", all_dreams, exp_dir)
    ui_print(f"\nGenerated {len(all_dreams)} dreams for Experiment 9")
    return all_dreams


def run_experiment_10_dream_modes(
    dreamer: Dreamer,
    output_dir: Path,
) -> list[Dream]:
    """
    Experiment 10: Dream Modes

    Test the three purpose-driven dream modes:
    - TODAY: Consolidate memories since a given time
    - BIZARRE: Surreal contrast-seeking dreams
    - CONNECT: Find connections between memories
    """
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT 10: Dream Modes")
    ui_print("=" * 60)

    all_dreams: list[Dream] = []
    exp_dir = output_dir / "exp10_modes"
    exp_dir.mkdir(exist_ok=True)

    # Test TODAY mode
    ui_print("\n--- Mode: TODAY ---")
    try:
        # Get the oldest memory timestamp and use something slightly before the median
        # This simulates "today" for this test data
        timestamps = [mem.timestamp for mem in dreamer.memory_graph.elements.values()]
        timestamps.sort()
        # Use a timestamp that captures roughly the last 20% of memories
        cutoff_idx = int(len(timestamps) * 0.8)
        since_time = timestamps[cutoff_idx] if timestamps else datetime.now()
        ui_print(f"  Using timestamp cutoff: {since_time}")

        dream = dreamer.dream_mode(DreamMode.TODAY, since_timestamp=since_time, depth=5)
        all_dreams.append(dream)
        save_dream(dream, exp_dir, "today", 0)
        ui_print(f"  Generated: {len(dream.narrative)} chars")
        ui_print(f"  Themes: {dream.themes_emerged[:3]}")
        ui_print(f"  Memories used: {len(dream.traversal_path)}")
    except Exception as e:
        ui_print(f"  ERROR - {e}")
        logger.exception(f"Failed to generate TODAY dream: {e}")

    # Test BIZARRE mode
    ui_print("\n--- Mode: BIZARRE ---")
    try:
        dream = dreamer.dream_mode(DreamMode.BIZARRE, depth=5)
        all_dreams.append(dream)
        save_dream(dream, exp_dir, "bizarre", 0)
        ui_print(f"  Generated: {len(dream.narrative)} chars")
        ui_print(f"  Themes: {dream.themes_emerged[:3]}")
    except Exception as e:
        ui_print(f"  ERROR - {e}")
        logger.exception(f"Failed to generate BIZARRE dream: {e}")

    # Test CONNECT mode
    ui_print("\n--- Mode: CONNECT ---")
    try:
        dream = dreamer.dream_mode(DreamMode.CONNECT, depth=5)
        all_dreams.append(dream)
        save_dream(dream, exp_dir, "connect", 0)
        ui_print(f"  Generated: {len(dream.narrative)} chars")
        ui_print(f"  Themes: {dream.themes_emerged[:3]}")
        ui_print(f"  Discovered connections: {len(dream.discovered_connections)}")
        for conn in dream.discovered_connections:
            ui_print(
                f"    - {conn.edge_type}: {conn.source_id[:8]} -> {conn.target_id[:8]}"
            )
            ui_print(f"      Reason: {conn.reasoning[:60]}...")
    except Exception as e:
        ui_print(f"  ERROR - {e}")
        logger.exception(f"Failed to generate CONNECT dream: {e}")

    save_experiment_summary("exp10_modes", all_dreams, exp_dir)
    ui_print(f"\nGenerated {len(all_dreams)} dreams for Experiment 10")
    return all_dreams


def run_experiment_6_qualitative_review(
    dreamer: Dreamer, output_dir: Path, num_dreams: int = 10
) -> list[Dream]:
    """
    Experiment 6: Qualitative Review Session

    Generate a collection of "best" dreams for manual review.
    Uses optimal settings discovered from other experiments.
    """
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT 6: Qualitative Review (Best Dreams)")
    ui_print("=" * 60)

    all_dreams: list[Dream] = []
    exp_dir = output_dir / "exp6_review"
    exp_dir.mkdir(exist_ok=True)

    # Mix of configurations that tend to produce good dreams
    configs = [
        DreamConfig(
            SeedSelection.EMOTIONAL,
            TraversalStrategy.RANDOM_JUMP,
            5,
            NarrativeStyle.FRAGMENT,
        ),
        DreamConfig(
            SeedSelection.EMOTIONAL,
            TraversalStrategy.CONTRAST_SEEKING,
            5,
            NarrativeStyle.POETIC,
        ),
        DreamConfig(
            SeedSelection.RECENT,
            TraversalStrategy.SEMANTIC_DRIFT,
            5,
            NarrativeStyle.SENSORY,
        ),
        DreamConfig(
            SeedSelection.EMOTIONAL,
            TraversalStrategy.RANDOM_JUMP,
            7,
            NarrativeStyle.STREAM,
        ),
        DreamConfig(
            SeedSelection.UNPROCESSED,
            TraversalStrategy.CONTRAST_SEEKING,
            5,
            NarrativeStyle.FRAGMENT,
        ),
        DreamConfig(
            SeedSelection.EMOTIONAL,
            TraversalStrategy.RECENCY_WEIGHTED,
            5,
            NarrativeStyle.POETIC,
        ),
        DreamConfig(
            SeedSelection.RECENT,
            TraversalStrategy.RANDOM_JUMP,
            5,
            NarrativeStyle.SENSORY,
        ),
        DreamConfig(
            SeedSelection.EMOTIONAL,
            TraversalStrategy.SEMANTIC_DRIFT,
            7,
            NarrativeStyle.FRAGMENT,
        ),
        DreamConfig(
            SeedSelection.RANDOM,
            TraversalStrategy.CONTRAST_SEEKING,
            5,
            NarrativeStyle.STREAM,
        ),
        DreamConfig(
            SeedSelection.EMOTIONAL,
            TraversalStrategy.RANDOM_JUMP,
            5,
            NarrativeStyle.POETIC,
        ),
    ]

    for i, config in enumerate(configs[:num_dreams]):
        ui_print(f"\n--- Dream {i + 1}/{num_dreams} ---")
        ui_print(f"  Config: {config}")

        try:
            dream = dreamer.dream(config)
            all_dreams.append(dream)
            save_dream(dream, exp_dir, "review", i)
            ui_print(f"  Generated: {len(dream.narrative)} chars")
            ui_print(f"  Themes: {dream.themes_emerged}")
        except Exception as e:
            ui_print(f"  ERROR - {e}")
            logger.exception(f"Failed to generate dream: {e}")

    save_experiment_summary("exp6_review", all_dreams, exp_dir)
    ui_print(f"\nGenerated {len(all_dreams)} dreams for qualitative review")
    return all_dreams


def main():
    """Main entry point for the dreams experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Run dreams experiment")
    parser.add_argument(
        "--conversation",
        type=str,
        default=DEFAULT_CONVERSATION,
        help="Conversation ID prefix to load",
    )
    parser.add_argument(
        "--conversations-dir",
        type=str,
        default="conversations",
        help="Directory containing conversation files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: dreams_output in experiment dir)",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        default=None,
        help="Run specific experiment (1-10). If not specified, runs all.",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    ui_print("=" * 60)
    ui_print("DREAMS EXPERIMENT")
    ui_print("=" * 60)
    ui_print(f"Conversation: {args.conversation}")
    ui_print(f"Output directory: {output_dir}")

    # Load agent data
    ui_print("\nLoading agent data...")
    persistence = ConversationPersistence(conversations_dir=args.conversations_dir)
    agent_data = persistence.load_agent_data(
        args.conversation, use_individual_formatting=True
    )

    if not isinstance(agent_data.memory, DagMemoryManager):
        raise ValueError("Dreams experiment requires DAG memory type")
    memory_graph = agent_data.memory.get_memory_graph()
    ui_print(
        f"Loaded {len(memory_graph.elements)} memories, {len(memory_graph.edges)} edges"
    )

    # Create LLM and Dreamer
    ui_print("\nInitializing LLM...")
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    dreamer = Dreamer(memory_graph, llm, model)
    ui_print("Dreamer initialized")

    # Run experiments
    all_dreams: list[Dream] = []

    if args.experiment is None or args.experiment == 1:
        dreams = run_experiment_1_traversal_comparison(dreamer, output_dir)
        all_dreams.extend(dreams)

    if args.experiment is None or args.experiment == 2:
        dreams = run_experiment_2_narrative_comparison(dreamer, output_dir)
        all_dreams.extend(dreams)

    if args.experiment is None or args.experiment == 3:
        dreams = run_experiment_3_depth_analysis(dreamer, output_dir)
        all_dreams.extend(dreams)

    if args.experiment is None or args.experiment == 4:
        dreams = run_experiment_4_seed_selection(dreamer, output_dir)
        all_dreams.extend(dreams)

    if args.experiment is None or args.experiment == 5:
        dreams = run_experiment_5_graph_structure(dreamer, output_dir)
        all_dreams.extend(dreams)

    if args.experiment is None or args.experiment == 6:
        dreams = run_experiment_6_qualitative_review(dreamer, output_dir)
        all_dreams.extend(dreams)

    if args.experiment is None or args.experiment == 7:
        dreams = run_experiment_7_seed_influence(dreamer, output_dir)
        all_dreams.extend(dreams)

    if args.experiment is None or args.experiment == 8:
        dreams = run_experiment_8_topology(dreamer, output_dir)
        all_dreams.extend(dreams)

    if args.experiment is None or args.experiment == 9:
        dreams = run_experiment_9_ordering(dreamer, output_dir)
        all_dreams.extend(dreams)

    if args.experiment is None or args.experiment == 10:
        dreams = run_experiment_10_dream_modes(dreamer, output_dir)
        all_dreams.extend(dreams)

    # Final summary
    ui_print("\n" + "=" * 60)
    ui_print("EXPERIMENT COMPLETE")
    ui_print("=" * 60)
    ui_print(f"Total dreams generated: {len(all_dreams)}")
    ui_print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
