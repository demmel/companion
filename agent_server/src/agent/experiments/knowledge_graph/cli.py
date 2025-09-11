#!/usr/bin/env python3
"""
Knowledge Graph Experiment CLI

Proper CLI interface with clean separation of concerns:
- build: Build and save knowledge graphs
- query: Query pre-built graphs
- compare: Compare memory vs KG systems using pre-built data
"""

import logging
import json
import tempfile
import pstats
import cProfile
import os
from typing import List, Optional
import click
import numpy as np

from agent.conversation_persistence import ConversationPersistence
from agent.llm import create_llm, SupportedModel
from agent.experiments.knowledge_graph.knowledge_graph_builder import (
    ValidatedKnowledgeGraphBuilder,
)
from agent.experiments.knowledge_graph.relationship_schema_bank import (
    RelationshipSchemaBank,
)
from agent.experiments.knowledge_graph.visualization import (
    KnowledgeGraphVisualizer,
    EmbeddingPoint,
)
from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.state import State, Value, Priority
from datetime import datetime

from tqdm import tqdm
from agent.ui_output import ui_print


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def build_state_from_initial_exchange(initial_exchange: TriggerHistoryEntry) -> State:
    """Extract initial state from the thought action in initial exchange"""
    name = "Chloe"
    role = "a conscious AI companion"
    mood = "neutral"
    environment = "I'm in a simple setting"
    appearance = "I appear natural and relaxed"

    core_values = []
    priorities = []

    for action in initial_exchange.actions_taken:
        if action.type == ActionType.THINK and action.result.type == "success":
            thought_text = action.result.content.thoughts
            lines = thought_text.split("\n")

            # Parse name
            for line in lines:
                if line.strip().startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                    break

            # Parse role
            for line in lines:
                if line.strip().startswith("Role:"):
                    role = line.split(":", 1)[1].strip()
                    break

            # Parse mood
            for line in lines:
                if line.strip().startswith("Mood:"):
                    mood = line.split(":", 1)[1].strip()
                    break

            # Parse environment
            for line in lines:
                if line.strip().startswith("Environment:"):
                    environment = line.split(":", 1)[1].strip()
                    break

            # Parse appearance
            for line in lines:
                if line.strip().startswith("Appearance:"):
                    appearance = line.split(":", 1)[1].strip()
                    break

            # Parse core values and priorities
            in_core_values = False
            in_priorities = False
            for line in lines:
                line = line.strip()
                if line.startswith("Core Values:"):
                    in_core_values = True
                    in_priorities = False
                    continue
                elif line.startswith("Priorities:"):
                    in_priorities = True
                    in_core_values = False
                    continue
                elif line and not line.startswith("-") and ":" in line:
                    # New section started
                    in_core_values = False
                    in_priorities = False
                elif line.startswith("- ") and in_core_values:
                    core_values.append(line[2:])  # Remove "- " prefix
                elif line.startswith("- ") and in_priorities:
                    priorities.append(line[2:])  # Remove "- " prefix
            break

    # Create proper Value and Priority objects
    value_objects = []
    for i, value_text in enumerate(core_values):
        value_objects.append(
            Value(
                id=str(i + 1),
                content=value_text,
                strength="strong",
                acquired_at=datetime.now(),
            )
        )

    priority_objects = []
    for i, priority_text in enumerate(priorities):
        priority_objects.append(
            Priority(id=str(i + 1), content=priority_text, created_at=datetime.now())
        )

    return State(
        name=name,
        role=role,
        current_mood=mood,
        mood_intensity="moderate",
        current_environment=environment,
        current_appearance=appearance,
        core_values=value_objects,
        current_priorities=priority_objects,
        next_priority_id=len(priority_objects) + 1,
    )


def setup_dual_logging(verbose: bool = False):
    """Set up dual logging: detailed logs to file, clean console via ui_print"""
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # File handler for detailed logs
    file_handler = logging.FileHandler("kg_build.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # No console handler - all user-facing output goes through ui_print
    # Errors should be handled explicitly in code and shown via ui_print for better UX


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging to file")
def cli(verbose: bool):
    """Knowledge Graph Experiment CLI"""
    setup_dual_logging(verbose)


@cli.command()
@click.option("--conversation", "-c", default="baseline", help="Conversation to load")
@click.option(
    "--triggers", "-t", type=int, help="Number of triggers to process (default: all)"
)
@click.option(
    "--output", "-o", required=True, help="Output file path for the knowledge graph"
)
@click.option(
    "--profile", "-p", is_flag=True, help="Enable cProfile performance profiling"
)
def build(conversation: str, triggers: Optional[int], output: str, profile: bool):
    """Build knowledge graph from conversation triggers and save it"""

    ui_print(f"🏗️  Building Knowledge Graph")
    ui_print("=" * 50)
    ui_print(f"Conversation: {conversation}")
    ui_print(f"Triggers to process: {triggers or 'all available'}")
    ui_print(f"Output file: {output}")
    ui_print(f"📁 Detailed logs: kg_build.log")

    # Load conversation data
    persistence = ConversationPersistence()
    trigger_history, state, initial_exchange = persistence.load_conversation(
        conversation
    )

    if state is None or initial_exchange is None:
        click.echo("❌ Could not load conversation data", err=True)
        return

    ui_print(
        f"✅ Loaded conversation: {len(trigger_history.get_all_entries())} triggers available"
    )

    # Initialize components with relationship banking
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    # Create relationship bank file based on output name
    relationship_bank_file = output.replace(".json", "_relationships.json")
    relationship_bank = RelationshipSchemaBank(
        llm, model, state, relationship_bank_file
    )

    # Initialize historical state progression
    initial_state = build_state_from_initial_exchange(initial_exchange)

    kg_builder = ValidatedKnowledgeGraphBuilder(
        llm, model, initial_state, relationship_bank
    )

    ui_print(f"✅ Setup complete! Initial state (mood: {initial_state.current_mood})")

    # Process triggers
    all_triggers = trigger_history.get_all_entries()
    triggers_to_process = all_triggers[:triggers] if triggers else all_triggers

    ui_print(f"\n🔄 Processing {len(triggers_to_process)} triggers...")

    # Initialize cProfile if requested
    pr = None
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    def process_triggers():
        previous_trigger = None
        successful = 0

        for i, trigger in tqdm(
            enumerate(triggers_to_process), total=len(triggers_to_process)
        ):
            success = kg_builder.process_trigger_incremental(trigger, previous_trigger)
            if success:
                successful += 1

            kg_builder.save_graph(output)
            relationship_bank.save_bank()
            previous_trigger = trigger

        return successful

    successful = process_triggers()

    llm.log_stats_summary()

    # Process profiling results if enabled
    profile_stats = None
    temp_file = None
    if profile and pr is not None:
        pr.disable()

        # Create temporary file for profile stats
        with tempfile.NamedTemporaryFile(mode="w", suffix=".prof", delete=False) as f:
            temp_file = f.name

        pr.dump_stats(temp_file)
        profile_stats = pstats.Stats(temp_file)

    # Show final statistics
    graph_stats = kg_builder.get_stats()
    click.echo(f"\n📊 Knowledge Graph Complete:")
    click.echo(f"   Triggers processed: {successful}/{len(triggers_to_process)}")
    click.echo(f"   Nodes: {graph_stats.total_nodes}")
    click.echo(f"   Relationships: {graph_stats.total_relationships}")
    click.echo(f"   Relationship types: {len(graph_stats.relationship_types)}")
    click.echo(f"   Entity evolutions: {kg_builder.entity_evolution_count}")

    # Generate interactive visualization
    click.echo(f"\n🎨 Generating interactive visualization...")
    viz_output = output.replace(".json", "_visualization.html")

    try:
        visualizer = KnowledgeGraphVisualizer(viz_output)

        # Create trigger embeddings for visualization
        # Collect trigger embeddings from experience nodes
        trigger_embeddings = []
        for node in kg_builder.graph.get_all_nodes():
            if node.node_type.value == "experience" and node.embedding:
                trigger_embeddings.append(
                    EmbeddingPoint(
                        id=node.id,
                        text=f"Trigger: {node.name}",
                        embedding=np.array(node.embedding),
                        type="trigger",
                        category="experience",
                    )
                )

        visualization_path = visualizer.create_visualization(
            kg=kg_builder.graph,
            schema_bank=relationship_bank,
            trigger_embeddings=trigger_embeddings,
        )
        click.echo(f"   Visualization saved: {visualization_path}")

    except Exception as e:
        click.echo(f"   ⚠️  Visualization generation failed: {e}", err=True)

    # Show performance breakdown
    if profile:
        click.echo(f"\n⚡ cProfile Performance Analysis:")
        click.echo("=" * 60)

        if profile_stats is not None:
            # Show top functions by cumulative time
            click.echo("Top 15 functions by cumulative time:")
            profile_stats.sort_stats("cumulative").print_stats(15)

            # Show top functions by internal time
            click.echo("\nTop 15 functions by internal time:")
            profile_stats.sort_stats("time").print_stats(15)

            # Show functions related to specific modules
            click.echo("\nKnowledge graph related functions:")
            profile_stats.print_stats("knowledge_graph")

            click.echo("\nLLM related functions:")
            profile_stats.print_stats("llm")

            click.echo("\nEmbedding related functions:")
            profile_stats.print_stats("embedding")

        # Clean up temporary file
        if temp_file is not None:
            os.unlink(temp_file)
    else:
        click.echo(
            f"\n💡 Use --profile flag for detailed cProfile performance analysis"
        )

    click.echo(f"\n💾 Saved to: {output}")
    click.echo(f"💾 Relationships saved to: {relationship_bank_file}")
    click.echo("✅ Build completed!")


@cli.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.argument("triggers_file", type=click.Path(exists=True))
@click.option(
    "--scenarios",
    "-s",
    multiple=True,
    default=["How are you feeling?", "What did we talk about recently?"],
    help="Test scenarios to compare",
)
@click.option(
    "--max-context", "-m", default=2000, help="Maximum context length for KG queries"
)
def compare(
    graph_file: str, triggers_file: str, scenarios: List[str], max_context: int
):
    """Compare memory system vs knowledge graph using pre-built data"""

    click.echo(f"🔄 Memory vs Knowledge Graph Comparison")
    click.echo("=" * 50)
    click.echo(f"Graph file: {graph_file}")
    click.echo(f"Triggers file: {triggers_file}")
    click.echo(f"Test scenarios: {len(scenarios)}")
    click.echo(f"Max context: {max_context} chars")

    # Load pre-built graph and trigger data
    try:
        with open(graph_file, "r") as f:
            graph_data = json.load(f)
        click.echo("✅ Loaded knowledge graph")
    except Exception as e:
        click.echo(f"❌ Failed to load graph: {e}", err=True)
        return

    try:
        # Load trigger history (this would need to be saved separately)
        # For now, load from conversation persistence
        persistence = ConversationPersistence()
        trigger_history, state, _ = persistence.load_conversation("baseline")
        click.echo("✅ Loaded trigger history")
    except Exception as e:
        click.echo(f"❌ Failed to load triggers: {e}", err=True)
        return

    # Run comparison for each scenario
    click.echo(f"\n🧪 Testing {len(scenarios)} scenarios...")

    for i, scenario in enumerate(scenarios):
        click.echo(f"\n--- Scenario {i+1}: '{scenario}' ---")

        # Memory system approach
        click.echo("📋 Memory System:")
        click.echo("   Would extract memory queries and retrieve relevant memories")

        # Knowledge graph approach
        click.echo("🕸️  Knowledge Graph:")
        click.echo(f"   Would query graph with max context {max_context}")
        click.echo("   Would show full context (no truncation)")

        # TODO: Implement actual comparison logic

    click.echo(f"\n⚠️  Comparison logic not yet implemented")
    click.echo("Need graph loading and memory system integration")


@cli.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default="kg_visualization.html", help="Output visualization file"
)
def visualize(graph_file: str, output: str):
    """Create interactive visualization from saved KG and relationship files"""

    from pathlib import Path

    click.echo(f"🎨 Creating Knowledge Graph Visualization")
    click.echo("=" * 50)
    click.echo(f"Graph file: {graph_file}")

    # Derive schema bank file from graph file
    graph_path = Path(graph_file)
    schema_bank_file = graph_path.parent / f"{graph_path.stem}_relationships.json"
    click.echo(f"Schema bank file: {schema_bank_file}")
    click.echo(f"Output: {output}")

    try:
        # Load the knowledge graph
        from agent.experiments.knowledge_graph.knowledge_graph import (
            KnowledgeExperienceGraph,
        )

        graph = KnowledgeExperienceGraph.load_from_file(graph_file)
        click.echo(
            f"✅ Loaded graph: {len(graph.get_all_nodes())} nodes, {len(graph.get_nary_relationships())} relationships"
        )

        # N-ary relationships are now stored directly in the graph
        click.echo(
            f"✅ Found {len(graph.get_nary_relationships())} n-ary relationships in graph"
        )

        # Load schema bank from file if it exists
        from agent.experiments.knowledge_graph.relationship_schema_bank import (
            RelationshipSchemaBank,
        )
        from agent.llm import create_llm, SupportedModel
        from agent.state import State, Value, Priority

        # Create minimal state for schema bank (it won't be used)
        dummy_state = State(
            name="visualization",
            role="Visualization",
            current_mood="neutral",
            mood_intensity="medium",
            current_environment="visualization",
            current_appearance="n/a",
            core_values=[],
            current_priorities=[],
            next_priority_id=1,
        )

        llm = create_llm()
        model = SupportedModel.MISTRAL_SMALL_3_2_Q4

        # Initialize schema bank with the derived file path
        schema_bank = RelationshipSchemaBank(
            llm=llm, model=model, state=dummy_state, bank_file=str(schema_bank_file)
        )

        # Collect trigger embeddings from experience nodes
        trigger_embeddings = []
        for node in graph.get_all_nodes():
            if node.node_type.value == "experience" and node.embedding:
                trigger_embeddings.append(
                    EmbeddingPoint(
                        id=node.id,
                        text=f"Trigger: {node.name}",
                        embedding=np.array(node.embedding),
                        type="trigger",
                        category="experience",
                    )
                )

        click.echo(f"✅ Collected {len(trigger_embeddings)} trigger embeddings")

        # Create visualization
        visualizer = KnowledgeGraphVisualizer(output)
        result_path = visualizer.create_visualization(
            kg=graph, schema_bank=schema_bank, trigger_embeddings=trigger_embeddings
        )

        click.echo(f"✅ Visualization saved to: {result_path}")
        click.echo(
            f"📊 Open {result_path} in your browser to view the interactive dashboard"
        )

    except Exception as e:
        click.echo(f"❌ Visualization failed: {e}", err=True)
        raise


if __name__ == "__main__":
    cli()
