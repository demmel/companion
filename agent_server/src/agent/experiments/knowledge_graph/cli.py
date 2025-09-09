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
from typing import List, Optional
import click

from agent.conversation_persistence import ConversationPersistence
from agent.llm import create_llm, SupportedModel
from agent.experiments.knowledge_graph.knowledge_graph_builder import (
    ValidatedKnowledgeGraphBuilder,
)
from agent.experiments.knowledge_graph.relationship_schema_bank import (
    RelationshipSchemaBank,
)
from agent.experiments.knowledge_graph.knowledge_graph_querying import (
    GraphQuery,
    KnowledgeGraphQuerying,
)
from agent.experiments.knowledge_graph.graph_maintenance import (
    GraphMaintenanceSystem,
)
from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.state import State, Value, Priority
from datetime import datetime


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


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """Knowledge Graph Experiment CLI"""
    # Always show INFO level for user feedback, DEBUG only in verbose mode
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,  # Override existing configuration
    )


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

    click.echo(f"🏗️  Building Knowledge Graph")
    click.echo("=" * 50)
    click.echo(f"Conversation: {conversation}")
    click.echo(f"Triggers to process: {triggers or 'all available'}")
    click.echo(f"Output file: {output}")

    # Load conversation data
    persistence = ConversationPersistence()
    trigger_history, state, initial_exchange = persistence.load_conversation(
        conversation
    )

    if state is None or initial_exchange is None:
        click.echo("❌ Could not load conversation data", err=True)
        return

    click.echo(
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

    # Initialize graph maintenance
    graph_maintenance = GraphMaintenanceSystem(llm, model, initial_state)

    click.echo(f"✅ Setup complete! Initial state (mood: {initial_state.current_mood})")

    # Process triggers
    all_triggers = trigger_history.get_all_entries()
    triggers_to_process = all_triggers[:triggers] if triggers else all_triggers

    click.echo(f"\n🔄 Processing {len(triggers_to_process)} triggers...")

    # Initialize cProfile if requested
    if profile:
        import cProfile
        import pstats
        import tempfile
        import os

        pr = cProfile.Profile()
        pr.enable()

    def process_triggers():
        previous_trigger = None
        successful = 0

        with click.progressbar(
            triggers_to_process,
            label="Building knowledge graph",
            show_eta=True,
            show_percent=True,
            show_pos=True,
            width=80,
        ) as bar:
            for i, trigger in enumerate(bar):
                success = kg_builder.process_trigger_incremental(
                    trigger, previous_trigger
                )
                if success:
                    successful += 1

                # Run maintenance every 5 triggers or on the last trigger
                if (i + 1) % 5 == 0 or i == len(triggers_to_process) - 1:
                    click.echo(f"\n🔧 Running graph maintenance...")
                    maintenance_results = graph_maintenance.analyze_graph(
                        kg_builder.graph
                    )
                    if len(maintenance_results.contradictions) > 0:
                        click.echo(
                            f"   Found {len(maintenance_results.contradictions)} contradictions"
                        )
                    if len(maintenance_results.logical_leaps) > 0:
                        click.echo(
                            f"   Found {len(maintenance_results.logical_leaps)} logical leaps"
                        )
                    if len(maintenance_results.semantic_refinements) > 0:
                        click.echo(
                            f"   Found {len(maintenance_results.semantic_refinements)} refinement opportunities"
                        )

                kg_builder.save_graph(output)
                relationship_bank.save_bank()
                previous_trigger = trigger

        return successful

    successful = process_triggers()

    llm.log_stats_summary()

    # Process profiling results if enabled
    if profile:
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

    # Show performance breakdown
    if profile:
        click.echo(f"\n⚡ cProfile Performance Analysis:")
        click.echo("=" * 60)

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
        import os

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
@click.argument("query_text")
@click.option(
    "--max-context",
    "-m",
    default=2000,
    help="Maximum context length for graph traversal",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(
        ["structured", "narrative", "bullet_points", "confidence_weighted"]
    ),
    default="structured",
    help="Context format to use",
)
@click.option(
    "--conversation", "-c", default="baseline", help="Conversation for state context"
)
def query(
    graph_file: str, query_text: str, max_context: int, format: str, conversation: str
):
    """Query a pre-built knowledge graph"""

    click.echo(f"🔍 Querying Knowledge Graph")
    click.echo("=" * 50)
    click.echo(f"Graph file: {graph_file}")
    click.echo(f"Query: '{query_text}'")
    click.echo(f"Max context: {max_context} chars")
    click.echo(f"Format: {format}")

    # Load the saved graph
    try:
        from agent.experiments.knowledge_graph.knowledge_graph_prototype import (
            KnowledgeExperienceGraph,
        )

        graph = KnowledgeExperienceGraph.load_from_file(graph_file)
        click.echo(
            f"✅ Loaded knowledge graph: {len(graph.get_all_nodes())} nodes, {len(graph.get_all_relationships())} relationships"
        )
    except Exception as e:
        click.echo(f"❌ Failed to load graph: {e}", err=True)
        return

    # Load conversation data for state context
    try:
        persistence = ConversationPersistence()
        trigger_history, state, initial_exchange = persistence.load_conversation(
            conversation
        )
        if state is None:
            click.echo("❌ Could not load conversation data", err=True)
            return
        click.echo(f"✅ Loaded conversation context")
    except Exception as e:
        click.echo(f"❌ Failed to load conversation: {e}", err=True)
        return

    # Initialize LLM and querying system
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    querying = KnowledgeGraphQuerying(graph, llm, model, state)

    # Create a fake recent triggers context for the query system
    recent_triggers = trigger_history.get_all_entries()[-3:]  # Last 3 for context

    click.echo(f"\n🧠 Processing Query...")

    # Determine what context is needed
    query_determination = querying.determine_context_needs(query_text, recent_triggers)

    if query_determination is None or not query_determination.should_query:
        click.echo("❌ Query determined that no graph context is needed")
        click.echo(
            f"Reasoning: {query_determination.reasoning if query_determination else 'Unknown'}"
        )
        return

    click.echo(f"✅ Query analysis: {query_determination.reasoning}")
    if query_determination.query:
        click.echo(f"Focus entities: {query_determination.query.focus_entities}")
        click.echo(
            f"Relationship types: {query_determination.query.relationship_types}"
        )

    # Execute the graph query using the querying system's built-in method
    # This bypasses the context builder since they use incompatible GraphQuery objects
    formatted_context = querying.construct_agent_context(query_text, recent_triggers)

    # Truncate to max_context if needed
    if len(formatted_context) > max_context:
        formatted_context = formatted_context[:max_context] + "..."

    # Get context details for breakdown (this may not work perfectly but shows concept)
    context = querying.execute_graph_query(
        query_determination.query or GraphQuery(context_purpose="general")
    )

    click.echo(f"\n📝 Context Result ({format.upper()}):")
    click.echo("=" * 60)
    click.echo(formatted_context)
    click.echo("=" * 60)
    click.echo(f"Context length: {len(formatted_context)} characters")

    # Show context breakdown
    click.echo(f"\n📊 Context Breakdown:")
    click.echo(f"   Entities found: {len(context.relevant_entities)}")
    click.echo(f"   Relationships: {len(context.relevant_relationships)}")
    click.echo(f"   Recent experiences: {len(context.recent_experiences)}")
    click.echo(f"   Patterns identified: {len(context.patterns_and_insights)}")
    click.echo(f"   Emotional context: {len(context.emotional_context)}")
    click.echo(f"   Commitments: {len(context.commitments_and_promises)}")

    click.echo(f"\n✅ Query completed!")


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
    "--similarity-threshold",
    "-t",
    default=0.85,
    help="Cosine similarity threshold for duplicate detection (0.0-1.0)",
)
@click.option(
    "--auto-merge",
    "-a",
    is_flag=True,
    help="Automatically merge high-confidence duplicates without review",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    help="Show what would be merged without making changes",
)
def deduplicate(
    graph_file: str, similarity_threshold: float, auto_merge: bool, dry_run: bool
):
    """Scan knowledge graph for duplicates using embedding similarity"""

    click.echo(f"🔍 Knowledge Graph Deduplication")
    click.echo("=" * 50)
    click.echo(f"Graph file: {graph_file}")
    click.echo(f"Similarity threshold: {similarity_threshold}")
    click.echo(f"Auto-merge: {auto_merge}")
    click.echo(f"Dry run: {dry_run}")

    # Load the knowledge graph
    try:
        from agent.experiments.knowledge_graph.knowledge_graph_prototype import (
            KnowledgeExperienceGraph,
        )
        from agent.memory.embedding_service import get_embedding_service

        graph = KnowledgeExperienceGraph.load_from_file(graph_file)
        click.echo(
            f"✅ Loaded graph: {len(graph.get_all_nodes())} nodes, {len(graph.get_all_relationships())} relationships"
        )

        embedding_service = get_embedding_service()
        click.echo(f"✅ Loaded embedding service")

    except Exception as e:
        click.echo(f"❌ Failed to load graph or embedding service: {e}", err=True)
        return

    # Scan for node duplicates using embeddings
    click.echo(f"\n🔍 Scanning for duplicate nodes...")

    node_duplicates = []
    nodes = list(graph.get_all_nodes())

    # Group nodes by type for more efficient comparison
    nodes_by_type = {}
    for node in nodes:
        node_type = node.node_type.value
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append(node)

    duplicate_pairs = []
    total_comparisons = 0

    # Compare nodes within each type
    for node_type, type_nodes in nodes_by_type.items():
        click.echo(f"   Checking {len(type_nodes)} {node_type} nodes...")

        for i in range(len(type_nodes)):
            for j in range(i + 1, len(type_nodes)):
                node1, node2 = type_nodes[i], type_nodes[j]
                total_comparisons += 1

                # Generate embeddings if missing
                if node1.embedding is None:
                    embedding_text = (
                        f"[{node1.node_type.value}] {node1.name}: {node1.description}"
                    )
                    node1.embedding = embedding_service.encode(embedding_text)

                if node2.embedding is None:
                    embedding_text = (
                        f"[{node2.node_type.value}] {node2.name}: {node2.description}"
                    )
                    node2.embedding = embedding_service.encode(embedding_text)

                # Calculate similarity
                similarity = embedding_service.cosine_similarity(
                    node1.embedding, node2.embedding
                )

                if similarity >= similarity_threshold:
                    duplicate_pairs.append((node1, node2, similarity))

    click.echo(f"   Completed {total_comparisons} comparisons")

    if not duplicate_pairs:
        click.echo("✅ No duplicate nodes found!")
        return

    click.echo(f"\n📋 Found {len(duplicate_pairs)} potential duplicate pairs:")

    for i, (node1, node2, similarity) in enumerate(duplicate_pairs):
        click.echo(f"\n--- Duplicate {i+1} (similarity: {similarity:.3f}) ---")
        click.echo(f"Node 1: {node1.name} ({node1.node_type.value})")
        click.echo(f"  Description: {node1.description[:100]}...")
        click.echo(f"  Created: {node1.created_at}")
        click.echo(f"  Access count: {node1.access_count}")

        click.echo(f"Node 2: {node2.name} ({node2.node_type.value})")
        click.echo(f"  Description: {node2.description[:100]}...")
        click.echo(f"  Created: {node2.created_at}")
        click.echo(f"  Access count: {node2.access_count}")

    # Scan for relationship duplicates
    click.echo(f"\n🔍 Scanning for duplicate relationships...")

    relationships = list(graph.get_all_relationships())
    relationship_groups = {}

    # Group relationships by source-target-type triplets
    for rel in relationships:
        key = (rel.source_node_id, rel.target_node_id, rel.relationship_type)
        if key not in relationship_groups:
            relationship_groups[key] = []
        relationship_groups[key].append(rel)

    duplicate_relationships = [
        (key, rels) for key, rels in relationship_groups.items() if len(rels) > 1
    ]

    if duplicate_relationships:
        click.echo(
            f"\n📋 Found {len(duplicate_relationships)} sets of duplicate relationships:"
        )

        for i, ((source_id, target_id, rel_type), rels) in enumerate(
            duplicate_relationships
        ):
            source_node = graph.get_node(source_id)
            target_node = graph.get_node(target_id)

            click.echo(f"\n--- Duplicate Relationship Set {i+1} ---")
            click.echo(
                f"Relationship: {source_node.name if source_node else source_id} --[{rel_type}]--> {target_node.name if target_node else target_id}"
            )
            click.echo(f"Duplicate count: {len(rels)}")

            for j, rel in enumerate(rels):
                click.echo(
                    f"  {j+1}. Created: {rel.created_at}, Confidence: {rel.confidence}, Strength: {rel.strength}"
                )

    else:
        click.echo("✅ No duplicate relationships found!")

    if dry_run:
        click.echo(
            f"\n💡 This was a dry run. Use --auto-merge to actually merge duplicates."
        )
    elif auto_merge:
        click.echo(f"\n⚠️  Auto-merge functionality not yet implemented")
        click.echo("Would merge duplicate nodes and relationships here")
    else:
        click.echo(
            f"\n💡 Use --auto-merge to automatically merge duplicates, or --dry-run to preview changes"
        )

    click.echo(f"\n✅ Deduplication scan completed!")


if __name__ == "__main__":
    cli()
