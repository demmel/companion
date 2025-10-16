"""
Main experiment runner for tiered memory system.

Loads existing memory data, builds tier 3 & 4 structures, performs
retrieval experiments, and outputs results for analysis.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import json
from tqdm import tqdm

from agent.conversation_persistence import ConversationPersistence
from agent.llm import create_llm, SupportedModel
from agent.state import State
from agent.ui_output import ui_print

from .models import TieredMemoryGraph, MemoryTier
from .conversation_detection import detect_conversations
from .semantic_clustering import (
    create_semantic_clusters_from_conversations,
    create_semantic_clusters_from_trigger_entries,
)
from .tiered_retrieval import retrieve_multi_tier
from .context_builder import estimate_context_tokens, format_for_llm_prompt

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TieredMemoryExperiment:
    """Main experiment orchestrator."""

    def __init__(
        self,
        conversation_prefix: str,
        conversations_dir: str = "conversations",
    ):
        """
        Initialize experiment.

        Args:
            conversation_prefix: Conversation ID prefix (e.g., "conversation_20251004_012921_639291")
            conversations_dir: Directory containing conversation files
        """
        self.conversation_prefix = conversation_prefix
        self.llm = create_llm()
        self.model = SupportedModel.MISTRAL_SMALL_3_2_Q4

        # Load agent data using conversation persistence
        logger.info(f"Loading agent data from {conversation_prefix}")
        persistence = ConversationPersistence(conversations_dir=conversations_dir)
        agent_data = persistence.load_agent_data(conversation_prefix)

        self.state = agent_data.state
        self.trigger_history = agent_data.trigger_history
        self.dag_manager = agent_data.dag_memory_manager
        self.memory_graph = self.dag_manager.get_memory_graph()

        logger.info(
            f"Loaded agent data: {len(self.memory_graph.elements)} elements, "
            f"{len(self.memory_graph.containers)} containers, "
            f"{len(self.trigger_history.entries)} trigger entries"
        )

        self.tiered_graph: Optional[TieredMemoryGraph] = None
        self.trigger_entries_dict: Dict[str, Any] = {}
        self.output_dir: Optional[Path] = None

    def build_tier_3(
        self, use_topic_detection: bool = True, output_dir: Optional[Path] = None
    ) -> None:
        """
        Build tier 3 (conversation boundaries) with incremental saving and resume support.

        Args:
            use_topic_detection: Whether to use LLM-based topic shift detection
            output_dir: Directory to save incremental progress (optional)
        """
        logger.info("Building tier 3: Conversation boundaries")

        # Get all trigger entries
        trigger_entries = self.trigger_history.get_all_entries()
        logger.info(f"Processing {len(trigger_entries)} trigger entries")

        # Build lookup dict
        self.trigger_entries_dict = {entry.entry_id: entry for entry in trigger_entries}

        # Get existing conversations for resume support
        existing_conversations = None
        if self.tiered_graph and self.tiered_graph.conversations:
            existing_conversations = list(self.tiered_graph.conversations.values())
            logger.info(
                f"Resuming from {len(existing_conversations)} existing conversations"
            )

        # Detect conversations with incremental callback
        def on_conversation_created(conversation):
            """Callback to save progress after each conversation."""
            if output_dir:
                # Initialize tiered graph if needed
                if self.tiered_graph is None:
                    self.tiered_graph = TieredMemoryGraph(
                        memory_graph_path=None, conversations={}
                    )

                # Add conversation
                self.tiered_graph.conversations[conversation.id] = conversation

                # Save incrementally
                self.save_tier_3(output_dir)

        # Detect conversations with callback and resume support
        conversations = detect_conversations(
            trigger_entries=trigger_entries,
            llm=self.llm,
            model=self.model,
            state=self.state,
            use_topic_detection=use_topic_detection,
            on_conversation_created=on_conversation_created,
            existing_conversations=existing_conversations or [],
        )

        # Update tiered graph with all conversations
        if self.tiered_graph is None:
            self.tiered_graph = TieredMemoryGraph(
                memory_graph_path=None,
                conversations={conv.id: conv for conv in conversations},
            )
        else:
            # Update with any new conversations
            for conv in conversations:
                self.tiered_graph.conversations[conv.id] = conv

        logger.info(f"Total: {len(conversations)} conversation boundaries (tier 3)")

    def save_tier_3(self, output_dir: Path) -> None:
        """Save tier 3 results to JSON."""
        if not self.tiered_graph:
            return

        tier3_file = output_dir / "tier3_conversations.json"
        logger.info(f"Saving tier 3 to {tier3_file}")

        # Serialize conversations
        conversations_data = [
            conv.model_dump() for conv in self.tiered_graph.conversations.values()
        ]

        with open(tier3_file, "w") as f:
            json.dump(conversations_data, f, indent=2, default=str)

        logger.info(f"Saved {len(conversations_data)} conversations to {tier3_file}")

    def load_tier_3(self, output_dir: Path) -> bool:
        """Load tier 3 results from JSON if available."""
        tier3_file = output_dir / "tier3_conversations.json"

        if not tier3_file.exists():
            return False

        logger.info(f"Loading tier 3 from {tier3_file}")

        with open(tier3_file, "r") as f:
            conversations_data = json.load(f)

        # Deserialize conversations
        from .models import ConversationBoundary

        conversations = [
            ConversationBoundary.model_validate(conv_data)
            for conv_data in conversations_data
        ]

        # Initialize or update tiered graph
        if self.tiered_graph is None:
            self.tiered_graph = TieredMemoryGraph(
                memory_graph_path=None,
                conversations={conv.id: conv for conv in conversations},
            )
        else:
            self.tiered_graph.conversations = {conv.id: conv for conv in conversations}

        # Build trigger_entries_dict (needed for retrieval formatting)
        trigger_entries = self.trigger_history.get_all_entries()
        self.trigger_entries_dict = {entry.entry_id: entry for entry in trigger_entries}

        logger.info(f"Loaded {len(conversations)} conversations from {tier3_file}")
        return True

    def build_tier_4(
        self,
        clustering_source: str = "conversations",
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Build tier 4 (semantic clusters) with incremental saving.

        Args:
            clustering_source: Source for clustering ('conversations', 'triggers', 'memories')
            output_dir: Directory to save incremental progress (optional)
        """
        logger.info(f"Building tier 4: Semantic clusters from {clustering_source}")

        if not self.tiered_graph:
            raise ValueError("Must build tier 3 before tier 4")

        # Callback for incremental saving
        def on_cluster_created(cluster):
            """Callback to save progress after each cluster."""
            if output_dir and self.tiered_graph:
                # Add cluster
                self.tiered_graph.semantic_clusters[cluster.id] = cluster

                # Save incrementally
                self.save_tier_4(output_dir)

        if clustering_source == "conversations":
            # Cluster conversations
            conversations = list(self.tiered_graph.conversations.values())
            if not conversations:
                logger.warning("No conversations available for clustering")
                return

            clusters = create_semantic_clusters_from_conversations(
                conversations=conversations,
                llm=self.llm,
                model=self.model,
                state=self.state,
                on_cluster_created=on_cluster_created,
            )

        elif clustering_source == "triggers":
            # Cluster trigger entries directly
            trigger_entries = self.trigger_history.get_all_entries()
            clusters = create_semantic_clusters_from_trigger_entries(
                trigger_entries=trigger_entries,
                llm=self.llm,
                model=self.model,
                state=self.state,
                on_cluster_created=on_cluster_created,
            )

        elif clustering_source == "memories":
            # Cluster memory elements
            from .semantic_clustering import (
                create_semantic_clusters_from_memory_elements,
            )

            clusters = create_semantic_clusters_from_memory_elements(
                memory_graph=self.memory_graph,
                llm=self.llm,
                model=self.model,
                state=self.state,
                on_cluster_created=on_cluster_created,
            )

        else:
            raise ValueError(f"Unknown clustering source: {clustering_source}")

        # Ensure all clusters are in the graph (in case callback wasn't used)
        for cluster in clusters:
            if cluster.id not in self.tiered_graph.semantic_clusters:
                self.tiered_graph.semantic_clusters[cluster.id] = cluster

        logger.info(f"Created {len(clusters)} semantic clusters (tier 4)")

    def save_tier_4(self, output_dir: Path) -> None:
        """Save tier 4 results to JSON."""
        if not self.tiered_graph:
            return

        tier4_file = output_dir / "tier4_semantic_clusters.json"
        logger.info(f"Saving tier 4 to {tier4_file}")

        # Serialize clusters
        clusters_data = [
            cluster.model_dump()
            for cluster in self.tiered_graph.semantic_clusters.values()
        ]

        with open(tier4_file, "w") as f:
            json.dump(clusters_data, f, indent=2, default=str)

        logger.info(f"Saved {len(clusters_data)} clusters to {tier4_file}")

    def load_tier_4(self, output_dir: Path) -> bool:
        """Load tier 4 results from JSON if available."""
        tier4_file = output_dir / "tier4_semantic_clusters.json"

        if not tier4_file.exists():
            return False

        logger.info(f"Loading tier 4 from {tier4_file}")

        with open(tier4_file, "r") as f:
            clusters_data = json.load(f)

        # Deserialize clusters
        from .models import SemanticCluster

        clusters = [
            SemanticCluster.model_validate(cluster_data)
            for cluster_data in clusters_data
        ]

        # Update tiered graph
        if self.tiered_graph is None:
            raise ValueError("Must load or build tier 3 before loading tier 4")

        self.tiered_graph.semantic_clusters = {
            cluster.id: cluster for cluster in clusters
        }

        logger.info(f"Loaded {len(clusters)} clusters from {tier4_file}")
        return True

    def run_retrieval_experiment(
        self,
        test_queries: List[str],
        token_budget: int = 8000,
        use_iterative: bool = False,
    ) -> Dict[str, Any]:
        """
        Run retrieval experiments with test queries.

        Args:
            test_queries: List of test query strings
            token_budget: Maximum tokens for context (default 8000)
            use_iterative: Whether to use iterative drill-down retrieval

        Returns:
            Dictionary of results by query
        """
        if not self.tiered_graph:
            raise ValueError("Must build tiered graph before retrieval")

        mode = "iterative drill-down" if use_iterative else "single-pass"
        logger.info(
            f"Running {mode} retrieval experiment with {len(test_queries)} queries "
            f"(token budget: {token_budget})"
        )

        results = {}
        trigger_entries = self.trigger_history.get_all_entries()

        for query in tqdm(
            test_queries, desc="Running retrieval experiments", file=sys.stdout
        ):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Query: '{query}'")
            logger.info(f"{'=' * 80}")

            if use_iterative:
                # Use iterative drill-down approach
                from .iterative_retrieval import iterative_drill_down_retrieval

                iterative_result = iterative_drill_down_retrieval(
                    query=query,
                    tiered_graph=self.tiered_graph,
                    memory_graph=self.memory_graph,
                    trigger_entries=trigger_entries,
                    trigger_entries_dict=self.trigger_entries_dict,
                    state=self.state,
                    llm=self.llm,
                    model=self.model,
                    token_budget=token_budget,
                    min_similarity=0.3,
                )

                results[query] = {
                    "llm_prompt_context": iterative_result["final_context"],
                    "llm_prompt_tokens": iterative_result["final_tokens"],
                    "iterations": iterative_result["iterations"],
                    "iteration_log": iterative_result["iteration_log"],
                    "elements_shown": iterative_result["elements_shown"],
                }

                logger.info(
                    f"Iterative retrieval: {iterative_result['iterations']} iterations, "
                    f"{iterative_result['elements_shown']} elements, "
                    f"{iterative_result['final_tokens']} tokens"
                )

            else:
                # Use single-pass approach
                retrieval_results = retrieve_multi_tier(
                    query=query,
                    tiered_graph=self.tiered_graph,
                    memory_graph=self.memory_graph,
                    trigger_entries=trigger_entries,
                    min_similarity=0.3,
                )

                # Format as LLM prompt (what would actually be shown to the agent)
                llm_prompt_context = format_for_llm_prompt(
                    retrieval_results=retrieval_results,
                    tiered_graph=self.tiered_graph,
                    memory_graph=self.memory_graph,
                    trigger_entries_dict=self.trigger_entries_dict,
                    token_budget=token_budget,
                )

                # Estimate tokens
                llm_prompt_tokens = estimate_context_tokens(llm_prompt_context)

                results[query] = {
                    "retrieval_results": retrieval_results,
                    "llm_prompt_context": llm_prompt_context,
                    "llm_prompt_tokens": llm_prompt_tokens,
                    "total_results": retrieval_results.total_results,
                    "tiers_used": [t.value for t in retrieval_results.tiers_used],
                }

                logger.info(
                    f"Retrieved {retrieval_results.total_results} results, "
                    f"LLM prompt tokens: {llm_prompt_tokens}"
                )

        return results

    def print_summary(self) -> None:
        """Print summary of tiered graph statistics."""
        ui_print("\n" + "=" * 80)
        ui_print("TIERED MEMORY GRAPH SUMMARY")
        ui_print("=" * 80)

        ui_print(f"\nTier 1 (Atomic):")
        ui_print(f"  Memory elements: {len(self.memory_graph.elements)}")
        ui_print(f"  Edges: {len(self.memory_graph.edges)}")

        ui_print(f"\nTier 2 (Trigger-Response):")
        ui_print(f"  Trigger entries: {len(self.memory_graph.containers)}")

        if self.tiered_graph:
            ui_print(f"\nTier 3 (Conversations):")
            ui_print(
                f"  Conversation boundaries: {len(self.tiered_graph.conversations)}"
            )

            if self.tiered_graph.conversations:
                avg_duration = sum(
                    c.duration_seconds for c in self.tiered_graph.conversations.values()
                ) / len(self.tiered_graph.conversations)
                ui_print(f"  Average duration: {avg_duration:.0f} seconds")

                all_tags = set()
                for conv in self.tiered_graph.conversations.values():
                    all_tags.update(conv.topic_tags)
                ui_print(f"  Unique topic tags: {len(all_tags)}")

            ui_print(f"\nTier 4 (Semantic Clusters):")
            ui_print(f"  Clusters: {len(self.tiered_graph.semantic_clusters)}")

            if self.tiered_graph.semantic_clusters:
                avg_size = sum(
                    c.cluster_size for c in self.tiered_graph.semantic_clusters.values()
                ) / len(self.tiered_graph.semantic_clusters)
                ui_print(f"  Average cluster size: {avg_size:.1f} elements")

                topics = [
                    c.cluster_topic
                    for c in self.tiered_graph.semantic_clusters.values()
                ]
                ui_print(f"  Topics: {', '.join(topics)}")

        ui_print("=" * 80)


def create_default_test_queries() -> List[str]:
    """Create default set of test queries for the experiment."""
    return [
        # Broad topic queries (tier 4)
        "all discussions about creative projects and art",
        "conversations about devotion and relationships with David",
        # Specific conversation queries (tier 3)
        "what did we discuss in our last conversation",
        "conversations about the electrolysis art project",
        "discussions about fashion design and creative expression",
        # Specific exchange queries (tier 2)
        "when did we go for tea or boba",
        "what did David say about the art project",
        "discussions about walks and outdoor activities",
        # Specific detail queries (tier 1)
        "what were David's exact words when expressing affection",
        "specific details about the interactive art piece design",
    ]


def main():
    """Main entry point for experiment."""
    parser = argparse.ArgumentParser(description="Run tiered memory experiment")
    parser.add_argument(
        "--conversation",
        type=str,
        required=True,
        help="Conversation ID prefix (e.g., conversation_20251004_012921_639291)",
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
        default="./experiment_results",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--skip-topic-detection",
        action="store_true",
        help="Skip LLM-based topic shift detection in tier 3",
    )
    parser.add_argument(
        "--clustering-source",
        type=str,
        choices=["conversations", "triggers", "memories"],
        default="conversations",
        help="Source for tier 4 clustering",
    )
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Use iterative drill-down retrieval instead of single-pass",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment
    experiment = TieredMemoryExperiment(
        conversation_prefix=args.conversation,
        conversations_dir=args.conversations_dir,
    )

    # PHASE 1: Build or load tier 3
    ui_print("\n" + "=" * 80)
    ui_print("PHASE 1: Building Tier 3 (Conversation Boundaries)")
    ui_print("=" * 80)

    if experiment.load_tier_3(output_dir):
        ui_print("✓ Tier 3 loaded from saved file - skipping rebuild")
    else:
        ui_print("Building tier 3 from scratch with incremental saving...")
        experiment.build_tier_3(
            use_topic_detection=not args.skip_topic_detection, output_dir=output_dir
        )
        ui_print("✓ Phase 1 complete - Tier 3 saved")

    # PHASE 2: Build or load tier 4
    ui_print("\n" + "=" * 80)
    ui_print("PHASE 2: Building Tier 4 (Semantic Clusters)")
    ui_print("=" * 80)

    if experiment.load_tier_4(output_dir):
        ui_print("✓ Tier 4 loaded from saved file - skipping rebuild")
    else:
        ui_print("Building tier 4 from scratch with incremental saving...")
        experiment.build_tier_4(
            clustering_source=args.clustering_source, output_dir=output_dir
        )
        ui_print("✓ Phase 2 complete - Tier 4 saved")

    # PHASE 3: Print summary
    ui_print("\n" + "=" * 80)
    ui_print("PHASE 3: Summary")
    ui_print("=" * 80)
    experiment.print_summary()

    # PHASE 4: Run retrieval experiments
    ui_print("\n" + "=" * 80)
    mode_str = "Iterative Drill-Down" if args.iterative else "Single-Pass"
    ui_print(f"PHASE 4: Running {mode_str} Retrieval Experiments")
    ui_print("=" * 80)
    test_queries = create_default_test_queries()
    token_budget = 8000
    results = experiment.run_retrieval_experiment(
        test_queries,
        token_budget=token_budget,
        use_iterative=args.iterative
    )

    # PHASE 5: Save retrieval results and generate summary report
    ui_print("\n" + "=" * 80)
    ui_print("PHASE 5: Saving Retrieval Results")
    ui_print("=" * 80)

    # Save individual query results (LLM prompt format only)
    for query, data in tqdm(
        results.items(), desc="Saving result files", file=sys.stdout
    ):
        query_slug = query.replace(" ", "_")[:50]

        # Save LLM prompt format (what would actually be shown to the agent)
        filename = f"{query_slug}.txt"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            f.write(f"Query: {query}\n\n")
            f.write(data["llm_prompt_context"])

    # Generate summary report
    ui_print("\nGenerating results summary report...")
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("RETRIEVAL RESULTS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append(
        f"Token Budget: {token_budget} tokens per query"
    )
    summary_lines.append(
        f"Total Queries: {len(test_queries)}"
    )
    summary_lines.append("")

    # Per-query details
    total_tokens = 0
    for query, data in results.items():
        tokens = data["llm_prompt_tokens"]
        total_tokens += tokens

        summary_lines.append(f"\nQuery: {query}")
        summary_lines.append("-" * 80)
        summary_lines.append(f"  LLM Prompt Tokens: {tokens:>6} / {token_budget}")

        # Add mode-specific fields
        if "iterations" in data:
            # Iterative mode
            summary_lines.append(f"  Iterations: {data['iterations']}")
            summary_lines.append(f"  Elements Shown: {data['elements_shown']}")
        else:
            # Single-pass mode
            summary_lines.append(f"  Results Retrieved: {data['total_results']}")
            summary_lines.append(f"  Tiers Used: {', '.join(data['tiers_used'])}")

        # Calculate budget utilization
        utilization = (tokens / token_budget) * 100
        summary_lines.append(f"  Budget Utilization: {utilization:.1f}%")

    # Overall statistics
    summary_lines.append("")
    summary_lines.append("=" * 80)
    summary_lines.append("OVERALL STATISTICS")
    summary_lines.append("=" * 80)
    summary_lines.append(f"  Total Tokens Used: {total_tokens:>7}")
    summary_lines.append(f"  Average Tokens per Query: {total_tokens / len(test_queries):.0f}")
    avg_utilization = (total_tokens / (token_budget * len(test_queries))) * 100
    summary_lines.append(f"  Average Budget Utilization: {avg_utilization:.1f}%")

    summary_lines.append("")
    summary_lines.append("=" * 80)

    # Save summary report
    summary_path = output_dir / "results_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    ui_print(f"Saved results summary to {summary_path}")
    ui_print(f"\n✓ Experiment complete! All results saved to {output_dir}")


if __name__ == "__main__":
    main()
