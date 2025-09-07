#!/usr/bin/env python3
"""
Full KG vs Memory Comparison Pipeline

Comprehensive experimental comparison that tests the complete KG-based agent flow
against the current memory-based flow using all the new experimental components.

Flow comparison:
CURRENT: context prompt → memory query extraction → memory retrieval → situational analysis
KG-BASED: context prompt → KG query extraction → KG context building → situational analysis
"""

import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from agent.conversation_persistence import ConversationPersistence
from agent.memory.memory_extraction import (
    extract_memory_queries,
    retrieve_relevant_memories,
)
from agent.experiments.knowledge_graph.knowledge_graph_builder import (
    ValidatedKnowledgeGraphBuilder,
)
from agent.experiments.knowledge_graph.knowledge_graph_querying import (
    KnowledgeGraphQuerying,
)
from agent.experiments.knowledge_graph.kg_context_builder import (
    KGContextBuilder,
    ContextFormat,
)
from agent.llm import create_llm, SupportedModel
from agent.chain_of_action.trigger import UserInputTrigger

logger = logging.getLogger(__name__)


class ComparisonScenario:
    """A test scenario for comparing memory vs KG approaches"""

    def __init__(self, name: str, user_input: str, description: str):
        self.name = name
        self.user_input = user_input
        self.description = description


class ComparisonResults:
    """Results from comparing memory vs KG approaches"""

    def __init__(self):
        # Memory approach results
        self.memory_queries_extracted = 0
        self.memory_items_retrieved = 0
        self.memory_context_length = 0
        self.memory_context = ""
        self.memory_extraction_time = 0.0

        # KG approach results
        self.kg_should_query = False
        self.kg_focus_entities = []
        self.kg_relationship_types = []
        self.kg_context_length = 0
        self.kg_context = ""
        self.kg_extraction_time = 0.0

        # Comparison metrics
        self.context_richness_ratio = 0.0
        self.kg_advantages = []


class FullComparisonPipeline:
    """Comprehensive comparison between memory and KG approaches"""

    def __init__(self, conversation_name: str = "baseline"):
        self.conversation_name = conversation_name
        self.llm = create_llm()
        self.model = SupportedModel.MISTRAL_SMALL_3_2_Q4

        # Load conversation data
        persistence = ConversationPersistence()
        self.trigger_history, state, _ = persistence.load_conversation(
            conversation_name
        )

        if state is None:
            raise ValueError(f"Could not load conversation: {conversation_name}")

        self.state = state  # Now guaranteed to be non-None

        # Build KG for testing (limited size for comparison)
        print("🏗️  Building knowledge graph for comparison...")
        self.kg_builder = ValidatedKnowledgeGraphBuilder(
            self.llm, self.model, self.state
        )

        # Use recent triggers to build a representative but manageable KG
        all_triggers = self.trigger_history.get_all_entries()
        recent_triggers = all_triggers[-20:]  # Last 20 triggers for quick comparison

        previous_trigger = None
        for i, trigger in enumerate(recent_triggers):
            print(f"  Processing trigger {i+1}/20...")
            self.kg_builder.process_trigger_incremental(trigger, previous_trigger)
            previous_trigger = trigger

        stats = self.kg_builder.get_stats()
        print(
            f"✅ Built KG: {stats['total_nodes']} nodes, {stats['total_relationships']} relationships"
        )

        # Create experimental components
        self.kg_context_builder = KGContextBuilder(self.kg_builder.graph, self.state)
        self.available_entities = [
            n.name
            for n in self.kg_builder.graph.get_all_nodes()
            if n.node_type.value != "EXPERIENCE"
        ]
        self.available_relationship_types = list(
            self.kg_builder.graph.get_relationship_type_stats().keys()
        )

    def run_comparison(
        self,
        scenarios: List[ComparisonScenario],
        context_formats: Optional[List[ContextFormat]] = None,
    ) -> Dict[str, Any]:
        """Run full comparison across scenarios and context formats"""

        if context_formats is None:
            context_formats = [
                ContextFormat.STRUCTURED,
                ContextFormat.NARRATIVE,
                ContextFormat.BULLET_POINTS,
            ]

        print(f"🧪 Running Full Comparison Pipeline")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Context formats: {[f.value for f in context_formats]}")
        print("=" * 80)

        all_results = {}

        for scenario in scenarios:
            print(f"\n🎯 Scenario: {scenario.name}")
            print(f'Input: "{scenario.user_input}"')
            print(f"Description: {scenario.description}")
            print("-" * 60)

            scenario_results = {}

            # Test each context format with this scenario
            for format_type in context_formats:
                print(f"\n📊 Testing {format_type.value.upper()} format:")

                results = self._compare_single_scenario(scenario, format_type)
                scenario_results[format_type.value] = results

                self._print_comparison_summary(results, format_type)

            all_results[scenario.name] = scenario_results

        # Generate overall analysis
        overall_analysis = self._generate_overall_analysis(all_results)
        all_results["overall_analysis"] = overall_analysis

        self._print_overall_summary(overall_analysis)

        return all_results

    def _compare_single_scenario(
        self, scenario: ComparisonScenario, format_type: ContextFormat
    ) -> ComparisonResults:
        """Compare memory vs KG for a single scenario and format"""

        results = ComparisonResults()
        trigger = UserInputTrigger(content=scenario.user_input)
        recent_triggers = self.trigger_history.get_all_entries()[
            -5:
        ]  # Last 5 for context

        # === MEMORY APPROACH ===
        print("  📋 Memory approach...")
        start_time = time.time()

        try:
            # Extract memory queries
            memory_queries = extract_memory_queries(
                state=self.state,
                trigger=trigger,
                trigger_history=self.trigger_history,
                llm=self.llm,
                model=self.model,
            )

            results.memory_queries_extracted = 1 if memory_queries else 0

            # Retrieve memories
            memories = (
                retrieve_relevant_memories(
                    memory_query=memory_queries,
                    trigger_history=self.trigger_history,
                    max_results=5,
                )
                if memory_queries
                else []
            )

            results.memory_items_retrieved = len(memories)
            results.memory_context = str(memories)
            results.memory_context_length = len(results.memory_context)

        except Exception as e:
            logger.error(f"Memory approach failed: {e}")
            results.memory_context = f"Memory extraction failed: {e}"

        results.memory_extraction_time = time.time() - start_time

        # === KG APPROACH ===
        print("  🕸️  KG approach...")
        start_time = time.time()

        try:
            # Use the consolidated querying system
            querying = KnowledgeGraphQuerying(
                self.kg_builder.graph, self.llm, self.model, self.state
            )
            recent_triggers = self.trigger_history.get_all_entries()[
                -3:
            ]  # Last 3 for context

            # Get user input from trigger
            user_input = trigger.content

            query_determination = querying.determine_context_needs(
                user_input, recent_triggers
            )

            if (
                query_determination
                and query_determination.should_query
                and query_determination.query
            ):
                results.kg_should_query = True
                results.kg_focus_entities = query_determination.query.focus_entities
                results.kg_relationship_types = (
                    query_determination.query.relationship_types
                )

                # Build context using experimental context builder
                kg_context = self.kg_context_builder.build_context_from_kg_query(
                    query_determination.query,
                    format_type=format_type,
                    max_context_length=2000,
                )

                results.kg_context = kg_context
                results.kg_context_length = len(kg_context)
            else:
                results.kg_context = "No KG query needed or extraction failed"
                results.kg_context_length = len(results.kg_context)

        except Exception as e:
            logger.error(f"KG approach failed: {e}")
            results.kg_context = f"KG extraction failed: {e}"
            results.kg_context_length = len(results.kg_context)

        results.kg_extraction_time = time.time() - start_time

        # === ANALYSIS ===
        results.context_richness_ratio = results.kg_context_length / max(
            results.memory_context_length, 1
        )

        # Determine KG advantages
        advantages = []
        if results.kg_should_query:
            if len(results.kg_focus_entities) > results.memory_items_retrieved:
                advantages.append("More comprehensive entity coverage")
            if results.kg_context_length > results.memory_context_length:
                advantages.append("Richer context information")
            if (
                "RELATIONSHIPS" in results.kg_context
                or "CONNECTIONS" in results.kg_context
            ):
                advantages.append("Relationship awareness")
            if results.kg_extraction_time < results.memory_extraction_time:
                advantages.append("Faster extraction")

        results.kg_advantages = advantages

        return results

    def _print_comparison_summary(
        self, results: ComparisonResults, format_type: ContextFormat
    ):
        """Print summary for a single comparison"""

        print(
            f"    Memory: {results.memory_items_retrieved} items, {results.memory_context_length} chars, {results.memory_extraction_time:.2f}s"
        )
        print(
            f"    KG: {len(results.kg_focus_entities)} entities, {results.kg_context_length} chars, {results.kg_extraction_time:.2f}s"
        )
        print(f"    Richness ratio: {results.context_richness_ratio:.1f}x")
        if results.kg_advantages:
            print(f"    KG advantages: {', '.join(results.kg_advantages)}")

    def _generate_overall_analysis(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis across all scenarios and formats"""

        analysis = {
            "total_scenarios": len(
                [k for k in all_results.keys() if k != "overall_analysis"]
            ),
            "format_performance": {},
            "kg_success_rate": 0.0,
            "avg_richness_improvement": 0.0,
            "common_advantages": {},
            "best_format": None,
        }

        format_scores = {}
        total_comparisons = 0
        successful_kg_queries = 0
        total_richness_ratio = 0.0
        all_advantages = []

        for scenario_name, scenario_results in all_results.items():
            if scenario_name == "overall_analysis":
                continue

            for format_name, results in scenario_results.items():
                total_comparisons += 1

                # Track format performance
                if format_name not in format_scores:
                    format_scores[format_name] = []

                # Score based on multiple factors
                score = 0
                if results.kg_should_query:
                    successful_kg_queries += 1
                    score += 2  # Base points for successful query
                    score += min(
                        results.context_richness_ratio, 3
                    )  # Richness bonus (capped at 3x)
                    score += len(results.kg_advantages)  # Advantage bonus

                format_scores[format_name].append(score)
                total_richness_ratio += results.context_richness_ratio
                all_advantages.extend(results.kg_advantages)

        # Calculate metrics
        analysis["kg_success_rate"] = (
            successful_kg_queries / total_comparisons if total_comparisons > 0 else 0
        )
        analysis["avg_richness_improvement"] = (
            total_richness_ratio / total_comparisons if total_comparisons > 0 else 0
        )

        # Format performance
        for format_name, scores in format_scores.items():
            analysis["format_performance"][format_name] = {
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "total_tests": len(scores),
            }

        # Best format
        if format_scores:
            best_format = max(
                format_scores.keys(),
                key=lambda f: analysis["format_performance"][f]["avg_score"],
            )
            analysis["best_format"] = best_format

        # Common advantages
        advantage_counts = {}
        for advantage in all_advantages:
            advantage_counts[advantage] = advantage_counts.get(advantage, 0) + 1
        analysis["common_advantages"] = advantage_counts

        return analysis

    def _print_overall_summary(self, analysis: Dict[str, Any]):
        """Print overall analysis summary"""

        print(f"\n📈 OVERALL ANALYSIS:")
        print("=" * 80)
        print(f"Total scenarios tested: {analysis['total_scenarios']}")
        print(f"KG query success rate: {analysis['kg_success_rate']:.1%}")
        print(
            f"Average context richness improvement: {analysis['avg_richness_improvement']:.1f}x"
        )

        if analysis["best_format"]:
            print(
                f"Best performing format: {analysis['best_format']} (avg score: {analysis['format_performance'][analysis['best_format']]['avg_score']:.1f})"
            )

        print(f"\nFormat Performance:")
        for format_name, perf in analysis["format_performance"].items():
            print(
                f"  {format_name}: {perf['avg_score']:.1f} avg score ({perf['total_tests']} tests)"
            )

        if analysis["common_advantages"]:
            print(f"\nKG Advantages (frequency):")
            for advantage, count in analysis["common_advantages"].items():
                print(f"  {advantage}: {count} times")


def main():
    """Run the full comparison pipeline"""

    print("🚀 Full KG vs Memory Comparison Pipeline")
    print("=" * 80)

    # Define test scenarios
    scenarios = [
        ComparisonScenario(
            name="Personal Relationship Query",
            user_input="How are you feeling about David lately?",
            description="Test handling of personal relationships and emotional context",
        ),
        ComparisonScenario(
            name="Recent Activity Question",
            user_input="What have we been talking about recently?",
            description="Test retrieval of recent conversational context",
        ),
        ComparisonScenario(
            name="Emotional State Inquiry",
            user_input="You seem a bit anxious today, what's going on?",
            description="Test emotional awareness and self-reflection",
        ),
        ComparisonScenario(
            name="Topic Exploration",
            user_input="Let's continue our discussion about programming",
            description="Test topic continuity and knowledge recall",
        ),
        ComparisonScenario(
            name="General Greeting",
            user_input="Hello, how are you doing?",
            description="Test basic interaction handling",
        ),
    ]

    # Run comparison with multiple context formats
    context_formats = [
        ContextFormat.STRUCTURED,
        ContextFormat.NARRATIVE,
        ContextFormat.BULLET_POINTS,
        ContextFormat.CONFIDENCE_WEIGHTED,
    ]

    try:
        pipeline = FullComparisonPipeline("baseline")
        results = pipeline.run_comparison(scenarios, context_formats)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"full_comparison_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to {results_file}")
        print(f"\n✅ Full comparison pipeline completed!")

    except Exception as e:
        logger.error(f"Comparison pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
