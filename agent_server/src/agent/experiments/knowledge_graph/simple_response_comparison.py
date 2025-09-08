#!/usr/bin/env python3
"""
Simple Response Comparison

A simplified comparison between current memory system and knowledge graph
that focuses on demonstrating the value proposition without complex integrations.
"""

import json
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
from agent.llm import create_llm, SupportedModel
from agent.chain_of_action.trigger import UserInputTrigger

logger = logging.getLogger(__name__)


def simple_comparison_test():
    """Simple comparison between current and knowledge graph approaches"""

    logging.basicConfig(level=logging.INFO)

    print("🧪 Simple Response Comparison Test")
    print("=" * 50)

    # Load baseline data
    persistence = ConversationPersistence()
    trigger_history, state, _ = persistence.load_conversation("baseline")

    if not state:
        print("❌ Could not load baseline data")
        return

    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    # Build knowledge graph (limited for demo)
    print("\n📈 Building knowledge graph from recent triggers...")
    kg_builder = ValidatedKnowledgeGraphBuilder(llm, model, state)

    # Use just the most recent 5 triggers for quick demo
    all_triggers = trigger_history.get_all_entries()
    recent_triggers = all_triggers[-5:]

    previous_trigger = None
    for i, trigger in enumerate(recent_triggers):
        print(f"   Processing trigger {i+1}/5...")
        kg_builder.process_trigger_incremental(trigger, previous_trigger)
        previous_trigger = trigger

    stats = kg_builder.get_stats()
    print(
        f"✅ Knowledge graph: {stats.total_nodes} nodes, {stats.total_relationships} relationships"
    )

    # Test scenarios
    test_scenarios = [
        {
            "name": "Anime Interest",
            "user_input": "I want to learn more about anime recommendations",
            "expected_context": "Should find anime-related discussions and preferences",
        },
        {
            "name": "Follow-up Question",
            "user_input": "What were we talking about earlier?",
            "expected_context": "Should find recent conversation context",
        },
    ]

    results = []

    for scenario in test_scenarios:
        print(f"\n🎯 Test Scenario: {scenario['name']}")
        print(f"User Input: \"{scenario['user_input']}\"")
        print("-" * 40)

        # === CURRENT APPROACH ===
        print("📋 CURRENT MEMORY APPROACH:")
        try:
            # Simulate current memory extraction
            trigger = UserInputTrigger(content=scenario["user_input"])

            memory_queries = extract_memory_queries(
                state=state,
                trigger=trigger,
                trigger_history=trigger_history,
                llm=llm,
                model=model,
            )

            memories = (
                retrieve_relevant_memories(
                    memory_query=memory_queries,
                    trigger_history=trigger_history,
                    max_results=5,
                )
                if memory_queries
                else []
            )

            print(f"   Memory queries extracted: {1 if memory_queries else 0}")
            print(f"   Relevant memories found: {len(memories)}")

            # Show memory content
            if memories:
                print("   Memory snippets:")
                for i, mem in enumerate(memories[:3]):
                    content = (
                        mem.get("content", str(mem))[:100]
                        if isinstance(mem, dict)
                        else str(mem)[:100]
                    )
                    print(f"   - {content}...")
            else:
                print("   - No relevant memories found")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            memories = []

        # === KNOWLEDGE GRAPH APPROACH ===
        print("\n🕸️  KNOWLEDGE GRAPH APPROACH:")

        # Create KnowledgeGraphQuerying system
        querying = KnowledgeGraphQuerying(
            graph=kg_builder.graph, llm=llm, model=model, state=state
        )

        # Get context from knowledge graph
        formatted_kg_context = querying.construct_agent_context(
            current_input=scenario["user_input"], recent_triggers=recent_triggers
        )

        print(f"   Knowledge graph context generated")
        print(f"   Graph nodes total: {stats.total_nodes}")

        if "KEY ENTITIES" in formatted_kg_context:
            entity_count = len(
                [
                    line
                    for line in formatted_kg_context.split("\n")
                    if line.startswith("- ")
                ]
            )
            print(f"   Context entities found: {entity_count}")

        if (
            formatted_kg_context
            and formatted_kg_context != "No relevant context found in knowledge graph."
        ):
            print("   Context preview:")
            preview_lines = formatted_kg_context.split("\n")[:5]
            for line in preview_lines:
                if line.strip():
                    print(f"   {line[:80]}{'...' if len(line) > 80 else ''}")

        # === COMPARISON ===
        print(f"\n📊 COMPARISON:")
        current_context_length = len(str(memories))
        kg_context_length = len(formatted_kg_context)

        print(f"   Current approach context length: {current_context_length} chars")
        print(f"   Knowledge graph context length: {kg_context_length} chars")
        print(
            f"   Context richness ratio: {kg_context_length / max(current_context_length, 1):.1f}x"
        )

        advantages = []
        if "KEY ENTITIES" in formatted_kg_context:
            entity_count = len(
                [
                    line
                    for line in formatted_kg_context.split("\n")
                    if line.startswith("- ")
                ]
            )
            if entity_count > len(memories):
                advantages.append("More comprehensive knowledge retrieval")
        if "EMOTIONAL CONTEXT" in formatted_kg_context:
            advantages.append("Emotional awareness")
        if "PATTERNS & INSIGHTS" in formatted_kg_context:
            advantages.append("Relationship insights")

        if advantages:
            print(f"   Knowledge graph advantages: {', '.join(advantages)}")

        # Count entities/knowledge items from formatted context
        kg_knowledge_count = len(
            [line for line in formatted_kg_context.split("\n") if line.startswith("- ")]
        )

        results.append(
            {
                "scenario": scenario["name"],
                "user_input": scenario["user_input"],
                "current_memories_count": len(memories),
                "kg_knowledge_count": kg_knowledge_count,
                "current_context_length": current_context_length,
                "kg_context_length": kg_context_length,
                "kg_advantages": advantages,
                "full_kg_context": formatted_kg_context,
            }
        )

    # === SUMMARY ===
    print(f"\n📈 OVERALL COMPARISON SUMMARY:")
    print("=" * 50)

    total_current_items = sum(r["current_memories_count"] for r in results)
    total_kg_items = sum(r["kg_knowledge_count"] for r in results)
    avg_context_ratio = sum(
        r["kg_context_length"] / max(r["current_context_length"], 1) for r in results
    ) / len(results)

    print(f"Total memory items retrieved (current): {total_current_items}")
    print(f"Total knowledge items retrieved (KG): {total_kg_items}")
    print(f"Average context richness improvement: {avg_context_ratio:.1f}x")

    all_advantages = []
    for result in results:
        all_advantages.extend(result["kg_advantages"])

    if all_advantages:
        advantage_counts = {}
        for adv in all_advantages:
            advantage_counts[adv] = advantage_counts.get(adv, 0) + 1

        print("Knowledge graph advantages observed:")
        for advantage, count in advantage_counts.items():
            print(f"  - {advantage} ({count}/{len(results)} scenarios)")

    # Save results
    with open("simple_comparison_results.json", "w") as f:
        json.dump(
            {
                "test_summary": {
                    "total_scenarios": len(results),
                    "kg_stats": stats,
                    "avg_context_improvement": avg_context_ratio,
                    "total_current_items": total_current_items,
                    "total_kg_items": total_kg_items,
                },
                "scenario_results": results,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Detailed results saved to simple_comparison_results.json")

    # Show one formatted example
    if results:
        print(f"\n🔍 EXAMPLE KNOWLEDGE GRAPH CONTEXT:")
        print("=" * 50)
        print(results[0]["full_kg_context"])

    print(f"\n✅ Simple comparison completed!")


if __name__ == "__main__":
    simple_comparison_test()
