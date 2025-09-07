#!/usr/bin/env python3
"""
Knowledge Graph Query Extraction

Extracts knowledge graph queries from existing context prompts, similar to how
memory query extraction works. This replaces memory query extraction in the
experimental KG-based agent flow.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from agent.state import State
from agent.chain_of_action.trigger import BaseTrigger
from agent.chain_of_action.trigger_history import TriggerHistory
from agent.chain_of_action.prompts import build_memory_extraction_prompt
from agent.structured_llm import direct_structured_llm_call
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class KnowledgeGraphQuery(BaseModel):
    """A structured query for the knowledge graph"""

    focus_entities: List[str] = Field(
        default_factory=list,
        description="Specific entities (people, concepts, emotions) to focus retrieval on",
    )
    relationship_types: List[str] = Field(
        default_factory=list,
        description="Types of relationships to explore (e.g., 'caused', 'involves', 'relates_to')",
    )
    max_depth: int = Field(
        default=2, description="Maximum depth to traverse from focus entities (1-3)"
    )
    include_recent_experiences: bool = Field(
        default=True, description="Whether to include recent experience nodes"
    )
    temporal_focus: Optional[str] = Field(
        default=None,
        description="Temporal focus like 'recent', 'past week', or specific timeframe",
    )


class KGQueryExtraction(BaseModel):
    """What the LLM extracts from context for KG retrieval"""

    should_query: bool = Field(
        description="Whether querying the knowledge graph would be helpful for this context"
    )
    reasoning: str = Field(
        description="Why this query approach was chosen or why no query is needed"
    )
    kg_query: Optional[KnowledgeGraphQuery] = Field(
        default=None,
        description="The knowledge graph query to execute (only if should_query=True)",
    )


def build_kg_query_extraction_prompt(
    state: State,
    trigger: BaseTrigger,
    trigger_history: TriggerHistory,
    available_entities: List[str],
    available_relationship_types: List[str],
) -> str:
    """Build prompt for extracting KG queries from current context"""

    # Reuse the memory extraction context building logic
    base_prompt = build_memory_extraction_prompt(state, trigger, trigger_history)

    # Replace the memory-specific instructions with KG-specific ones
    kg_specific_section = f"""**KNOWLEDGE GRAPH QUERY ANALYSIS:**

Based on my current context, would querying my knowledge graph be helpful? If so, what should I focus on?

Available entities in my knowledge graph: {', '.join(available_entities[:20])}{'...' if len(available_entities) > 20 else ''}
Available relationship types: {', '.join(available_relationship_types[:15])}{'...' if len(available_relationship_types) > 15 else ''}

Guidelines for KG queries:
1. Focus on specific entities (people, concepts, emotions) that are relevant
2. Choose relationship types that would reveal useful connections
3. Set appropriate traversal depth (1=direct connections, 2=friends-of-friends, 3=deeper network)
4. Consider whether recent experiences should be included
5. Think about temporal aspects if timing is important

Query when:
- Specific people, concepts, or emotions are mentioned
- I need to understand relationships or patterns
- Context involves past experiences or commitments
- Emotional context would be helpful

Don't query when:
- The context is purely procedural
- No specific entities or relationships are relevant
- Simple acknowledgments or basic responses
"""

    # Replace the memory section with KG section
    return base_prompt.replace(
        "**MEMORY QUERY ANALYSIS:**\n\nBased on my current context, what would help me find relevant past memories?\n\nGuidelines for conceptual_query:",
        kg_specific_section,
    )


def extract_kg_queries(
    state: State,
    trigger: BaseTrigger,
    trigger_history: TriggerHistory,
    available_entities: List[str],
    available_relationship_types: List[str],
    llm: LLM,
    model: SupportedModel,
) -> Optional[KGQueryExtraction]:
    """
    Extract KG queries from current context using structured LLM call.

    Args:
        state: Current agent state
        trigger: Current trigger being processed
        trigger_history: Agent's trigger history
        available_entities: List of entities available in the KG
        available_relationship_types: List of relationship types in the KG
        llm: LLM instance for making the call
        model: Model to use for extraction

    Returns:
        KGQueryExtraction with query details, or None if extraction fails
    """

    prompt = build_kg_query_extraction_prompt(
        state,
        trigger,
        trigger_history,
        available_entities,
        available_relationship_types,
    )

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=KGQueryExtraction,
            model=model,
            llm=llm,
            caller="kg_query_extraction",
            temperature=0.3,
        )

        logger.info(f"KG query extraction: should_query={response.should_query}")
        if response.should_query and response.kg_query:
            logger.info(f"Focus entities: {response.kg_query.focus_entities}")
            logger.info(f"Relationship types: {response.kg_query.relationship_types}")

        return response

    except Exception as e:
        logger.error(f"KG query extraction failed: {e}")
        return None


def test_kg_query_extraction():
    """Test the KG query extraction system"""

    logging.basicConfig(level=logging.INFO)

    from agent.conversation_persistence import ConversationPersistence
    from agent.llm import create_llm, SupportedModel
    from agent.chain_of_action.trigger import UserInputTrigger

    # Load baseline conversation data
    persistence = ConversationPersistence()
    trigger_history, state, initial_exchange = persistence.load_conversation("baseline")

    if state is None:
        print("❌ Could not load baseline state")
        return

    print(f"✅ Loaded baseline: {len(trigger_history.get_all_entries())} triggers")

    # Create test scenario
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    test_trigger = UserInputTrigger(content="How are you feeling about David lately?")

    # Mock available entities and relationships
    available_entities = [
        "David",
        "anxiety",
        "work",
        "programming",
        "friendship",
        "stress",
    ]
    available_relationship_types = [
        "involves",
        "caused",
        "relates_to",
        "triggered_emotion",
        "happened_before",
    ]

    print(f"\n🧪 Testing KG Query Extraction...")
    print(f"Test input: '{test_trigger.content}'")

    # Extract KG query
    extraction = extract_kg_queries(
        state=state,
        trigger=test_trigger,
        trigger_history=trigger_history,
        available_entities=available_entities,
        available_relationship_types=available_relationship_types,
        llm=llm,
        model=model,
    )

    if extraction:
        print(f"\n📊 Extraction Results:")
        print(f"Should query: {extraction.should_query}")
        print(f"Reasoning: {extraction.reasoning}")

        if extraction.kg_query:
            print(f"Focus entities: {extraction.kg_query.focus_entities}")
            print(f"Relationship types: {extraction.kg_query.relationship_types}")
            print(f"Max depth: {extraction.kg_query.max_depth}")
            print(f"Include recent: {extraction.kg_query.include_recent_experiences}")
            print(f"Temporal focus: {extraction.kg_query.temporal_focus}")
    else:
        print("❌ Extraction failed")

    print(f"\n✅ KG query extraction test completed!")


if __name__ == "__main__":
    test_kg_query_extraction()
