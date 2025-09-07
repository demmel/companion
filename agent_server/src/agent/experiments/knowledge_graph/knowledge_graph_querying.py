#!/usr/bin/env python3
"""
Knowledge Graph Querying and Context Construction

Implements intelligent querying of the knowledge graph to construct contextual
information for the agent. This replaces the current memory/context system by
dynamically traversing the graph based on current situation and recent activity.
"""

from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field
import logging

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.chain_of_action.trigger import UserInputTrigger
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.state import State
from agent.experiments.knowledge_graph.knowledge_graph_prototype import (
    KnowledgeExperienceGraph,
)

logger = logging.getLogger(__name__)


class GraphQuery(BaseModel):
    """A structured query for the knowledge graph"""

    focus_entities: List[str] = Field(
        default_factory=list, description="Main entities to focus retrieval on"
    )
    relationship_types: List[str] = Field(
        default_factory=list, description="Types of relationships to traverse"
    )
    max_depth: int = Field(
        default=2, description="Maximum depth to traverse from focus entities"
    )
    include_recent: bool = Field(
        default=True, description="Whether to include recent experiences"
    )
    include_historical: bool = Field(
        default=False,
        description="Whether to include historical/superseded relationships",
    )
    at_time: Optional[datetime] = Field(
        default=None,
        description="Time perspective for temporal queries (default: current time)",
    )
    context_purpose: str = Field(description="What this context will be used for")


class GraphQueryDetermination(BaseModel):
    """LLM-determined graph query based on current situation"""

    should_query: bool = Field(
        description="Whether querying the graph would be helpful"
    )
    query: Optional[GraphQuery] = Field(
        default=None, description="The graph query to execute"
    )
    reasoning: str = Field(description="Why this query approach was chosen")


@dataclass
class ContextEntity:
    """A relevant entity from the knowledge graph"""

    name: str
    node_type: str
    description: str
    confidence: float


@dataclass
class ContextRelationship:
    """A relevant relationship from the knowledge graph"""

    source: str
    target: str
    relationship_type: str
    description: str
    confidence: float


@dataclass
class ContextExperience:
    """A recent experience from the knowledge graph"""

    name: str
    description: str
    timestamp: Optional[datetime] = None


class ContextualInformation(BaseModel):
    """Information retrieved from the knowledge graph for context"""

    relevant_entities: List[ContextEntity] = Field(
        default_factory=list, description="Key entities and their descriptions"
    )
    relevant_relationships: List[ContextRelationship] = Field(
        default_factory=list, description="Important relationships between entities"
    )
    recent_experiences: List[ContextExperience] = Field(
        default_factory=list, description="Recent experience nodes"
    )
    patterns_and_insights: List[str] = Field(
        default_factory=list, description="Patterns noticed from the data"
    )
    emotional_context: List[str] = Field(
        default_factory=list, description="Emotional context from experiences"
    )
    commitments_and_promises: List[str] = Field(
        default_factory=list, description="Things I've committed to"
    )


class KnowledgeGraphQuerying:
    """
    Intelligent querying system for the knowledge graph.
    Determines what context is needed and constructs relevant information.
    """

    def __init__(
        self,
        graph: KnowledgeExperienceGraph,
        llm: LLM,
        model: SupportedModel,
        state: State,
    ):
        self.graph = graph
        self.llm = llm
        self.model = model
        self.state = state

    def determine_context_needs(
        self, current_input: str, recent_triggers: List[TriggerHistoryEntry]
    ) -> Optional[GraphQueryDetermination]:
        """Use LLM to determine what graph context would be helpful"""

        # Extract recent activity summary
        recent_activity = []
        for trigger in recent_triggers[-3:]:  # Last 3 triggers for context
            user_input = ""
            if isinstance(trigger.trigger, UserInputTrigger):
                user_input = trigger.trigger.content or ""

            if user_input:
                recent_activity.append(f"- User: {user_input[:100]}...")

        recent_activity_text = (
            "\n".join(recent_activity) if recent_activity else "No recent activity"
        )

        # Get available relationship types from graph
        available_relationships = self.graph.get_relationship_type_stats()
        relationship_types = list(available_relationships.keys())[
            :20
        ]  # Top 20 most common

        # Get entity names for reference
        all_entities = [
            node.name
            for node in self.graph.get_all_nodes()
            if node.node_type.value != "EXPERIENCE"
        ]
        entity_names = all_entities[:30]  # Top 30 entities

        prompt = f"""I am {self.state.name}. I need to decide what context from my knowledge graph would help me respond appropriately to the current situation.

CURRENT SITUATION:
User Input: {current_input}

RECENT ACTIVITY:
{recent_activity_text}

MY CURRENT STATE:
- Mood: {self.state.current_mood}
- Environment: {self.state.current_environment} 
- Appearance: {self.state.current_appearance}

AVAILABLE KNOWLEDGE:
- I have {len(all_entities)} entities in my knowledge graph
- Top entities: {', '.join(entity_names[:10])}...
- Available relationship types: {', '.join(relationship_types[:10])}...

Should I query my knowledge graph for context? If yes, what should I focus on?

Consider:
1. Does the user mention entities I might know about?
2. Are there patterns from recent interactions I should remember?
3. Would understanding relationships help me respond better?
4. Do I need to remember commitments, preferences, or emotional context?
5. Are there similar past experiences I should draw from?

If querying would be helpful, specify:
- focus_entities: Names of specific entities to center the search on
- relationship_types: Types of relationships to explore (from available types)
- max_depth: How far to traverse (1-3, where 1=direct connections only)
- context_purpose: What I'll use this context for
"""

        try:
            determination = direct_structured_llm_call(
                prompt=prompt,
                response_model=GraphQueryDetermination,
                model=self.model,
                llm=self.llm,
                caller="graph_query_determination",
                temperature=0.3,
            )

            return determination

        except Exception as e:
            logger.error(f"Graph query determination failed: {e}")
            return None

    def execute_graph_query(self, query: GraphQuery) -> ContextualInformation:
        """Execute a graph query and gather contextual information"""

        context = ContextualInformation()
        visited_nodes = set()

        # Start with focus entities
        current_layer = []
        for entity_name in query.focus_entities:
            node = self.graph.get_node_by_name(entity_name)
            if node and node.id not in visited_nodes:
                current_layer.append(node)
                visited_nodes.add(node.id)

                # Add entity to context
                context.relevant_entities.append(
                    ContextEntity(
                        name=node.name,
                        node_type=node.node_type.value,
                        description=node.description,
                        confidence=node.importance,
                    )
                )

        # Traverse relationships up to max_depth
        for depth in range(query.max_depth):
            next_layer = []

            for node in current_layer:
                # Get outgoing relationships
                outgoing = self.graph.get_relationships_from_node(node.id)
                # Get incoming relationships
                incoming = self.graph.get_relationships_to_node(node.id)

                all_relationships = outgoing + incoming

                # Filter to only relationships that were active at the query time
                at_time = query.at_time if query.at_time else datetime.now()
                active_relationships = [
                    rel
                    for rel in all_relationships
                    if rel
                    in self.graph.get_active_relationships(
                        at_time=at_time, include_historical=query.include_historical
                    )
                ]

                for rel in active_relationships:
                    # Filter by relationship type if specified
                    if (
                        query.relationship_types
                        and rel.relationship_type not in query.relationship_types
                    ):
                        continue

                    # Add relationship to context
                    source_node = self.graph.get_node(rel.source_node_id)
                    target_node = self.graph.get_node(rel.target_node_id)

                    if source_node and target_node:
                        context.relevant_relationships.append(
                            ContextRelationship(
                                source=source_node.name,
                                target=target_node.name,
                                relationship_type=rel.relationship_type,
                                description=rel.properties.get("description", ""),
                                confidence=rel.confidence,
                            )
                        )

                    # Add connected nodes for next layer
                    connected_node = None
                    if rel.source_node_id == node.id:
                        connected_node = self.graph.get_node(rel.target_node_id)
                    else:
                        connected_node = self.graph.get_node(rel.source_node_id)

                    if connected_node and connected_node.id not in visited_nodes:
                        next_layer.append(connected_node)
                        visited_nodes.add(connected_node.id)

                        # Add entity to context if not already there
                        entity_names = [e.name for e in context.relevant_entities]
                        if connected_node.name not in entity_names:
                            context.relevant_entities.append(
                                ContextEntity(
                                    name=connected_node.name,
                                    node_type=connected_node.node_type.value,
                                    description=connected_node.description,
                                    confidence=connected_node.importance,
                                )
                            )

            current_layer = next_layer
            if not current_layer:  # No more nodes to explore
                break

        # Include recent experiences if requested
        if query.include_recent:
            recent_experiences = [
                node
                for node in self.graph.get_all_nodes()
                if node.node_type.value == "EXPERIENCE"
            ]
            # Sort by created_at timestamp
            recent_experiences.sort(key=lambda x: x.created_at)

            for exp in recent_experiences[:5]:  # Last 5 experiences
                context.recent_experiences.append(
                    ContextExperience(
                        name=exp.name,
                        description=exp.description,
                        timestamp=exp.created_at,
                    )
                )

        # Extract patterns and insights
        context.patterns_and_insights = self._extract_patterns(context)

        # Extract emotional context
        context.emotional_context = self._extract_emotional_context(context)

        # Extract commitments
        context.commitments_and_promises = self._extract_commitments(context)

        return context

    def _extract_patterns(self, context: ContextualInformation) -> List[str]:
        """Extract patterns from the contextual information"""
        patterns = []

        # Look for repeated relationship types
        rel_type_counts = {}
        for rel in context.relevant_relationships:
            rel_type = rel.relationship_type
            rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1

        for rel_type, count in rel_type_counts.items():
            if count > 2:
                patterns.append(
                    f"Multiple {rel_type} relationships ({count} instances)"
                )

        # Look for emotional patterns
        emotions = []
        for entity in context.relevant_entities:
            if entity.node_type == "EMOTION":
                emotions.append(entity.name)

        if emotions:
            patterns.append(f"Emotional themes: {', '.join(emotions)}")

        return patterns

    def _extract_emotional_context(self, context: ContextualInformation) -> List[str]:
        """Extract emotional context from relationships and entities"""
        emotional_context = []

        # Find emotion entities and their relationships
        for entity in context.relevant_entities:
            if entity.node_type == "EMOTION":
                # Find what triggers this emotion
                related_rels = [
                    rel
                    for rel in context.relevant_relationships
                    if rel.target == entity.name or rel.source == entity.name
                ]

                if related_rels:
                    triggers = []
                    for rel in related_rels:
                        other_entity = (
                            rel.source if rel.target == entity.name else rel.target
                        )
                        triggers.append(f"{other_entity} {rel.relationship_type}")

                    emotional_context.append(
                        f"{entity.name}: {', '.join(triggers[:3])}"
                    )

        return emotional_context

    def _extract_commitments(self, context: ContextualInformation) -> List[str]:
        """Extract commitments and promises from the context"""
        commitments = []

        # Look for commitment-related relationships
        commitment_relationships = [
            "promised_to",
            "committed_to",
            "will_do",
            "agreed_to",
        ]

        for rel in context.relevant_relationships:
            if any(
                commitment in rel.relationship_type.lower()
                for commitment in commitment_relationships
            ):
                commitments.append(f"{rel.description}")
            elif rel.description and (
                "promise" in rel.description.lower()
                or "commit" in rel.description.lower()
            ):
                commitments.append(f"{rel.source} -> {rel.target}: {rel.description}")

        return commitments

    def construct_agent_context(
        self, current_input: str, recent_triggers: List[TriggerHistoryEntry]
    ) -> str:
        """Main method to construct context for the agent from the knowledge graph"""

        # Determine if we need graph context
        query_determination = self.determine_context_needs(
            current_input, recent_triggers
        )

        if (
            not query_determination
            or not query_determination.should_query
            or not query_determination.query
        ):
            return "No relevant context found in knowledge graph."

        # Execute the query
        context = self.execute_graph_query(query_determination.query)

        # Format context for agent
        context_sections = []

        if context.relevant_entities:
            context_sections.append("KEY ENTITIES:")
            for entity in context.relevant_entities[:8]:  # Top 8 entities
                context_sections.append(
                    f"- {entity.name} ({entity.node_type}): {entity.description}"
                )

        if context.relevant_relationships:
            context_sections.append("\nRELATIONSHIPS:")
            for rel in context.relevant_relationships[:10]:  # Top 10 relationships
                context_sections.append(
                    f"- {rel.source} --[{rel.relationship_type}]--> {rel.target}: {rel.description}"
                )

        if context.commitments_and_promises:
            context_sections.append("\nCOMMITMENTS & PROMISES:")
            for commitment in context.commitments_and_promises:
                context_sections.append(f"- {commitment}")

        if context.emotional_context:
            context_sections.append("\nEMOTIONAL CONTEXT:")
            for emotion in context.emotional_context:
                context_sections.append(f"- {emotion}")

        if context.patterns_and_insights:
            context_sections.append("\nPATTERNS & INSIGHTS:")
            for pattern in context.patterns_and_insights:
                context_sections.append(f"- {pattern}")

        if context.recent_experiences:
            context_sections.append("\nRECENT EXPERIENCES:")
            for exp in context.recent_experiences[:3]:  # Last 3 experiences
                context_sections.append(f"- {exp.name}: {exp.description[:100]}...")

        context_text = "\n".join(context_sections)

        # Add query reasoning
        context_header = (
            f"KNOWLEDGE GRAPH CONTEXT (Reasoning: {query_determination.reasoning}):\n\n"
        )

        return context_header + context_text

    def construct_temporal_context(
        self,
        focus_entities: List[str],
        at_time: datetime,
        include_evolution_history: bool = False,
    ) -> str:
        """
        Construct context for a specific point in time with temporal awareness.

        Args:
            focus_entities: Entities to focus the query on
            at_time: The time perspective to use
            include_evolution_history: Whether to include relationship evolution history

        Returns:
            Formatted temporal context string
        """

        # Create temporal query
        query = GraphQuery(
            focus_entities=focus_entities,
            at_time=at_time,
            include_historical=include_evolution_history,
            context_purpose=f"Temporal context at {at_time.strftime('%Y-%m-%d %H:%M')}",
        )

        context = self.execute_graph_query(query)

        # Build temporal-aware context
        sections = []

        # Add temporal header
        sections.append(f"CONTEXT AT {at_time.strftime('%Y-%m-%d %H:%M')}:")

        if context.relevant_entities:
            sections.append("\nRELEVANT ENTITIES:")
            for entity in context.relevant_entities[:10]:
                sections.append(
                    f"- {entity.name} ({entity.node_type}): {entity.description}"
                )

        if context.relevant_relationships:
            sections.append("\nKEY RELATIONSHIPS:")
            for rel in context.relevant_relationships[:15]:
                sections.append(
                    f"- {rel.source} --[{rel.relationship_type}]--> {rel.target}"
                )
                if rel.description:
                    sections.append(f"  {rel.description}")

        # If including evolution history, add supersession information
        if include_evolution_history:
            sections.append("\nRELATIONSHIP EVOLUTION:")
            evolution_info = []

            for entity_name in focus_entities:
                node = self.graph.get_node_by_name(entity_name)
                if not node:
                    continue

                # Get relationships involving this entity
                outgoing = self.graph.get_relationships_from_node(node.id)
                incoming = self.graph.get_relationships_to_node(node.id)
                all_rels = outgoing + incoming

                # Group by supersession chains
                superseded_rels = [r for r in all_rels if r.superseded_by]
                if superseded_rels:
                    evolution_info.append(f"\nEvolution for {entity_name}:")
                    for old_rel in superseded_rels:
                        source = self.graph.get_node(old_rel.source_node_id)
                        target = self.graph.get_node(old_rel.target_node_id)
                        if source and target:
                            evolution_info.append(
                                f"- Was: {source.name} --[{old_rel.relationship_type}]--> {target.name} "
                                f"({old_rel.lifecycle_state.value})"
                            )

                            # Find what replaced it
                            superseded_by = old_rel.superseded_by
                            assert (
                                superseded_by is not None
                            ), "list constructed by superseded_by check, should not be None"
                            new_rel = self.graph.relationships.get(superseded_by)
                            if new_rel:
                                new_source = self.graph.get_node(new_rel.source_node_id)
                                new_target = self.graph.get_node(new_rel.target_node_id)
                                if new_source and new_target:
                                    evolution_info.append(
                                        f"  Now: {new_source.name} --[{new_rel.relationship_type}]--> {new_target.name}"
                                    )

            if evolution_info:
                sections.extend(evolution_info)
            else:
                sections.append("- No relationship evolution history found")

        return "\n".join(sections)


def test_knowledge_graph_querying():
    """Test the knowledge graph querying system"""

    logging.basicConfig(level=logging.INFO)

    from agent.conversation_persistence import ConversationPersistence
    from agent.llm import create_llm, SupportedModel
    from agent.experiments.knowledge_graph.knowledge_graph_builder import (
        ValidatedKnowledgeGraphBuilder,
    )

    # Load baseline conversation data
    persistence = ConversationPersistence()
    trigger_history, state, initial_exchange = persistence.load_conversation("baseline")

    if state is None:
        print("❌ Could not load baseline state")
        return

    print(f"✅ Loaded baseline: {len(trigger_history.get_all_entries())} triggers")

    # Build a small knowledge graph for testing
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    print(f"\n🏗️  Building test knowledge graph...")
    builder = ValidatedKnowledgeGraphBuilder(llm, model, state)

    # Process first 5 triggers to create a testable graph
    test_triggers = trigger_history.get_all_entries()[:5]
    for trigger in test_triggers:
        builder.add_trigger(trigger)

    graph = builder.graph
    print(
        f"Built graph: {len(graph.get_all_nodes())} nodes, {len(graph.get_all_relationships())} relationships"
    )

    # Create querying system
    querying = KnowledgeGraphQuerying(graph, llm, model, state)

    # Test different query scenarios
    test_scenarios = [
        ("Hey, what do you think about David?", "Test: Direct person mention"),
        ("I'm feeling anxious about work", "Test: Emotional context"),
        ("What did we talk about recently?", "Test: Recent experience query"),
        ("Let's discuss anime", "Test: Topic/concept mention"),
        ("How are you doing?", "Test: General greeting"),
    ]

    print(f"\n🧪 Testing Knowledge Graph Querying...")

    recent_triggers = trigger_history.get_all_entries()[-3:]  # Last 3 for context

    for input_text, description in test_scenarios:
        print(f"\n--- {description} ---")
        print(f"Input: {input_text}")

        # Get agent context
        context = querying.construct_agent_context(input_text, recent_triggers)

        print(f"Context Generated:")
        print(f"{context}")
        print()

    # Show graph statistics
    print(f"\n📊 Graph Statistics:")
    node_types = {}
    for node in graph.get_all_nodes():
        node_types[node.node_type.value] = node_types.get(node.node_type.value, 0) + 1

    for node_type, count in node_types.items():
        print(f"  {node_type}: {count} nodes")

    rel_stats = graph.get_relationship_type_stats()
    print(f"\nTop Relationship Types:")
    for rel_type, count in list(rel_stats.items())[:5]:
        print(f"  {rel_type}: {count} relationships")

    print(f"\n✅ Knowledge graph querying test completed!")


if __name__ == "__main__":
    test_knowledge_graph_querying()
