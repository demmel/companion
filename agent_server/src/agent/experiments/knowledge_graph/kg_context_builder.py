#!/usr/bin/env python3
"""
Knowledge Graph Context Prompt Builder

Experimental component for building agent context prompts from KG query results.
Tests different formatting approaches to find what works best for agent situational analysis.
"""

import logging
from typing import List, Dict, Optional
from enum import Enum

from agent.state import State
from agent.experiments.knowledge_graph.knowledge_graph_prototype import (
    KnowledgeExperienceGraph,
)
from agent.experiments.knowledge_graph.knowledge_graph_querying import (
    GraphQuery,
    ContextualInformation,
    ContextEntity,
    ContextRelationship,
    ContextExperience,
)

logger = logging.getLogger(__name__)


class ContextFormat(str, Enum):
    """Different experimental context formatting approaches"""

    STRUCTURED = "structured"  # Organized sections with headers
    NARRATIVE = "narrative"  # Flowing narrative format
    BULLET_POINTS = "bullet_points"  # Simple bullet point lists
    RELATIONSHIP_FOCUSED = "relationship_focused"  # Emphasizes connections
    TEMPORAL = "temporal"  # Chronological organization
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Sorted by confidence scores


class KGContextBuilder:
    """
    Experimental context builder for testing different KG context formatting approaches.
    """

    def __init__(self, graph: KnowledgeExperienceGraph, state: State):
        self.graph = graph
        self.state = state

    def build_context_from_kg_query(
        self,
        kg_query: GraphQuery,
        format_type: ContextFormat = ContextFormat.STRUCTURED,
        max_context_length: int = 2000,
    ) -> str:
        """
        Build agent context from KG query using specified formatting approach.

        Args:
            kg_query: The KG query specifying what to retrieve
            format_type: How to format the context
            max_context_length: Maximum length of context in characters

        Returns:
            Formatted context string for agent situational analysis
        """

        # Execute the KG query to get contextual information
        contextual_info = self._execute_kg_query(kg_query)

        if not contextual_info:
            return "No relevant knowledge found in graph for this query."

        # Format based on the specified approach
        if format_type == ContextFormat.STRUCTURED:
            context = self._format_structured(contextual_info)
        elif format_type == ContextFormat.NARRATIVE:
            context = self._format_narrative(contextual_info)
        elif format_type == ContextFormat.BULLET_POINTS:
            context = self._format_bullet_points(contextual_info)
        elif format_type == ContextFormat.RELATIONSHIP_FOCUSED:
            context = self._format_relationship_focused(contextual_info)
        elif format_type == ContextFormat.TEMPORAL:
            context = self._format_temporal(contextual_info)
        elif format_type == ContextFormat.CONFIDENCE_WEIGHTED:
            context = self._format_confidence_weighted(contextual_info)
        else:
            context = self._format_structured(contextual_info)  # Default

        # Truncate if too long
        if len(context) > max_context_length:
            context = context[: max_context_length - 3] + "..."

        return context

    def _execute_kg_query(
        self, kg_query: GraphQuery
    ) -> Optional[ContextualInformation]:
        """Execute a KG query and return contextual information"""

        context = ContextualInformation()
        visited_nodes = set()

        # Start with focus entities
        current_layer = []
        for entity_name in kg_query.focus_entities:
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
        for depth in range(kg_query.max_depth):
            next_layer = []

            for node in current_layer:
                # Get connected relationships
                outgoing = self.graph.get_relationships_from_node(node.id)
                incoming = self.graph.get_relationships_to_node(node.id)
                all_relationships = outgoing + incoming

                for rel in all_relationships:
                    # Filter by relationship type if specified
                    if (
                        kg_query.relationship_types
                        and rel.relationship_type not in kg_query.relationship_types
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

                        # Add entity to context
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
            if not current_layer:
                break

        # Include recent experiences if requested
        if kg_query.include_recent:
            recent_experiences = [
                node
                for node in self.graph.get_all_nodes()
                if node.node_type.value == "EXPERIENCE"
            ]
            recent_experiences.sort(key=lambda x: x.created_at, reverse=True)

            for exp in recent_experiences[:5]:
                context.recent_experiences.append(
                    ContextExperience(
                        name=exp.name,
                        description=exp.description,
                        timestamp=exp.created_at,
                    )
                )

        return (
            context
            if (context.relevant_entities or context.relevant_relationships)
            else None
        )

    def _format_structured(self, context: ContextualInformation) -> str:
        """Format context with clear sections and headers"""
        sections = []

        if context.relevant_entities:
            sections.append("RELEVANT KNOWLEDGE:")
            for entity in sorted(
                context.relevant_entities, key=lambda x: x.confidence, reverse=True
            )[:8]:
                sections.append(
                    f"• {entity.name} ({entity.node_type}): {entity.description}"
                )

        if context.relevant_relationships:
            sections.append("\nKEY CONNECTIONS:")
            for rel in sorted(
                context.relevant_relationships, key=lambda x: x.confidence, reverse=True
            )[:8]:
                sections.append(
                    f"• {rel.source} --[{rel.relationship_type}]--> {rel.target}"
                )
                if rel.description:
                    sections.append(f"  {rel.description}")

        if context.recent_experiences:
            sections.append("\nRECENT RELEVANT EXPERIENCES:")
            for exp in context.recent_experiences[:3]:
                sections.append(f"• {exp.name}: {exp.description[:100]}...")

        return "\n".join(sections)

    def _format_narrative(self, context: ContextualInformation) -> str:
        """Format context as a flowing narrative"""
        parts = []

        # Start with key entities
        if context.relevant_entities:
            entity_names = [e.name for e in context.relevant_entities[:5]]
            parts.append(
                f"In this context, the key elements involve {', '.join(entity_names[:-1])}"
            )
            if len(entity_names) > 1:
                parts[-1] += f" and {entity_names[-1]}."
            else:
                parts[-1] += "."

        # Add relationship insights
        if context.relevant_relationships:
            high_conf_rels = [
                r for r in context.relevant_relationships if r.confidence > 0.7
            ]
            if high_conf_rels:
                rel = high_conf_rels[0]  # Use highest confidence relationship
                parts.append(
                    f"There's a strong connection where {rel.source} {rel.relationship_type.replace('_', ' ')} {rel.target}."
                )

        # Add recent context
        if context.recent_experiences:
            parts.append(
                f"Recent experiences include {context.recent_experiences[0].name.lower()}."
            )

        return " ".join(parts)

    def _format_bullet_points(self, context: ContextualInformation) -> str:
        """Simple bullet point format"""
        points = []

        for entity in context.relevant_entities[:6]:
            points.append(f"• {entity.name}: {entity.description}")

        for rel in context.relevant_relationships[:6]:
            points.append(f"• {rel.source} → {rel.target} ({rel.relationship_type})")

        return "\n".join(points)

    def _format_relationship_focused(self, context: ContextualInformation) -> str:
        """Emphasize relationships and connections"""
        lines = []

        # Group relationships by type
        rel_groups: Dict[str, List[ContextRelationship]] = {}
        for rel in context.relevant_relationships:
            if rel.relationship_type not in rel_groups:
                rel_groups[rel.relationship_type] = []
            rel_groups[rel.relationship_type].append(rel)

        for rel_type, rels in rel_groups.items():
            lines.append(f"{rel_type.upper().replace('_', ' ')} CONNECTIONS:")
            for rel in rels[:4]:  # Top 4 per type
                lines.append(f"  {rel.source} ←→ {rel.target}")

        return "\n".join(lines)

    def _format_temporal(self, context: ContextualInformation) -> str:
        """Organize by temporal information"""
        lines = []

        if context.recent_experiences:
            lines.append("RECENT TIMELINE:")
            for exp in sorted(
                context.recent_experiences, key=lambda x: x.timestamp or 0, reverse=True
            ):
                lines.append(f"• {exp.name}: {exp.description[:80]}...")

        # Add other context
        if context.relevant_entities:
            lines.append("\nRELEVANT BACKGROUND:")
            for entity in context.relevant_entities[:5]:
                lines.append(f"• {entity.name}: {entity.description}")

        return "\n".join(lines)

    def _format_confidence_weighted(self, context: ContextualInformation) -> str:
        """Sort everything by confidence scores"""
        lines = ["HIGH CONFIDENCE KNOWLEDGE:"]

        # Combine entities and relationships, sort by confidence
        all_items = []

        for entity in context.relevant_entities:
            all_items.append(
                (entity.confidence, f"ENTITY: {entity.name} - {entity.description}")
            )

        for rel in context.relevant_relationships:
            all_items.append(
                (
                    rel.confidence,
                    f"CONNECTION: {rel.source} --[{rel.relationship_type}]--> {rel.target}",
                )
            )

        # Sort by confidence and take top items
        all_items.sort(key=lambda x: x[0], reverse=True)
        for confidence, description in all_items[:10]:
            lines.append(f"• ({confidence:.1f}) {description}")

        return "\n".join(lines)


def test_context_formats():
    """Test different context formatting approaches"""

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

    # Build a small KG for testing
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    builder = ValidatedKnowledgeGraphBuilder(llm, model, state)

    # Process a few triggers to create test data
    test_triggers = trigger_history.get_all_entries()[:5]
    for trigger in test_triggers:
        builder.add_trigger(trigger)

    graph = builder.graph
    print(
        f"Built test graph: {len(graph.get_all_nodes())} nodes, {len(graph.get_all_relationships())} relationships"
    )

    # Create context builder
    context_builder = KGContextBuilder(graph, state)

    # Test different formats with a sample query
    test_query = GraphQuery(
        focus_entities=["David", "anxiety"],
        relationship_types=["involves", "caused", "relates_to"],
        max_depth=2,
        include_recent=True,
        context_purpose="testing different context formats",
    )

    print(f"\n🧪 Testing Context Formats...")
    print(f"Test query: focus on David & anxiety, depth=2")
    print("=" * 60)

    for format_type in ContextFormat:
        print(f"\n--- {format_type.value.upper()} FORMAT ---")
        context = context_builder.build_context_from_kg_query(test_query, format_type)
        print(context)
        print(f"Length: {len(context)} chars")

    print(f"\n✅ Context format testing completed!")


if __name__ == "__main__":
    test_context_formats()
