#!/usr/bin/env python3
"""
Knowledge+Experience Graph Prototype

Standalone prototype that builds a knowledge+experience graph from trigger history
using incremental processing (no future knowledge) with confidence-weighted relationships.
"""

import uuid
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Removed pydantic import - not needed for prototype
import logging

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.chain_of_action.trigger import UserInputTrigger
from agent.chain_of_action.action.action_types import ActionType
from agent.conversation_persistence import ConversationPersistence
from agent.llm import LLM, SupportedModel, create_llm

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of nodes in the knowledge+experience graph"""

    # Experience nodes - preserve rich trigger context
    EXPERIENCE = "experience"

    # Knowledge nodes - abstract concepts extracted from experiences
    CONCEPT = "concept"  # Ideas, topics, skills
    PERSON = "person"  # People agent interacts with
    EMOTION = "emotion"  # Emotional states and patterns
    GOAL = "goal"  # Intentions, objectives
    PATTERN = "pattern"  # Behavioral/learning patterns
    CONTEXT = "context"  # Situational contexts


class RelationshipType(str, Enum):
    """Types of relationships with confidence implications"""

    # High-confidence structural relationships
    HAPPENED_BEFORE = "happened_before"  # Temporal from timestamps
    HAPPENED_AFTER = "happened_after"  # Temporal from timestamps
    USER_SAID = "user_said"  # From structured data
    AGENT_SAID = "agent_said"  # From structured data
    AGENT_DID = "agent_did"  # From action logs
    AGENT_FELT = "agent_felt"  # From agent's emotional reports

    # Variable-confidence LLM-inferred relationships
    CAUSED = "caused"  # Causal reasoning
    ENABLED = "enabled"  # Learning/capability connections
    RELATES_TO = "relates_to"  # Conceptual associations
    INVOLVES = "involves"  # Participation/inclusion
    SIMILAR_TO = "similar_to"  # Pattern recognition
    TRIGGERED_EMOTION = "triggered_emotion"  # Emotional causation
    CONTRIBUTED_TO = "contributed_to"  # Partial causation


@dataclass
class GraphNode:
    """A node in the knowledge+experience graph"""

    id: str
    node_type: NodeType
    name: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 1.0

    # For experience nodes - preserve full trigger context
    source_trigger_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "description": self.description,
            "properties": self.properties,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "source_trigger_id": self.source_trigger_id,
        }


@dataclass
class GraphRelationship:
    """A relationship between nodes with confidence scoring"""

    id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str  # Dynamic relationship types instead of enum
    confidence: float  # 0.0 to 1.0
    strength: float = 1.0  # Relationship strength
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)
    reinforcement_count: int = 1
    properties: Dict[str, Any] = field(default_factory=dict)
    source_trigger_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat(),
            "reinforcement_count": self.reinforcement_count,
            "properties": self.properties,
            "source_trigger_id": self.source_trigger_id,
        }


class KnowledgeExperienceGraph:
    """Knowledge+Experience Graph with confidence-weighted relationships"""

    def __init__(self):
        # Core storage
        self.nodes: Dict[str, GraphNode] = {}
        self.relationships: Dict[str, GraphRelationship] = {}

        # Indexes for efficient lookup
        self.nodes_by_type: Dict[NodeType, Set[str]] = {nt: set() for nt in NodeType}
        self.relationships_by_source: Dict[str, Set[str]] = {}
        self.relationships_by_target: Dict[str, Set[str]] = {}
        self.relationships_by_type: Dict[str, Set[str]] = {}

        # Processing history
        self.processed_triggers: Set[str] = set()

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.nodes_by_type[node.node_type].add(node.id)
        logger.debug(f"Added node: {node.node_type.value} - {node.name}")

    def add_relationship(self, relationship: GraphRelationship) -> None:
        """Add a relationship to the graph"""
        self.relationships[relationship.id] = relationship

        # Update indexes
        self.relationships_by_source.setdefault(relationship.source_node_id, set()).add(
            relationship.id
        )
        self.relationships_by_target.setdefault(relationship.target_node_id, set()).add(
            relationship.id
        )
        self.relationships_by_type.setdefault(
            relationship.relationship_type, set()
        ).add(relationship.id)

        logger.debug(
            f"Added relationship: {relationship.relationship_type} "
            f"(confidence: {relationship.confidence:.2f})"
        )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def find_nodes_by_name(
        self, name: str, node_type: Optional[NodeType] = None
    ) -> List[GraphNode]:
        """Find nodes by name (case-insensitive)"""
        matches = []
        candidate_nodes = (
            self.nodes_by_type.get(node_type, set()) if node_type else self.nodes.keys()
        )

        for node_id in candidate_nodes:
            node = self.nodes[node_id]
            if name.lower() in node.name.lower():
                matches.append(node)

        return matches

    def get_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
        min_confidence: float = 0.0,
    ) -> List[Tuple[GraphNode, GraphRelationship]]:
        """Get neighboring nodes through specified relationships"""
        neighbors = []

        # Get outgoing relationships
        if direction in ["both", "out"]:
            for rel_id in self.relationships_by_source.get(node_id, set()):
                rel = self.relationships[rel_id]
                if (
                    not relationship_types
                    or rel.relationship_type in relationship_types
                ) and rel.confidence >= min_confidence:
                    target_node = self.nodes.get(rel.target_node_id)
                    if target_node:
                        neighbors.append((target_node, rel))

        # Get incoming relationships
        if direction in ["both", "in"]:
            for rel_id in self.relationships_by_target.get(node_id, set()):
                rel = self.relationships[rel_id]
                if (
                    not relationship_types
                    or rel.relationship_type in relationship_types
                ) and rel.confidence >= min_confidence:
                    source_node = self.nodes.get(rel.source_node_id)
                    if source_node:
                        neighbors.append((source_node, rel))

        return neighbors

    def get_all_nodes(self) -> List[GraphNode]:
        """Get all nodes in the graph"""
        return list(self.nodes.values())

    def get_all_relationships(self) -> List[GraphRelationship]:
        """Get all relationships in the graph"""
        return list(self.relationships.values())

    def get_node_by_name(self, name: str) -> Optional[GraphNode]:
        """Get first node with matching name (case-insensitive)"""
        name_lower = name.lower()
        for node in self.nodes.values():
            if node.name.lower() == name_lower:
                return node
        return None

    def get_relationships_from_node(self, node_id: str) -> List[GraphRelationship]:
        """Get all relationships originating from a node"""
        if node_id not in self.relationships_by_source:
            return []
        return [
            self.relationships[rel_id]
            for rel_id in self.relationships_by_source[node_id]
        ]

    def get_relationships_to_node(self, node_id: str) -> List[GraphRelationship]:
        """Get all relationships targeting a node"""
        if node_id not in self.relationships_by_target:
            return []
        return [
            self.relationships[rel_id]
            for rel_id in self.relationships_by_target[node_id]
        ]

    def get_relationship_type_stats(self) -> Dict[str, int]:
        """Get counts of each relationship type"""
        rel_type_counts = {}
        for rel in self.relationships.values():
            rel_type_counts[rel.relationship_type] = (
                rel_type_counts.get(rel.relationship_type, 0) + 1
            )
        return rel_type_counts

    def export_to_dict(self) -> Dict[str, Any]:
        """Export entire graph to dictionary"""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "relationships": [rel.to_dict() for rel in self.relationships.values()],
            "processed_triggers": list(self.processed_triggers),
            "stats": self.get_stats(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        node_counts = {
            nt.value: len(nodes)
            for nt, nodes in self.nodes_by_type.items()
            if len(nodes) > 0
        }
        rel_counts = {
            rt: len(rels)
            for rt, rels in self.relationships_by_type.items()
            if len(rels) > 0
        }

        # Confidence distribution
        confidences = [rel.confidence for rel in self.relationships.values()]
        confidence_dist = {}
        if confidences:
            confidence_dist = {
                "mean": sum(confidences) / len(confidences),
                "high_confidence_count": sum(1 for c in confidences if c >= 0.8),
                "medium_confidence_count": sum(
                    1 for c in confidences if 0.5 <= c < 0.8
                ),
                "low_confidence_count": sum(1 for c in confidences if c < 0.5),
            }

        return {
            "total_nodes": len(self.nodes),
            "total_relationships": len(self.relationships),
            "processed_triggers": len(self.processed_triggers),
            "node_types": node_counts,
            "relationship_types": rel_counts,
            "confidence_distribution": confidence_dist,
        }

    def save_to_file(self, filename: str) -> None:
        """Save graph to JSON file"""
        data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "relationships": [rel.to_dict() for rel in self.relationships.values()],
            "processed_triggers": list(self.processed_triggers),
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved graph to {filename}")

    @classmethod
    def load_from_file(cls, filename: str) -> "KnowledgeExperienceGraph":
        """Load graph from JSON file"""
        with open(filename, "r") as f:
            data = json.load(f)

        graph = cls()

        # Load nodes
        for node_data in data["nodes"]:
            node = GraphNode(
                id=node_data["id"],
                node_type=NodeType(node_data["node_type"]),
                name=node_data["name"],
                description=node_data["description"],
                importance=node_data["importance"],
                created_at=datetime.fromisoformat(node_data["created_at"]),
                properties=node_data.get("properties", {}),
            )
            graph.add_node(node)

        # Load relationships
        for rel_data in data["relationships"]:
            # Handle optional fields that may not exist in older saves
            created_at = datetime.now()
            if "created_at" in rel_data:
                if isinstance(rel_data["created_at"], str):
                    created_at = datetime.fromisoformat(rel_data["created_at"])
                else:
                    created_at = rel_data["created_at"]

            rel = GraphRelationship(
                id=rel_data["id"],
                source_node_id=rel_data["source_node_id"],
                target_node_id=rel_data["target_node_id"],
                relationship_type=rel_data["relationship_type"],
                confidence=rel_data["confidence"],
                strength=rel_data.get("strength", 1.0),
                created_at=created_at,
                last_reinforced=created_at,
                reinforcement_count=rel_data.get("reinforcement_count", 1),
                properties=rel_data.get("properties", {}),
                source_trigger_id=rel_data.get("source_trigger_id"),
            )
            graph.add_relationship(rel)

        # Load processed triggers
        graph.processed_triggers = set(data.get("processed_triggers", []))

        logger.info(f"Loaded graph from {filename}")
        return graph


class TriggerProcessor:
    """Processes triggers incrementally to build the knowledge+experience graph"""

    def __init__(
        self, graph: KnowledgeExperienceGraph, llm: LLM, model: SupportedModel
    ):
        self.graph = graph
        self.llm = llm
        self.model = model

    def process_trigger(
        self,
        trigger: TriggerHistoryEntry,
        previous_trigger: Optional[TriggerHistoryEntry] = None,
    ) -> None:
        """Process a single trigger incrementally to update the graph"""

        if trigger.entry_id in self.graph.processed_triggers:
            logger.debug(f"Trigger {trigger.entry_id} already processed, skipping")
            return

        logger.info(f"Processing trigger: {trigger.entry_id} at {trigger.timestamp}")

        # Always create an experience node for the trigger
        experience_node = self._create_experience_node(trigger)
        self.graph.add_node(experience_node)

        # Add high-confidence structural relationships
        self._add_structural_relationships(trigger, previous_trigger, experience_node)

        # Extract and add knowledge nodes and relationships using LLM
        self._extract_knowledge_elements(trigger, experience_node)

        # Mark as processed
        self.graph.processed_triggers.add(trigger.entry_id)

    def _create_experience_node(self, trigger: TriggerHistoryEntry) -> GraphNode:
        """Create an experience node preserving the full trigger context"""

        # Create a meaningful name for the experience
        user_input = (
            trigger.trigger.content
            if isinstance(trigger.trigger, UserInputTrigger)
            else None
        )
        if user_input:
            name = f"User interaction: {user_input[:50]}..."
        else:
            name = f"Experience at {trigger.timestamp}"

        # Rich description including all context
        description_parts = []
        if user_input:
            description_parts.append(f"User said: {user_input}")
        if trigger.actions_taken:
            action_summaries = []
            for action in trigger.actions_taken:
                if action.type == ActionType.THINK:
                    if action.input.focus:
                        action_summaries.append(f"Thought: {action.input.focus}")
                elif action.type == ActionType.SPEAK:
                    if action.result and action.result.type == "success":
                        response = action.result.content.response
                        if response:
                            action_summaries.append(f"Responded: {response}")
            if action_summaries:
                description_parts.append("Actions: " + "; ".join(action_summaries))

        description = (
            "; ".join(description_parts) if description_parts else "Experience"
        )

        return GraphNode(
            id=f"exp_{trigger.entry_id}",
            node_type=NodeType.EXPERIENCE,
            name=name,
            description=description,
            properties={
                "timestamp": trigger.timestamp.isoformat(),
                "trigger_type": type(trigger.trigger).__name__,
                "user_input": (
                    trigger.trigger.content
                    if isinstance(trigger.trigger, UserInputTrigger)
                    else ""
                ),
                "actions": [(action.model_dump()) for action in trigger.actions_taken],
            },
            source_trigger_id=trigger.entry_id,
        )

    def _add_structural_relationships(
        self,
        trigger: TriggerHistoryEntry,
        previous_trigger: Optional[TriggerHistoryEntry],
        experience_node: GraphNode,
    ) -> None:
        """Add high-confidence structural relationships"""

        # Temporal relationships with previous trigger
        if previous_trigger:
            prev_exp_id = f"exp_{previous_trigger.entry_id}"
            if prev_exp_id in self.graph.nodes:
                # Previous experience happened before current
                rel = GraphRelationship(
                    id=str(uuid.uuid4()),
                    source_node_id=prev_exp_id,
                    target_node_id=experience_node.id,
                    relationship_type=RelationshipType.HAPPENED_BEFORE,
                    confidence=1.0,  # Perfect confidence from timestamps
                    source_trigger_id=trigger.entry_id,
                )
                self.graph.add_relationship(rel)

        # TODO: Add more structural relationships
        # - USER_SAID relationships from user_input
        # - AGENT_DID relationships from actions
        # - AGENT_SAID relationships from speak actions
        # - AGENT_FELT relationships from emotional content

    def _extract_knowledge_elements(
        self, trigger: TriggerHistoryEntry, experience_node: GraphNode
    ) -> None:
        """Extract knowledge nodes and relationships using LLM (placeholder for now)"""

        # TODO: Implement LLM-based extraction
        # For now, just add some basic knowledge extraction as examples

        # Extract person mentions (simple regex for prototype)
        user_input = (
            trigger.trigger.content
            if isinstance(trigger.trigger, UserInputTrigger)
            else None
        )
        if user_input:
            # Look for name patterns (capital letter followed by lowercase)
            import re

            name_pattern = r"\b[A-Z][a-z]+\b"
            potential_names = re.findall(name_pattern, user_input)

            for name in potential_names:
                if len(name) > 2 and name not in [
                    "The",
                    "This",
                    "That",
                    "When",
                    "Where",
                ]:  # Filter common words
                    # Check if person node already exists
                    existing_persons = self.graph.find_nodes_by_name(
                        name, NodeType.PERSON
                    )

                    if existing_persons:
                        person_node = existing_persons[0]
                    else:
                        # Create new person node
                        person_node = GraphNode(
                            id=f"person_{name.lower()}_{str(uuid.uuid4())[:8]}",
                            node_type=NodeType.PERSON,
                            name=name,
                            description=f"Person named {name}",
                            properties={"mentioned_in_triggers": [trigger.entry_id]},
                        )
                        self.graph.add_node(person_node)

                    # Add relationship: experience involves person
                    rel = GraphRelationship(
                        id=str(uuid.uuid4()),
                        source_node_id=experience_node.id,
                        target_node_id=person_node.id,
                        relationship_type=RelationshipType.INVOLVES,
                        confidence=0.7,  # Medium confidence from simple pattern matching
                        source_trigger_id=trigger.entry_id,
                    )
                    self.graph.add_relationship(rel)


def main():
    """Main prototype runner"""

    logging.basicConfig(level=logging.INFO)

    # Load baseline conversation data
    persistence = ConversationPersistence()
    trigger_history, state, initial_exchange = persistence.load_conversation("baseline")

    if state is None:
        print("❌ Could not load baseline state")
        return

    print(f"✅ Loaded baseline: {len(trigger_history.get_all_entries())} triggers")
    print(f"📊 Agent: {state.name} - {state.current_mood}")

    # Create LLM (for future knowledge extraction)
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    # Initialize graph and processor
    graph = KnowledgeExperienceGraph()
    processor = TriggerProcessor(graph, llm, model)

    # Process triggers incrementally
    all_triggers = trigger_history.get_all_entries()
    print(f"\n🔄 Processing {len(all_triggers)} triggers incrementally...")

    previous_trigger = None
    for i, trigger in enumerate(all_triggers):
        print(f"  Processing trigger {i+1}/{len(all_triggers)}: {trigger.entry_id}")
        processor.process_trigger(trigger, previous_trigger)
        previous_trigger = trigger

        # Show progress every 10 triggers
        if (i + 1) % 10 == 0:
            stats = graph.get_stats()
            print(
                f"    Graph stats: {stats['total_nodes']} nodes, {stats['total_relationships']} relationships"
            )

    # Final statistics
    print("\n📈 Final Graph Statistics:")
    stats = graph.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")

    # Save the graph
    graph.save_to_file("knowledge_experience_graph.json")
    print("\n💾 Graph saved to knowledge_experience_graph.json")


if __name__ == "__main__":
    main()
