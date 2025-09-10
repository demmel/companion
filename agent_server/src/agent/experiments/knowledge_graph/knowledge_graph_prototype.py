#!/usr/bin/env python3
"""
Knowledge+Experience Graph Prototype

Standalone prototype that builds a knowledge+experience graph from trigger history
using incremental processing (no future knowledge) with confidence-weighted relationships.
"""

import uuid
import json
from typing import Dict, List, Optional, Set, Tuple, Any, Type
from datetime import datetime
from enum import Enum

from agent.experiments.knowledge_graph.knn_entity_search import IKNNEntity
import numpy as np
from pydantic import BaseModel, Field


class ConfidenceDistribution(BaseModel):
    mean: float = 0.0
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0


class GraphStats(BaseModel):
    total_nodes: int
    total_relationships: int
    processed_triggers: int
    node_types: Dict[str, int]
    relationship_types: Dict[str, int]
    confidence_distribution: ConfidenceDistribution


# KnowledgeGraphData will be defined after imports are resolved
# to avoid forward reference issues
_KnowledgeGraphData: Any = None


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
    OBJECT = "object"  # Physical objects, tools, places, tangible items


class RelationshipLifecycleState(str, Enum):
    """Lifecycle states for temporal relationship management"""

    ACTIVE = "active"  # Currently valid relationship
    HISTORICAL = "historical"  # Was valid in the past, superseded by new information
    DEPRECATED = "deprecated"  # Was incorrect, should not have existed
    SUPERSEDED = "superseded"  # Was correct but replaced by evolved knowledge


class ChangeType(str, Enum):
    """Types of knowledge changes for temporal management"""

    CORRECTION = (
        "correction"  # Previous information was wrong, needs to be marked as deprecated
    )
    EVOLUTION = "evolution"  # Information changed over time, previous was correct then


class ChangeDetectionResult(BaseModel):
    """Result of analyzing whether new information is correction or evolution"""

    change_type: ChangeType = Field(
        description="Whether this is a correction or evolution"
    )
    reasoning: str = Field(description="Why this classification was made")
    confidence: float = Field(description="Confidence in this classification (0.0-1.0)")
    should_supersede: bool = Field(
        description="Whether existing relationships should be superseded"
    )
    temporal_context: str = Field(description="Temporal context of the change")


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


class GraphNode(BaseModel, IKNNEntity):
    """A node in the knowledge+experience graph"""

    id: str
    node_type: NodeType
    name: str
    description: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 1.0

    # For experience nodes - preserve full trigger context
    source_trigger_id: Optional[str] = None

    def get_id(self) -> str:
        return self.id

    def get_embedding(self) -> np.ndarray:
        if self.embedding is None:
            # Return empty array if embedding generation failed
            return np.array([])
        return np.array(self.embedding)

    def get_text(self) -> str:
        return f"{self.name} ({self.node_type.value}): {self.description}"


class GraphRelationship(BaseModel):
    """A relationship between nodes with confidence scoring and temporal bounds"""

    id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str  # Dynamic relationship types instead of enum
    confidence: float  # 0.0 to 1.0
    strength: float = 1.0  # Relationship strength
    created_at: datetime = Field(default_factory=datetime.now)
    last_reinforced: datetime = Field(default_factory=datetime.now)
    reinforcement_count: int = 1
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_trigger_id: Optional[str] = None

    # Temporal bounds for relationship validity
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_to: Optional[datetime] = None  # None means currently valid
    lifecycle_state: RelationshipLifecycleState = RelationshipLifecycleState.ACTIVE

    # Supersession tracking for temporal relationships
    superseded_by: Optional[str] = None  # ID of relationship that supersedes this one
    supersedes: Optional[str] = None  # ID of relationship this one supersedes


class KnowledgeExperienceGraph:
    """Knowledge+Experience Graph with confidence-weighted relationships"""

    def __init__(self):
        # Core storage
        self.nodes: Dict[str, GraphNode] = {}
        self.relationships: Dict[str, GraphRelationship] = {}

        # N-ary relationships storage (the main relationship system)
        from agent.experiments.knowledge_graph.n_ary_relationship import (
            NaryRelationship,
        )

        self.nary_relationships: Dict[str, NaryRelationship] = {}

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

    def add_nary_relationship(self, nary_relationship) -> None:
        """Add an n-ary relationship to the graph"""
        self.nary_relationships[nary_relationship.id] = nary_relationship
        logger.debug(
            f"Added n-ary relationship: {nary_relationship.relationship_type} "
            f"with {len(nary_relationship.participants)} participants"
        )

    def get_nary_relationships(self) -> List:
        """Get all n-ary relationships"""
        return list(self.nary_relationships.values())

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

    def detect_change_type(
        self,
        existing_relationships: List[GraphRelationship],
        new_relationship_info: str,
        context: str,
        llm: LLM,
        model: SupportedModel,
    ) -> ChangeDetectionResult:
        """
        Detect whether new information represents a correction or evolution.

        Args:
            existing_relationships: List of potentially conflicting relationships
            new_relationship_info: Description of the new relationship being added
            context: Context about when/how this new information was obtained
            llm: LLM instance for analysis
            model: Model to use for analysis

        Returns:
            ChangeDetectionResult indicating correction vs evolution
        """
        from agent.structured_llm import direct_structured_llm_call

        # Build summary of existing relationships
        existing_summary = []
        for rel in existing_relationships:
            source_node = self.nodes.get(rel.source_node_id)
            target_node = self.nodes.get(rel.target_node_id)
            if source_node and target_node:
                existing_summary.append(
                    f"- {source_node.name} --[{rel.relationship_type}]--> {target_node.name} "
                    f"(created: {rel.created_at.strftime('%Y-%m-%d')}, confidence: {rel.confidence})"
                )

        existing_text = (
            "\n".join(existing_summary)
            if existing_summary
            else "No existing relationships"
        )

        prompt = f"""I need to determine whether new information represents a CORRECTION of past mistakes or an EVOLUTION of knowledge over time.

EXISTING RELATIONSHIPS:
{existing_text}

NEW INFORMATION:
{new_relationship_info}

CONTEXT:
{context}

DEFINITIONS:
- CORRECTION: The previous information was factually wrong or misinterpreted. The agent made an error in understanding or inference. The old information should be marked as deprecated because it was never actually true.

- EVOLUTION: The previous information was correct at the time, but knowledge has genuinely changed. For example, preferences changing, situations evolving, or new developments occurring. The old information should be marked as historical/superseded.

EXAMPLES:
- CORRECTION: Agent thought "David likes anime" but David explicitly says "I never liked anime, I was just being polite"
- EVOLUTION: Agent knew "David likes anime" but now David says "I used to like anime but I've grown out of it"

Consider:
1. Does the new information contradict the old information?
2. Is there temporal language suggesting change over time ("used to", "now", "no longer")?
3. Is there language suggesting the old information was wrong ("actually", "never really", "misunderstood")?
4. What is the confidence level of the existing relationships?
5. How recent are the existing relationships vs the new information?

Analyze this situation and determine if this is a correction or evolution."""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=ChangeDetectionResult,
                model=model,
                llm=llm,
                caller="change_detection",
            )

            logger.info(
                f"Change detection: {result.change_type.value} - {result.reasoning}"
            )
            return result

        except Exception as e:
            logger.error(f"Change detection failed: {e}")
            # Fallback: assume evolution to be safe
            return ChangeDetectionResult(
                change_type=ChangeType.EVOLUTION,
                reasoning="LLM analysis failed, defaulting to evolution to preserve historical data",
                confidence=0.5,
                should_supersede=True,
                temporal_context="Unable to determine temporal context",
            )

    def supersede_relationships(
        self,
        relationships_to_supersede: List[GraphRelationship],
        new_relationship: GraphRelationship,
        change_type: ChangeType,
    ) -> None:
        """
        Mark existing relationships as superseded by a new relationship.

        Args:
            relationships_to_supersede: List of relationships to mark as superseded
            new_relationship: The new relationship that supersedes the old ones
            change_type: Whether this is a correction or evolution
        """

        for old_rel in relationships_to_supersede:
            # Set the appropriate lifecycle state based on change type
            if change_type == ChangeType.CORRECTION:
                old_rel.lifecycle_state = RelationshipLifecycleState.DEPRECATED
                logger.info(
                    f"Marked relationship {old_rel.id} as DEPRECATED (correction)"
                )
            else:  # EVOLUTION
                old_rel.lifecycle_state = RelationshipLifecycleState.SUPERSEDED
                logger.info(
                    f"Marked relationship {old_rel.id} as SUPERSEDED (evolution)"
                )

            # Set temporal bounds - old relationship is no longer valid
            old_rel.valid_to = new_relationship.created_at

            # Create supersession links
            old_rel.superseded_by = new_relationship.id
            new_relationship.supersedes = (
                old_rel.id
            )  # Note: this overwrites if multiple, could use a list

            # Update the relationship in storage
            self.relationships[old_rel.id] = old_rel

        # Ensure the new relationship is marked as active
        new_relationship.lifecycle_state = RelationshipLifecycleState.ACTIVE
        logger.info(
            f"Created new {new_relationship.lifecycle_state.value} relationship {new_relationship.id}"
        )

    def get_active_relationships(
        self, at_time: Optional[datetime] = None, include_historical: bool = False
    ) -> List[GraphRelationship]:
        """
        Get relationships that were active at a specific time.

        Args:
            at_time: Time to check for active relationships (default: now)
            include_historical: Whether to include historical/superseded relationships

        Returns:
            List of relationships that were active at the specified time
        """
        if at_time is None:
            at_time = datetime.now()

        active_rels = []
        for rel in self.relationships.values():
            # Skip deprecated relationships unless specifically requested
            if (
                not include_historical
                and rel.lifecycle_state == RelationshipLifecycleState.DEPRECATED
            ):
                continue

            # Check if relationship was valid at the requested time
            if rel.valid_from <= at_time:
                if rel.valid_to is None or rel.valid_to > at_time:
                    # Only include if we want all types or if it's currently active
                    if (
                        include_historical
                        or rel.lifecycle_state == RelationshipLifecycleState.ACTIVE
                    ):
                        active_rels.append(rel)

        return active_rels

    def get_relationship_history(
        self,
        source_node_id: str,
        target_node_id: str,
        relationship_type: Optional[str] = None,
    ) -> List[GraphRelationship]:
        """
        Get the full history of relationships between two nodes.

        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            relationship_type: Optional filter by relationship type

        Returns:
            List of relationships ordered by creation time (oldest first)
        """
        history = []

        for rel in self.relationships.values():
            if (
                rel.source_node_id == source_node_id
                and rel.target_node_id == target_node_id
                and (
                    relationship_type is None
                    or rel.relationship_type == relationship_type
                )
            ):
                history.append(rel)

        # Sort by creation time
        history.sort(key=lambda r: r.created_at)
        return history

    def export_to_dict(self) -> Dict[str, Any]:
        """Export entire graph to dictionary using Pydantic serialization"""
        return self.to_data().model_dump()

    def get_stats(self) -> GraphStats:
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
        if confidences:
            confidence_dist = ConfidenceDistribution(
                mean=sum(confidences) / len(confidences),
                high_confidence_count=sum(1 for c in confidences if c >= 0.8),
                medium_confidence_count=sum(1 for c in confidences if 0.5 <= c < 0.8),
                low_confidence_count=sum(1 for c in confidences if c < 0.5),
            )
        else:
            confidence_dist = ConfidenceDistribution()

        return GraphStats(
            total_nodes=len(self.nodes),
            total_relationships=len(self.relationships),
            processed_triggers=len(self.processed_triggers),
            node_types=node_counts,
            relationship_types=rel_counts,
            confidence_distribution=confidence_dist,
        )

    def to_data(self):
        """Convert to serializable Pydantic data model"""
        if _KnowledgeGraphData is None:
            raise RuntimeError("KnowledgeGraphData model not initialized")

        from agent.experiments.knowledge_graph.n_ary_relationship import (
            NaryRelationship,
        )

        return _KnowledgeGraphData(
            nodes=list(self.nodes.values()),
            relationships=list(self.relationships.values()),
            nary_relationships=list(self.nary_relationships.values()),
            processed_triggers=list(self.processed_triggers),
        )

    @classmethod
    def from_data(cls, data) -> "KnowledgeExperienceGraph":
        """Reconstruct KnowledgeExperienceGraph from Pydantic data model"""
        graph = cls()

        # Add nodes first (this rebuilds indexes)
        for node in data.nodes:
            graph.add_node(node)

        # Add relationships (this rebuilds relationship indexes)
        for rel in data.relationships:
            graph.add_relationship(rel)

        # Add n-ary relationships
        for nrel in data.nary_relationships:
            graph.add_nary_relationship(nrel)

        # Set processed triggers
        graph.processed_triggers = set(data.processed_triggers)

        return graph

    def save_to_file(self, filename: str) -> None:
        """Save graph to JSON file using Pydantic serialization"""
        data = self.to_data()

        with open(filename, "w") as f:
            f.write(data.model_dump_json(indent=2))

        logger.info(f"Saved graph to {filename}")

    @classmethod
    def load_from_file(cls, filename: str) -> "KnowledgeExperienceGraph":
        """Load graph from JSON file using Pydantic deserialization"""
        with open(filename, "r") as f:
            json_data = f.read()

        # Try to load using new Pydantic format if available
        if _KnowledgeGraphData is not None:
            try:
                data = _KnowledgeGraphData.model_validate_json(json_data)
                graph = cls.from_data(data)
                logger.info(f"Loaded graph from {filename}")
                return graph

            except Exception as e:
                # Fallback to manual loading for backward compatibility
                logger.warning(
                    f"Failed to load with Pydantic format, falling back to manual parsing: {e}"
                )
        else:
            logger.info("KnowledgeGraphData not available, using legacy loading")

        return cls._load_legacy_format(filename)

    @classmethod
    def _load_legacy_format(cls, filename: str) -> "KnowledgeExperienceGraph":
        """Load graph from legacy JSON format for backward compatibility"""
        with open(filename, "r") as f:
            data = json.load(f)

        graph = cls()

        # Load nodes - use Pydantic validation for individual objects
        for node_data in data["nodes"]:
            try:
                node = GraphNode.model_validate(node_data)
                graph.add_node(node)
            except Exception:
                # Fallback for very old node formats
                node = GraphNode(
                    id=node_data["id"],
                    node_type=NodeType(node_data["node_type"]),
                    name=node_data["name"],
                    description=node_data["description"],
                    importance=node_data["importance"],
                    created_at=(
                        datetime.fromisoformat(node_data["created_at"])
                        if isinstance(node_data["created_at"], str)
                        else node_data["created_at"]
                    ),
                    properties=node_data.get("properties", {}),
                )
                graph.add_node(node)

        # Load relationships - use Pydantic validation for individual objects
        for rel_data in data["relationships"]:
            try:
                rel = GraphRelationship.model_validate(rel_data)
                graph.add_relationship(rel)
            except Exception:
                # Fallback for old relationship formats with complex compatibility logic
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
                    valid_from=created_at,
                    valid_to=None,
                    lifecycle_state=RelationshipLifecycleState.ACTIVE,
                    superseded_by=rel_data.get("superseded_by"),
                    supersedes=rel_data.get("supersedes"),
                )
                graph.add_relationship(rel)

        # Load n-ary relationships
        if "nary_relationships" in data:
            from agent.experiments.knowledge_graph.n_ary_relationship import (
                NaryRelationship,
            )

            for nrel_data in data["nary_relationships"]:
                try:
                    nary_rel = NaryRelationship.model_validate(nrel_data)
                    graph.add_nary_relationship(nary_rel)
                except Exception:
                    # Fallback with defaults for missing fields
                    nary_rel = NaryRelationship(
                        id=nrel_data["id"],
                        relationship_type=nrel_data["relationship_type"],
                        participants=nrel_data["participants"],
                        confidence=nrel_data["confidence"],
                        strength=nrel_data.get("strength", nrel_data["confidence"]),
                        source_trigger_id=nrel_data.get("source_trigger_id", "unknown"),
                        created_at=(
                            datetime.fromisoformat(nrel_data["created_at"])
                            if isinstance(nrel_data["created_at"], str)
                            else nrel_data["created_at"]
                        ),
                        properties=nrel_data.get("properties", {}),
                        pattern=nrel_data.get("pattern"),
                    )
                    graph.add_nary_relationship(nary_rel)

        # Load processed triggers
        graph.processed_triggers = set(data.get("processed_triggers", []))

        logger.info(f"Loaded graph from {filename} (legacy format)")
        return graph


def _rebuild_pydantic_models():
    """Create KnowledgeGraphData model after all forward references are resolved"""
    global _KnowledgeGraphData

    try:
        # Import NaryRelationship here to avoid circular import
        from agent.experiments.knowledge_graph.n_ary_relationship import (
            NaryRelationship,
        )

        # Define KnowledgeGraphData now that all types are available
        class KnowledgeGraphData(BaseModel):
            """Pydantic surrogate model for KnowledgeExperienceGraph serialization"""

            nodes: List[GraphNode]
            relationships: List[GraphRelationship]
            nary_relationships: List[NaryRelationship]
            processed_triggers: List[str]

            class Config:
                arbitrary_types_allowed = True

        # Make it available globally
        globals()["_KnowledgeGraphData"] = KnowledgeGraphData

    except Exception as e:
        # If model creation fails, we'll fall back to legacy loading
        logger.warning(f"Failed to create KnowledgeGraphData model: {e}")


# Rebuild models when module is loaded
_rebuild_pydantic_models()


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
                f"    Graph stats: {stats.total_nodes} nodes, {stats.total_relationships} relationships"
            )

    # Final statistics
    print("\n📈 Final Graph Statistics:")
    stats = graph.get_stats()
    for key, value in stats.model_dump().items():
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
