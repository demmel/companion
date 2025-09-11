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
from agent.experiments.knowledge_graph.n_ary_relationship import NaryRelationship
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


class KnowledgeExperienceGraph:
    """Knowledge+Experience Graph with confidence-weighted relationships"""

    def __init__(self):
        # Core storage
        self.nodes: Dict[str, GraphNode] = {}

        self.nary_relationships: Dict[str, NaryRelationship] = {}

        # Indexes for efficient lookup
        self.nodes_by_type: Dict[NodeType, Set[str]] = {nt: set() for nt in NodeType}

        # Processing history
        self.processed_triggers: Set[str] = set()

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.nodes_by_type[node.node_type].add(node.id)
        logger.debug(f"Added node: {node.node_type.value} - {node.name}")

    def add_nary_relationship(self, nary_relationship: NaryRelationship) -> None:
        """Add an n-ary relationship to the graph"""
        self.nary_relationships[nary_relationship.id] = nary_relationship
        logger.debug(
            f"Added n-ary relationship: {nary_relationship.relationship_type} "
            f"with {len(nary_relationship.participants)} participants"
        )

    def get_nary_relationships(self) -> List[NaryRelationship]:
        """Get all n-ary relationships"""
        return list(self.nary_relationships.values())

    def get_nary_relationship(self, relationship_id: str) -> Optional[NaryRelationship]:
        """Get an n-ary relationship by ID"""
        return self.nary_relationships.get(relationship_id)

    def detect_nary_change_type(
        self,
        existing_relationships: List,  # List of NaryRelationship
        new_relationship_info: str,
        context: str,
        llm: LLM,
        model: SupportedModel,
    ) -> ChangeDetectionResult:
        """
        Detect whether new n-ary relationship information represents a correction or evolution.

        Args:
            existing_relationships: List of potentially conflicting n-ary relationships
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
            # Build participant description
            participant_parts = []
            for role, participant_id in rel.participants.items():
                node = self.nodes.get(participant_id)
                node_name = node.name if node else participant_id
                participant_parts.append(f"{role}={node_name}")
            participant_desc = ", ".join(participant_parts)
            existing_summary.append(
                f"- {rel.relationship_type}({participant_desc}) "
                f"(created: {rel.created_at.strftime('%Y-%m-%d')}, confidence: {rel.confidence})"
            )

        existing_text = (
            "\n".join(existing_summary)
            if existing_summary
            else "No existing relationships"
        )

        prompt = f"""I need to determine whether new information represents a CORRECTION of past mistakes or an EVOLUTION of knowledge over time.

EXISTING N-ARY RELATIONSHIPS:
{existing_text}

NEW INFORMATION:
{new_relationship_info}

CONTEXT:
{context}

DEFINITIONS:
- CORRECTION: The previous information was factually wrong or misinterpreted. The agent made an error in understanding or inference. The old information should be marked as deprecated because it was never actually true.

- EVOLUTION: The previous information was correct at the time, but knowledge has genuinely changed. For example, preferences changing, situations evolving, or new developments occurring. The old information should be marked as historical/superseded.

EXAMPLES:
- CORRECTION: Agent thought "gave(agent=David, object=penthouse, beneficiary=me)" but David explicitly says "I never gave you anything, you misunderstood"
- EVOLUTION: Agent knew "prefers(agent=David, preferred=anime, compared_to=movies)" but now David says "I used to prefer anime but now I prefer movies"

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
                caller="nary_change_detection",
            )

            logger.info(
                f"N-ary change detection: {result.change_type.value} - {result.reasoning}"
            )
            return result

        except Exception as e:
            logger.error(f"N-ary change detection failed: {e}")
            # Fallback: assume evolution to be safe
            return ChangeDetectionResult(
                change_type=ChangeType.EVOLUTION,
                reasoning="LLM analysis failed, defaulting to evolution to preserve historical data",
                confidence=0.5,
                should_supersede=True,
                temporal_context="Unable to determine temporal context",
            )

    def supersede_nary_relationships(
        self,
        relationships_to_supersede: List,  # List of NaryRelationship
        new_relationship,  # NaryRelationship
        change_type: ChangeType,
    ) -> None:
        """
        Mark existing n-ary relationships as superseded by a new relationship.

        Args:
            relationships_to_supersede: List of n-ary relationships to mark as superseded
            new_relationship: The new n-ary relationship that supersedes the old ones
            change_type: Whether this is a correction or evolution
        """

        for old_rel in relationships_to_supersede:
            # Set the appropriate lifecycle state based on change type
            if change_type == ChangeType.CORRECTION:
                old_rel.lifecycle_state = RelationshipLifecycleState.DEPRECATED
                logger.info(
                    f"Marked n-ary relationship {old_rel.id} as DEPRECATED (correction)"
                )
            else:  # EVOLUTION
                old_rel.lifecycle_state = RelationshipLifecycleState.SUPERSEDED
                logger.info(
                    f"Marked n-ary relationship {old_rel.id} as SUPERSEDED (evolution)"
                )

            # Set temporal bounds - old relationship is no longer valid
            old_rel.valid_to = new_relationship.created_at

            # Create supersession links
            old_rel.superseded_by = new_relationship.id
            new_relationship.supersedes = (
                old_rel.id
            )  # Note: this overwrites if multiple, could use a list

            # Update the relationship in storage
            self.nary_relationships[old_rel.id] = old_rel

        # Ensure the new relationship is marked as active
        new_relationship.lifecycle_state = RelationshipLifecycleState.ACTIVE
        logger.info(
            f"Created new {new_relationship.lifecycle_state.value} n-ary relationship {new_relationship.id}"
        )

    def get_active_nary_relationships(
        self, at_time: Optional[datetime] = None, include_historical: bool = False
    ) -> List:  # List of NaryRelationship
        """
        Get n-ary relationships that were active at a specific time.

        Args:
            at_time: Time to check for active relationships (default: now)
            include_historical: Whether to include historical/superseded relationships

        Returns:
            List of n-ary relationships that were active at the specified time
        """
        if at_time is None:
            at_time = datetime.now()

        active_rels = []
        for rel in self.nary_relationships.values():
            # Use the relationship's own validation method
            if rel.is_valid_at_time(at_time):
                if (
                    include_historical
                    or rel.lifecycle_state == RelationshipLifecycleState.ACTIVE
                ):
                    active_rels.append(rel)

        return active_rels

    def get_nary_relationship_history(
        self,
        participants: Dict[str, str],
        relationship_type: Optional[str] = None,
    ) -> List:  # List of NaryRelationship
        """
        Get the full history of n-ary relationships with the same participants.

        Args:
            participants: Dict of role -> node_id to match
            relationship_type: Optional filter by relationship type

        Returns:
            List of n-ary relationships ordered by creation time (oldest first)
        """
        history = []

        for rel in self.nary_relationships.values():
            # Check if participants match (same roles and same participants)
            if rel.participants == participants:
                if (
                    relationship_type is None
                    or rel.relationship_type == relationship_type
                ):
                    history.append(rel)

        # Sort by creation time
        history.sort(key=lambda r: r.created_at)
        return history

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
        semantic_roles: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ):
        """Get neighboring nodes through n-ary relationships"""
        neighbors = []

        for nary_rel in self.nary_relationships.values():
            # Check if this node participates in the relationship
            if not nary_rel.has_participant(node_id):
                continue

            # Filter by relationship type if specified
            if (
                relationship_types
                and nary_rel.relationship_type not in relationship_types
            ):
                continue

            # Filter by confidence
            if nary_rel.confidence < min_confidence:
                continue

            # Filter by semantic role if specified
            node_role = nary_rel.get_role_for_participant(node_id)
            if semantic_roles and node_role not in semantic_roles:
                continue

            # Add all other participants as neighbors
            for role, participant_id in nary_rel.participants.items():
                if participant_id != node_id:  # Don't include self
                    neighbor_node = self.nodes.get(participant_id)
                    if neighbor_node:
                        neighbors.append((neighbor_node, nary_rel, role))

        return neighbors

    def get_all_nodes(self) -> List[GraphNode]:
        """Get all nodes in the graph"""
        return list(self.nodes.values())

    def get_node_by_name(self, name: str) -> Optional[GraphNode]:
        """Get first node with matching name (case-insensitive)"""
        name_lower = name.lower()
        for node in self.nodes.values():
            if node.name.lower() == name_lower:
                return node
        return None

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

        # N-ary relationship counts (only n-ary relationships now)
        rel_counts = {}
        for rel in self.nary_relationships.values():
            rel_counts[rel.relationship_type] = (
                rel_counts.get(rel.relationship_type, 0) + 1
            )

        # Confidence distribution from n-ary relationships
        confidences = [rel.confidence for rel in self.nary_relationships.values()]

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
            total_relationships=len(self.nary_relationships),
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
            relationships=[],  # Empty list since we removed binary relationships
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

        # Skip binary relationships since they're deprecated
        # Only add n-ary relationships
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

        # Skip legacy binary relationships - they are no longer supported
        if "relationships" in data:
            logger.info(
                f"Skipping {len(data.get('relationships', []))} legacy binary relationships"
            )

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


class KnowledgeGraphData(BaseModel):
    """Pydantic surrogate model for KnowledgeExperienceGraph serialization"""

    nodes: List[GraphNode]
    nary_relationships: List[NaryRelationship]
    processed_triggers: List[str]

    class Config:
        arbitrary_types_allowed = True
