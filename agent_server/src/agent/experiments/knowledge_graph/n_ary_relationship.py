#!/usr/bin/env python3
"""
N-ary Relationship System

Implements relationships with multiple participants and semantic roles,
solving the limitation of binary relationships for complex interactions.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np
import logging

from agent.experiments.knowledge_graph.knowledge_graph_prototype import (
    GraphRelationship,
)
from agent.experiments.knowledge_graph.knn_entity_search import (
    IKNNEntity,
    KNNEntitySearch,
)
from agent.memory.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class SemanticRole(str, Enum):
    """Standard semantic roles for N-ary relationships"""

    # Core participant roles
    AGENT = "agent"  # The one who performs the action
    PATIENT = "patient"  # The one affected by the action
    BENEFICIARY = "beneficiary"  # The one who benefits from the action
    OBJECT = "object"  # The direct object of the action

    # Comparison roles
    PREFERRED = "preferred"  # The preferred option in a comparison
    COMPARED_TO = "compared_to"  # The comparison baseline

    # Instrumental roles
    INSTRUMENT = "instrument"  # The tool used to perform action
    PURPOSE = "purpose"  # The goal/purpose of the action

    # Spatial/temporal roles
    SOURCE = "source"  # Starting point
    DESTINATION = "destination"  # Ending point
    LOCATION = "location"  # Where something happens
    TIME = "time"  # When something happens

    # Generic roles for flexibility
    SUBJECT = "subject"  # Generic subject role
    TARGET = "target"  # Generic target role


class RelationshipPattern(BaseModel):
    """Template for common N-ary relationship patterns"""

    pattern_name: str
    relationship_type: str
    required_roles: List[SemanticRole]
    optional_roles: List[SemanticRole]

    description: str
    examples: List[str]

    def matches_participants(self, participants: Dict[str, str]) -> bool:
        """Check if participants match this pattern"""
        participant_roles = set(participants.keys())
        required_roles = set(role.value for role in self.required_roles)

        return required_roles.issubset(participant_roles)


class NaryRelationship(BaseModel, IKNNEntity):
    """Represents a relationship with multiple participants and semantic roles"""

    id: str
    relationship_type: str  # "gave", "prefers", "used_for"
    confidence: float
    strength: float

    # Participants with semantic roles
    participants: Dict[str, str] = Field(description="role -> node_id mapping")
    # Example: {"agent": "person_david", "beneficiary": "person_me", "object": "object_penthouse"}

    properties: Dict[str, Any] = Field(default_factory=dict)
    source_trigger_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    pattern: Optional[str] = None  # Which pattern this relationship follows

    # Optional embedding for KNN search
    _embedding: Optional[np.ndarray] = None
    _embedding_text: Optional[str] = None

    def get_id(self) -> str:
        """Return unique identifier for KNN search"""
        return self.id

    def get_embedding(self) -> np.ndarray:
        """Return embedding for KNN search"""
        if self._embedding is None:
            # Generate embedding on demand
            text = self.get_text()
            try:
                embedding_service = get_embedding_service()
                embedding_list = embedding_service.encode(text)
                if embedding_list:
                    self._embedding = np.array(embedding_list)
                    self._embedding_text = text
                else:
                    self._embedding = np.array([])
            except Exception:
                self._embedding = np.array([])

        return self._embedding if self._embedding is not None else np.array([])

    def get_text(self) -> str:
        """Return text representation for KNN search"""
        # Create a canonical text representation of this relationship
        # Format: relationship_type(role1=participant1, role2=participant2, ...)
        participants_text = ", ".join(
            [
                f"{role}={participant_id}"
                for role, participant_id in sorted(self.participants.items())
            ]
        )

        base_text = f"{self.relationship_type}({participants_text})"

        # Add evidence if available
        if "evidence" in self.properties:
            base_text += f" | Evidence: {self.properties['evidence']}"

        # Add pattern if available
        if self.pattern:
            base_text += f" | Pattern: {self.pattern}"

        return base_text

    def strengthen_with_evidence(
        self, new_evidence: str, new_confidence: float
    ) -> None:
        """Strengthen this relationship with new evidence instead of creating duplicate"""

        # Update confidence using weighted average
        # Give more weight to higher confidence evidence
        current_weight = self.confidence
        new_weight = new_confidence
        total_weight = current_weight + new_weight

        if total_weight > 0:
            self.confidence = (
                self.confidence * current_weight + new_confidence * new_weight
            ) / total_weight
            # Cap at 1.0
            self.confidence = min(self.confidence, 1.0)

        # Update strength similarly
        self.strength = (
            (self.strength * current_weight + new_confidence * new_weight)
            / total_weight
            if total_weight > 0
            else self.strength
        )
        self.strength = min(self.strength, 1.0)

        # Add new evidence to properties
        existing_evidence = self.properties.get("evidence", "")
        if existing_evidence:
            # Combine evidence, avoid exact duplicates
            if new_evidence not in existing_evidence:
                self.properties["evidence"] = existing_evidence + " | " + new_evidence
        else:
            self.properties["evidence"] = new_evidence

        # Update last seen timestamp
        self.properties["last_strengthened"] = datetime.now().isoformat()

        # Reset embedding cache since evidence changed
        self._embedding = None
        self._embedding_text = None

    def is_similar_relationship(
        self, other: "NaryRelationship", similarity_threshold: float = 0.85
    ) -> bool:
        """Check if this relationship is similar to another (for deduplication)"""

        # Must have same relationship type
        if self.relationship_type != other.relationship_type:
            return False

        # Must have same participants (regardless of role mapping differences)
        self_participants = set(self.participants.values())
        other_participants = set(other.participants.values())

        if self_participants != other_participants:
            return False

        # Check role mapping similarity (allow for minor role name differences)
        if set(self.participants.keys()) == set(other.participants.keys()):
            # Exact role match
            return True

        # Could add more sophisticated role similarity checking here
        # For now, require exact participant and role structure
        return False

    def get_primary_participants(self) -> Tuple[str, str]:
        """Get the two most important participants for binary compatibility"""

        # Priority order for determining primary relationship
        role_priority = [
            SemanticRole.AGENT.value,
            SemanticRole.SUBJECT.value,
            SemanticRole.SOURCE.value,
            SemanticRole.PATIENT.value,
            SemanticRole.OBJECT.value,
            SemanticRole.TARGET.value,
            SemanticRole.BENEFICIARY.value,
            SemanticRole.DESTINATION.value,
        ]

        sorted_roles = []
        for role in role_priority:
            if role in self.participants:
                sorted_roles.append(role)

        if len(sorted_roles) >= 2:
            return (
                self.participants[sorted_roles[0]],
                self.participants[sorted_roles[1]],
            )
        elif len(sorted_roles) == 1:
            # Single participant relationship
            participant_id = self.participants[sorted_roles[0]]
            return participant_id, participant_id

        # Fallback: use first two participants
        participant_ids = list(self.participants.values())
        return participant_ids[0], (
            participant_ids[1] if len(participant_ids) > 1 else participant_ids[0]
        )

    def to_binary_relationship(self) -> GraphRelationship:
        """Convert to binary relationship for backward compatibility"""

        source_id, target_id = self.get_primary_participants()

        return GraphRelationship(
            id=f"binary_{self.id}",
            source_node_id=source_id,
            target_node_id=target_id,
            relationship_type=self.relationship_type,
            confidence=self.confidence,
            strength=self.strength,
            properties={
                **self.properties,
                "nary_relationship_id": self.id,
                "semantic_roles": self.participants,
                "relationship_pattern": self.pattern,
            },
            source_trigger_id=self.source_trigger_id,
        )

    def get_role_for_participant(self, node_id: str) -> Optional[str]:
        """Get the semantic role of a specific participant"""
        for role, participant_id in self.participants.items():
            if participant_id == node_id:
                return role
        return None

    def has_participant(self, node_id: str) -> bool:
        """Check if a node is a participant in this relationship"""
        return node_id in self.participants.values()


# Predefined relationship patterns
RELATIONSHIP_PATTERNS = {
    "transfer": RelationshipPattern(
        pattern_name="transfer",
        relationship_type="gave",
        required_roles=[SemanticRole.AGENT, SemanticRole.OBJECT],
        optional_roles=[
            SemanticRole.BENEFICIARY,
            SemanticRole.LOCATION,
            SemanticRole.TIME,
        ],
        description="One entity transfers something to another entity",
        examples=[
            "David gave me penthouse",
            "I gave David a gift",
            "Someone handed the book to me",
        ],
    ),
    "preference": RelationshipPattern(
        pattern_name="preference",
        relationship_type="prefers",
        required_roles=[SemanticRole.AGENT, SemanticRole.PREFERRED],
        optional_roles=[SemanticRole.COMPARED_TO],
        description="One entity has a preference for something over something else",
        examples=[
            "David prefers chocolate over vanilla",
            "I prefer anime to movies",
            "She likes tea better than coffee",
        ],
    ),
    "instrumental_use": RelationshipPattern(
        pattern_name="instrumental_use",
        relationship_type="used",
        required_roles=[SemanticRole.AGENT, SemanticRole.INSTRUMENT],
        optional_roles=[
            SemanticRole.PURPOSE,
            SemanticRole.OBJECT,
            SemanticRole.LOCATION,
        ],
        description="One entity uses a tool/instrument for a purpose",
        examples=[
            "I used search tool to find anime",
            "David used laptop for gaming",
            "She used knife to cut bread",
        ],
    ),
    "causation": RelationshipPattern(
        pattern_name="causation",
        relationship_type="causes",
        required_roles=[SemanticRole.AGENT, SemanticRole.PATIENT],
        optional_roles=[
            SemanticRole.INSTRUMENT,
            SemanticRole.LOCATION,
            SemanticRole.TIME,
        ],
        description="One entity causes a change in another entity",
        examples=[
            "David causes my excitement",
            "Rain causes wet ground",
            "Exercise causes health improvement",
        ],
    ),
    "creation": RelationshipPattern(
        pattern_name="creation",
        relationship_type="creates",
        required_roles=[SemanticRole.AGENT, SemanticRole.OBJECT],
        optional_roles=[
            SemanticRole.INSTRUMENT,
            SemanticRole.PURPOSE,
            SemanticRole.LOCATION,
        ],
        description="One entity creates or produces another entity",
        examples=[
            "Artist creates painting",
            "Chef creates meal",
            "Writer creates story",
        ],
    ),
}


def identify_relationship_pattern(
    relationship_type: str, participants: Dict[str, str]
) -> Optional[RelationshipPattern]:
    """Identify which pattern best matches the relationship and participants"""

    # First, try exact relationship type match
    for pattern in RELATIONSHIP_PATTERNS.values():
        if (
            pattern.relationship_type == relationship_type
            and pattern.matches_participants(participants)
        ):
            return pattern

    # Then, try participant role matching regardless of relationship type
    participant_roles = set(participants.keys())

    best_match = None
    best_score = 0

    for pattern in RELATIONSHIP_PATTERNS.values():
        required_roles = set(role.value for role in pattern.required_roles)
        optional_roles = set(role.value for role in pattern.optional_roles)
        all_pattern_roles = required_roles | optional_roles

        # Score based on role overlap
        matching_roles = participant_roles & all_pattern_roles
        missing_required = required_roles - participant_roles

        if len(missing_required) == 0:  # All required roles present
            score = len(matching_roles) / len(all_pattern_roles)
            if score > best_score:
                best_score = score
                best_match = pattern

    return best_match if best_score > 0.5 else None


class NaryRelationshipManager:
    """Manages N-ary relationships and their conversion to binary relationships"""

    def __init__(self):
        self.nary_relationships: Dict[str, NaryRelationship] = {}
        self.patterns = RELATIONSHIP_PATTERNS

        # KNN search for relationship deduplication
        self.relationship_search = KNNEntitySearch[NaryRelationship]()

    def add_nary_relationship(
        self, nary_rel: NaryRelationship, new_evidence: str, new_confidence: float
    ) -> str:
        """Add an N-ary relationship to the manager with deduplication"""

        # Identify pattern if not already set
        if not nary_rel.pattern:
            pattern = identify_relationship_pattern(
                nary_rel.relationship_type, nary_rel.participants
            )
            if pattern:
                nary_rel.pattern = pattern.pattern_name

        # Check for similar existing relationships
        similar_rel = self.find_similar_relationship(nary_rel)

        if similar_rel:
            # Strengthen existing relationship instead of creating duplicate
            evidence = new_evidence or nary_rel.properties.get("evidence", "")
            confidence = new_confidence or nary_rel.confidence

            similar_rel.strengthen_with_evidence(evidence, confidence)
            logger.info(
                f"Strengthened existing relationship: {similar_rel.id} (similarity found)"
            )
            return similar_rel.id
        else:
            # Add new relationship
            self.nary_relationships[nary_rel.id] = nary_rel
            self.relationship_search.add_entity(nary_rel)
            logger.info(f"Added new relationship: {nary_rel.id}")
            return nary_rel.id

    def find_similar_relationship(
        self, nary_rel: NaryRelationship
    ) -> Optional[NaryRelationship]:
        """Find similar relationship using KNN search"""

        if not self.nary_relationships:
            return None

        try:
            # Use KNN search to find similar relationships
            best_match = self.relationship_search.find_best_match(nary_rel.get_text())

            if best_match and best_match.similarity >= 0.85:
                # Additional verification with our similarity method
                if nary_rel.is_similar_relationship(best_match.t):
                    return best_match.t

        except Exception as e:
            logger.warning(f"KNN relationship search failed: {e}")

        return None

    def get_nary_relationship(self, relationship_id: str) -> Optional[NaryRelationship]:
        """Get an N-ary relationship by ID"""
        return self.nary_relationships.get(relationship_id)

    def find_relationships_with_participant(
        self, node_id: str
    ) -> List[NaryRelationship]:
        """Find all N-ary relationships involving a specific participant"""
        return [
            rel
            for rel in self.nary_relationships.values()
            if rel.has_participant(node_id)
        ]

    def find_relationships_by_role(self, role: str) -> List[NaryRelationship]:
        """Find all N-ary relationships where someone has a specific semantic role"""
        return [
            rel for rel in self.nary_relationships.values() if role in rel.participants
        ]

    def find_relationships_by_pattern(
        self, pattern_name: str
    ) -> List[NaryRelationship]:
        """Find all N-ary relationships following a specific pattern"""
        return [
            rel
            for rel in self.nary_relationships.values()
            if rel.pattern == pattern_name
        ]

    def get_all_binary_relationships(self) -> List[GraphRelationship]:
        """Convert all N-ary relationships to binary relationships"""
        return [
            rel.to_binary_relationship() for rel in self.nary_relationships.values()
        ]
