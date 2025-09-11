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

from agent.experiments.knowledge_graph.knn_entity_search import (
    IKNNEntity,
    KNNEntitySearch,
)
from agent.memory.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class RelationshipLifecycleState(str, Enum):
    """Lifecycle states for temporal relationship management"""

    ACTIVE = "active"  # Currently valid relationship
    HISTORICAL = "historical"  # Was valid in the past, superseded by new information
    DEPRECATED = "deprecated"  # Was incorrect, should not have existed
    SUPERSEDED = "superseded"  # Was correct but replaced by evolved knowledge


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
    evidence: List[str] = Field(
        default_factory=list,
        description="List of evidence supporting this relationship",
    )
    source_trigger_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    category: Optional[str] = None  # e.g. "transaction", "preference", "usage"

    # Lifecycle management (migrated from binary relationships)
    lifecycle_state: RelationshipLifecycleState = Field(
        default=RelationshipLifecycleState.ACTIVE
    )
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_to: Optional[datetime] = None
    superseded_by: Optional[str] = None  # ID of relationship that supersedes this one
    supersedes: Optional[str] = None  # ID of relationship this one supersedes
    last_reinforced: datetime = Field(default_factory=datetime.now)
    reinforcement_count: int = Field(default=0)

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
        if self.evidence:
            base_text += f" | Evidence: {' | '.join(self.evidence)}"

        # Add pattern if available
        if self.category:
            base_text += f" | Pattern: {self.category}"

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

        # Add new evidence to evidence list
        if new_evidence and new_evidence not in self.evidence:
            self.evidence.append(new_evidence)

        # Update lifecycle tracking
        self.last_reinforced = datetime.now()
        self.reinforcement_count += 1

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

    def get_role_for_participant(self, node_id: str) -> Optional[str]:
        """Get the semantic role of a specific participant"""
        for role, participant_id in self.participants.items():
            if participant_id == node_id:
                return role
        return None

    def has_participant(self, node_id: str) -> bool:
        """Check if a node is a participant in this relationship"""
        return node_id in self.participants.values()

    def is_valid_at_time(self, at_time: datetime) -> bool:
        """Check if this relationship is valid at the given time"""
        if self.lifecycle_state in [
            RelationshipLifecycleState.DEPRECATED,
            RelationshipLifecycleState.SUPERSEDED,
        ]:
            return False

        if at_time < self.valid_from:
            return False

        if self.valid_to is not None and at_time > self.valid_to:
            return False

        return True

    def supersede_with(
        self, new_relationship_id: str, supersession_time: Optional[datetime] = None
    ) -> None:
        """Mark this relationship as superseded by another relationship"""
        self.lifecycle_state = RelationshipLifecycleState.SUPERSEDED
        self.superseded_by = new_relationship_id
        if supersession_time:
            self.valid_to = supersession_time
        else:
            self.valid_to = datetime.now()
