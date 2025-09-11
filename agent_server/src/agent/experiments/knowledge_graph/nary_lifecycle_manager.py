#!/usr/bin/env python3
"""
N-ary Relationship Lifecycle Management

Handles change detection, supersession, and temporal management for n-ary relationships.
"""

import logging
from typing import List, Optional
from enum import Enum

from agent.experiments.knowledge_graph.knowledge_graph import KnowledgeExperienceGraph
from agent.experiments.knowledge_graph.n_ary_relationship import (
    NaryRelationship,
    RelationshipLifecycleState,
)
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RelationshipResolution(str, Enum):
    """How to resolve relationships - covers duplicates, contradictions, and unrelated cases"""

    DUPLICATE = "duplicate"  # Same info, strengthen existing
    CLEAR_MISTAKE = "clear_mistake"  # Old was wrong, deprecate it
    TEMPORAL_CHANGE = "temporal_change"  # Changed over time, supersede old
    INSUFFICIENT_INFO = "insufficient_info"  # Conflicting sources, mark both
    UNRELATED = "unrelated"  # Different info, add as new


class RelationshipValidationResult(BaseModel):
    """Result of LLM validation for relationship resolution"""

    reasoning: str = Field(
        description="Explanation of the decision - analyze first before choosing resolution"
    )
    resolution: RelationshipResolution = Field(
        description="How to resolve this relationship"
    )
    confidence: float = Field(description="Confidence in this assessment (0.0-1.0)")


class NaryLifecycleManager:
    """Manages lifecycle of n-ary relationships including change detection and supersession"""

    def __init__(self, llm: LLM, model: SupportedModel):
        self.llm = llm
        self.model = model

    def add_relationship_with_lifecycle_management(
        self,
        graph: KnowledgeExperienceGraph,
        new_relationship: NaryRelationship,
        trigger: TriggerHistoryEntry,
    ) -> bool:
        """Add a relationship with proper lifecycle management using 2-step flow"""

        # STEP 1: Look up exact structural match for strengthen
        exact_match = self._find_exact_match_relationship(graph, new_relationship)
        if exact_match:
            # Strengthen existing relationship instead of creating duplicate
            evidence = (
                " | ".join(new_relationship.evidence)
                if new_relationship.evidence
                else ""
            )
            exact_match.strengthen_with_evidence(evidence, new_relationship.confidence)
            logger.info(
                f"Strengthened existing n-ary relationship: {exact_match.relationship_type}"
            )
            return True

        # STEP 2: LLM validation for high similarity relationships (duplicates and contradictions)
        high_similarity_rels = self._find_high_similarity_relationships(
            graph, new_relationship
        )

        if high_similarity_rels:
            validation_result = self._llm_validate_relationships(
                graph, high_similarity_rels, new_relationship, trigger
            )

            if validation_result.resolution == RelationshipResolution.DUPLICATE:
                # Strengthen existing relationship
                best_match = high_similarity_rels[
                    0
                ]  # Could be refined to pick best match
                evidence = (
                    " | ".join(new_relationship.evidence)
                    if new_relationship.evidence
                    else ""
                )
                best_match.strengthen_with_evidence(
                    evidence, new_relationship.confidence
                )
                logger.info(
                    f"LLM detected duplicate, strengthened: {best_match.relationship_type}"
                )
                return True

            elif validation_result.resolution in [
                RelationshipResolution.CLEAR_MISTAKE,
                RelationshipResolution.TEMPORAL_CHANGE,
                RelationshipResolution.INSUFFICIENT_INFO,
            ]:
                # Handle contradiction based on resolution type
                self._handle_contradiction(
                    graph,
                    high_similarity_rels,
                    new_relationship,
                    validation_result.resolution,
                )
                logger.info(
                    f"Handled contradiction with resolution: {validation_result.resolution}"
                )

            # If UNRELATED, fall through to add as new relationship

        # No duplicates or conflicts, add the new relationship
        graph.add_nary_relationship(new_relationship)
        logger.info(
            f"Added new n-ary relationship: {new_relationship.relationship_type}"
        )
        return True

    def _find_high_similarity_relationships(
        self, graph: KnowledgeExperienceGraph, new_relationship: NaryRelationship
    ) -> List[NaryRelationship]:
        """Find existing relationships with high similarity (> 0.7) for LLM validation"""

        from agent.memory.embedding_service import get_embedding_service

        embedding_service = get_embedding_service()

        new_rel_text = new_relationship.get_text()
        high_similarity_relationships = []

        for existing_rel in graph.get_nary_relationships():
            # Skip if not active
            if existing_rel.lifecycle_state != RelationshipLifecycleState.ACTIVE:
                continue

            # Calculate similarity
            try:
                similarity = embedding_service.cosine_similarity(
                    embedding_service.encode(new_rel_text),
                    embedding_service.encode(existing_rel.get_text()),
                )

                # High similarity indicates potential duplicate or contradiction
                if similarity > 0.7:
                    high_similarity_relationships.append(existing_rel)
                    logger.info(
                        f"Found high similarity relationship (similarity: {similarity:.4f}): {existing_rel.relationship_type}"
                    )

            except Exception as e:
                logger.warning(f"Failed to calculate similarity: {e}")
                continue

        return high_similarity_relationships

    def _llm_validate_relationships(
        self,
        graph: KnowledgeExperienceGraph,
        similar_relationships: List[NaryRelationship],
        new_relationship: NaryRelationship,
        trigger: TriggerHistoryEntry,
    ) -> RelationshipValidationResult:
        """Use LLM to validate if similar relationships are duplicates or contradictions"""

        # Extract trigger text for LLM context
        from agent.chain_of_action.prompts import format_single_trigger_entry

        trigger_text = format_single_trigger_entry(trigger, use_summary=False)

        # Build descriptions of similar relationships
        existing_descriptions = []
        for existing_rel in similar_relationships:
            participant_parts = []
            for role, participant_id in existing_rel.participants.items():
                node = graph.nodes.get(participant_id)
                node_name = node.name if node else participant_id
                participant_parts.append(f"{role}={node_name}")
            existing_desc = (
                f"{existing_rel.relationship_type}({', '.join(participant_parts)})"
            )
            existing_descriptions.append(
                f"- {existing_desc} (confidence: {existing_rel.confidence}, created: {existing_rel.created_at.strftime('%Y-%m-%d')})"
            )

        existing_text = "\n".join(existing_descriptions)

        # Build description of new relationship
        new_participant_parts = []
        for role, participant_id in new_relationship.participants.items():
            node = graph.nodes.get(participant_id)
            node_name = node.name if node else participant_id
            new_participant_parts.append(f"{role}={node_name}")
        new_desc = (
            f"{new_relationship.relationship_type}({', '.join(new_participant_parts)})"
        )

        prompt = f"""Analyze the relationship between existing and new information to determine the appropriate resolution.

EXISTING SIMILAR RELATIONSHIPS:
{existing_text}

NEW RELATIONSHIP:
{new_desc}
- Confidence: {new_relationship.confidence}
- Evidence: {' | '.join(new_relationship.evidence) if new_relationship.evidence else 'None'}

CONTEXT FROM TRIGGER:
{trigger_text}

RESOLUTION TYPES:
1. DUPLICATE - New expresses same core fact as existing (strengthen existing)
2. CLEAR_MISTAKE - Old information was factually wrong (deprecate old, add new)
3. TEMPORAL_CHANGE - Information legitimately changed over time (supersede old, add new)
4. INSUFFICIENT_INFO - Conflicting sources, unclear which is right (mark both as conflicted)
5. UNRELATED - Different information that doesn't conflict (add as new)

EXAMPLES:
- DUPLICATE: "likes(agent=David, object=anime)" vs "enjoys(agent=David, target=anime)" (same meaning)
- CLEAR_MISTAKE: Had "gave(agent=David, object=house, beneficiary=me)" but David says "I never gave you anything"
- TEMPORAL_CHANGE: Had "prefers(agent=David, preferred=anime)" but now "I used to prefer anime, now I prefer movies"
- INSUFFICIENT_INFO: Alice says "David gave me his book" but David says "I lent it to her" (conflicting accounts of same event)
- UNRELATED: "likes(agent=David, object=anime)" vs "likes(agent=David, object=movies)" (different objects)

ANALYSIS CRITERIA:
- Look for explicit contradiction language ("actually", "never really", "that's wrong")
- Look for temporal language ("used to", "now", "changed", "no longer")
- Consider source reliability and context
- Default to UNRELATED if no clear relationship

First provide your reasoning, then choose the resolution type."""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=RelationshipValidationResult,
                model=self.model,
                llm=self.llm,
                caller="relationship_validation",
                temperature=0.2,
            )

            logger.info(
                f"LLM relationship validation: {result.resolution} - {result.reasoning}"
            )
            return result

        except Exception as e:
            logger.error(f"LLM relationship validation failed: {e}")
            # Fallback: assume it's unrelated
            return RelationshipValidationResult(
                reasoning="LLM analysis failed, defaulting to add as new relationship",
                resolution=RelationshipResolution.UNRELATED,
                confidence=0.5,
            )

    def _find_exact_match_relationship(
        self, graph: KnowledgeExperienceGraph, new_relationship: NaryRelationship
    ) -> Optional[NaryRelationship]:
        """STEP 1: Find existing relationship that is structurally identical"""

        for existing_rel in graph.get_nary_relationships():
            # Skip if not active
            if existing_rel.lifecycle_state != RelationshipLifecycleState.ACTIVE:
                continue

            # Check for exact structural match
            if (
                existing_rel.relationship_type == new_relationship.relationship_type
                and existing_rel.participants == new_relationship.participants
            ):
                logger.info(
                    f"Found exact structural match: {existing_rel.relationship_type}"
                )
                return existing_rel

        return None

    def _handle_contradiction(
        self,
        graph: KnowledgeExperienceGraph,
        existing_relationships: List[NaryRelationship],
        new_relationship: NaryRelationship,
        resolution: RelationshipResolution,
    ) -> None:
        """Handle contradictions based on the resolution type"""

        if resolution == RelationshipResolution.CLEAR_MISTAKE:
            # Old information was wrong - deprecate it
            for old_rel in existing_relationships:
                old_rel.lifecycle_state = RelationshipLifecycleState.DEPRECATED
                old_rel.valid_to = new_relationship.created_at
                old_rel.superseded_by = new_relationship.id
                graph.nary_relationships[old_rel.id] = old_rel
                logger.info(f"Deprecated relationship {old_rel.id} as clear mistake")

        elif resolution == RelationshipResolution.TEMPORAL_CHANGE:
            # Information changed over time - supersede old with new
            for old_rel in existing_relationships:
                old_rel.lifecycle_state = RelationshipLifecycleState.SUPERSEDED
                old_rel.valid_to = new_relationship.created_at
                old_rel.superseded_by = new_relationship.id
                new_relationship.supersedes = old_rel.id  # Note: overwrites if multiple
                graph.nary_relationships[old_rel.id] = old_rel
                logger.info(
                    f"Superseded relationship {old_rel.id} due to temporal change"
                )

        elif resolution == RelationshipResolution.INSUFFICIENT_INFO:
            # Keep both relationships but mark the contradiction
            for old_rel in existing_relationships:
                # Add contradiction marker to both relationships
                if "contradictions" not in old_rel.properties:
                    old_rel.properties["contradictions"] = []
                old_rel.properties["contradictions"].append(new_relationship.id)
                graph.nary_relationships[old_rel.id] = old_rel
                logger.info(
                    f"Marked relationship {old_rel.id} as conflicted with {new_relationship.id}"
                )

            # Mark new relationship as conflicted too
            if "contradictions" not in new_relationship.properties:
                new_relationship.properties["contradictions"] = []
            for old_rel in existing_relationships:
                new_relationship.properties["contradictions"].append(old_rel.id)
            logger.info(f"New relationship {new_relationship.id} marked as conflicted")

        # Always add the new relationship (except for DUPLICATE which is handled separately)
        if resolution != RelationshipResolution.DUPLICATE:
            new_relationship.lifecycle_state = RelationshipLifecycleState.ACTIVE
            graph.add_nary_relationship(new_relationship)
