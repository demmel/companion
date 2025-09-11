#!/usr/bin/env python3
"""
N-ary Relationship Extraction

Extracts relationships with multiple participants and semantic roles from text,
replacing the binary-only extraction system.
"""

import uuid
import numpy as np
import logging
from typing import List, Optional
from agent.chain_of_action.prompts import format_section, format_single_trigger_entry
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from pydantic import BaseModel, Field
from dataclasses import dataclass

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.experiments.knowledge_graph.n_ary_relationship import (
    NaryRelationship,
)
from agent.experiments.knowledge_graph.relationship_schema_bank import (
    RelationshipSchemaBank,
)

logger = logging.getLogger(__name__)


@dataclass
class ExistingEntityData:
    id: str
    name: str
    type: str
    description: str


class ExtractedParticipant(BaseModel):
    """A participant in an N-ary relationship"""

    semantic_role: str = Field(
        description="Semantic role like 'agent', 'patient', 'object'"
    )
    entity_id: Optional[str] = Field(
        description="Optional existing entity ID if known", default=None
    )
    entity_name: str
    entity_type: str = Field(description="Type like 'person', 'object', 'concept'")
    description: str = Field(description="Brief description of the entity")


class ExtractedNaryRelationship(BaseModel):
    """An N-ary relationship extracted from text"""

    evidence: str = Field(description="Text evidence supporting this relationship")
    relationship_type: str = Field(
        description="Type of relationship like 'gave', 'prefers', 'used'"
    )
    category: str = Field(
        description="Semantic category: 'state', 'action', 'transfer', 'comparative', etc."
    )
    description: str = Field(
        description="Clear, specific description of what this relationship means (e.g., 'One entity transfers an object to another entity')"
    )
    participants: List[ExtractedParticipant] = Field(
        description="All participants with their semantic roles"
    )
    confidence: float = Field(description="Confidence in this relationship (0-1)")


class NaryRelationshipExtraction(BaseModel):
    """Collection of extracted N-ary relationships"""

    extraction_reasoning: str = Field(
        description="Explanation of the extraction process"
    )
    relationships: List[ExtractedNaryRelationship]


class NaryRelationshipExtractor:
    """Extracts N-ary relationships from text using LLM with relationship schema bank"""

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel,
        relationship_bank: RelationshipSchemaBank,
    ):
        self.llm = llm
        self.model = model
        self.relationship_bank = relationship_bank

    def extract_nary_relationships(
        self,
        trigger: TriggerHistoryEntry,
        existing_entities: List[ExistingEntityData],
    ) -> List[ExtractedNaryRelationship]:
        """Extract N-ary relationships from text using dynamic schema proposal"""

        trigger_text = format_single_trigger_entry(trigger, use_summary=False)

        used_context_chars = len(trigger_text)
        max_context_chars = 32000 - 4096  # Leave room for response

        # Generate trigger embedding for context filtering
        try:
            from agent.memory.embedding_service import get_embedding_service

            embedding_service = get_embedding_service()
            trigger_embedding = embedding_service.encode(trigger_text)
        except Exception as e:
            logger.warning(f"Failed to generate trigger embedding: {e}")
            trigger_embedding = None

        # Build filtered schemas context using KNN
        schema_examples = []
        if trigger_embedding is not None and self.relationship_bank.schema_search:
            # Find top-k most relevant schemas
            try:
                relevant_schemas = (
                    self.relationship_bank.schema_search.find_similar_entities(
                        np.array(trigger_embedding), k=5, similarity_threshold=0.3
                    )
                )

                if relevant_schemas:
                    schema_names = [
                        f"{s.t.name}({s.similarity:.2f})" for s in relevant_schemas
                    ]
                    logger.info(
                        f"KNN selected {len(relevant_schemas)} relevant schemas: {', '.join(schema_names)}"
                    )
                else:
                    logger.info(
                        "KNN found no relevant schemas above threshold, using fallback"
                    )

                for schema in relevant_schemas:
                    roles_text = ", ".join(schema.t.semantic_roles)
                    schema_examples.append(
                        f"- {schema.t.name} ({schema.t.category}): {roles_text} - {schema.t.description}"
                    )
            except Exception as e:
                logger.warning(f"Schema KNN filtering failed, using all schemas: {e}")
                # Fallback to all schemas (limited)
                current_schemas = list(
                    self.relationship_bank.relationship_schemas.items()
                )[:5]
                for name, entry in current_schemas:
                    roles_text = ", ".join(entry.semantic_roles)
                    schema_examples.append(
                        f"- {name} ({entry.category}): {roles_text} - {entry.description}"
                    )
        else:
            # Fallback to all schemas (limited)
            current_schemas = list(self.relationship_bank.relationship_schemas.items())[
                :5
            ]
            for name, entry in current_schemas:
                roles_text = ", ".join(entry.semantic_roles)
                schema_examples.append(
                    f"- {name} ({entry.category}): {roles_text} - {entry.description}"
                )

        schemas_context = "\n".join(schema_examples)
        used_context_chars += len(schemas_context)

        # Build filtered entities context using KNN (if we had entity search here)
        # For now, keep the existing approach but limit more aggressively
        entities_list = []
        remaining_chars = max_context_chars - used_context_chars

        for e in existing_entities[:20]:  # Limit to top 20 most relevant
            entity_entry = f"- {e.id}: {e.name} ({e.type}): {e.description}"
            if len(entity_entry) + 1 > remaining_chars:
                break

            entities_list.append(entity_entry)
            remaining_chars -= len(entity_entry) + 1
        entities_context = "\n".join(entities_list)

        prompt = f"""Extract N-ary relationships from text. For each relationship, provide the relationship type, category, description, and semantic roles.

{format_section("EXISTING ENTITIES", entities_context)}

{format_section("EXISTING RELATIONSHIP SCHEMAS", schemas_context)}

{format_section("TEXT TO ANALYZE", trigger_text)}

RELATIONSHIP EXTRACTION GUIDELINES:

1. **Identify Relationships**: Look for relationships involving 2 or more entities with distinct roles
2. **Propose Relationship Type**: Choose a clear, generic verb or relationship name (gave, prefers, used, etc.)
3. **Determine Category**: Classify each relationship:
   - **state**: Mental states, preferences, beliefs (prefers, likes, knows, believes)
   - **action**: Physical or deliberate actions (gave, created, destroyed, chose)
   - **transfer**: Moving objects/information between entities (sent, handed, provided)
   - **comparative**: Comparing options (prefers X over Y, ranks X above Y)
   - **causal**: One thing causing another (caused, led_to, resulted_in)
   - **instrumental**: Using tools/means for purposes (used X to do Y)

4. **Write Description**: Provide a clear, specific description of what this relationship means (NOT generic like "relationship involving X, Y")
   - Good: "One entity transfers an object to another entity as a gift or payment"
   - Bad: "Relationship involving agent, beneficiary, object"

5. **Assign Semantic Roles**: Create role names that clearly describe each participant's function:
   - Common roles: agent, patient, object, beneficiary, instrument, purpose, preferred, compared_to
   - Feel free to create specific roles that better capture the relationship meaning

6. **Reference Existing Entities**: For each participant, provide:
   - entity_id: Use the exact ID if it matches an entity from EXISTING ENTITIES above
   - entity_name: Always provide the entity name
   - entity_type: Always provide the entity type
   - description: Always provide entity description
   
7. **Extract Evidence**: Include the text that supports this relationship

PARTICIPANT REFERENCE RULES:
- If the participant matches an entity from EXISTING ENTITIES, use that entity's exact ID
- If it's a new entity, leave entity_id as null and provide full name/type/description
- Always provide name, type, and description even when using existing entity_id

EXAMPLES:
- "David gave me the penthouse" with EXISTING ENTITIES containing "person_david_123: David (person)" → 
  - Type: gave, Category: transfer
  - Description: "One entity transfers an object to another entity as a gift or payment"
  - Participants with roles: agent=David (use ID person_david_123), beneficiary=me (new entity), object=penthouse (new entity)

Extract all meaningful relationships with their specific descriptions and role structures:"""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=NaryRelationshipExtraction,
                model=self.model,
                llm=self.llm,
                caller="nary_relationship_extraction",
                temperature=0.2,
            )

            # Post-process with relationship bank integration
            processed_relationships = []
            for rel in result.relationships:
                # Use relationship bank to get or create schema
                schema_roles = [p.semantic_role for p in rel.participants]

                # Generate realistic example from the extracted relationship
                examples = [
                    f"{rel.relationship_type}({', '.join([f'{p.semantic_role}={p.entity_name}' for p in rel.participants])}"
                ]

                schema_match = self.relationship_bank.get_or_create_relationship_schema(
                    proposed_type=rel.relationship_type,
                    proposed_roles=schema_roles,
                    description=rel.description,  # Use LLM-generated description
                    category=rel.category,
                    examples=examples,
                    context=f"Extracted from: {rel.evidence}",
                )

                # Update the relationship with the final schema
                rel.relationship_type = schema_match.relationship_type
                if schema_match.role_mapping:
                    # Map participant roles if needed
                    for participant in rel.participants:
                        if participant.semantic_role in schema_match.role_mapping:
                            participant.semantic_role = schema_match.role_mapping[
                                participant.semantic_role
                            ]

                processed_relationships.append(rel)

            return processed_relationships

        except Exception as e:
            logger.error(f"N-ary relationship extraction failed: {e}")
            return []

    def convert_to_nary_relationship(
        self,
        extracted: ExtractedNaryRelationship,
        source_trigger_id: str,
        entity_resolver,
    ) -> Optional[NaryRelationship]:
        """Convert extracted relationship to NaryRelationship with proper node IDs"""

        participants = {}

        # Map entity names to node IDs using enhanced entity resolution
        failed_entities = []
        for participant in extracted.participants:
            entity_node_id = None

            # Strategy 1: Use provided entity_id if available and verified
            if participant.entity_id:
                if entity_resolver.verify_entity_match(
                    participant.entity_id,
                    participant.entity_name,
                    participant.entity_type,
                    participant.description,
                ):
                    entity_node_id = participant.entity_id
                    logger.info(
                        f"✅ Using verified entity ID: {participant.entity_name} -> {entity_node_id}"
                    )
                else:
                    logger.warning(
                        f"❌ Entity ID verification failed for {participant.entity_id}, falling back to resolution"
                    )

            # Strategy 2: Fall back to KNN resolution if no entity_id or verification failed
            if not entity_node_id:
                entity_node_id = entity_resolver.resolve_entity_to_node_id(
                    participant.entity_name,
                    participant.entity_type,
                    participant.description,
                )
                if entity_node_id:
                    logger.info(
                        f"🔍 Resolved via KNN: {participant.entity_name} -> {entity_node_id}"
                    )

            # Strategy 3: Create new entity if resolution failed
            if not entity_node_id:
                entity_node_id = entity_resolver.create_entity_from_nary_participant(
                    participant.entity_name,
                    participant.entity_type,
                    participant.description,
                    source_trigger_id,
                )
                if entity_node_id:
                    logger.info(
                        f"📝 Created new entity: {participant.entity_name} -> {entity_node_id}"
                    )

            # Final check
            if entity_node_id:
                participants[participant.semantic_role] = entity_node_id
            else:
                failed_entities.append(
                    f"{participant.entity_name} ({participant.entity_type})"
                )
                logger.warning(
                    f"❌ All strategies failed for entity: {participant.entity_name} ({participant.entity_type})"
                )

        if len(participants) < 2:
            logger.warning(
                f"Not enough valid participants for N-ary relationship '{extracted.relationship_type}': "
                f"found {len(participants)}, need at least 2. Failed entities: {failed_entities}"
            )
            return None  # Need at least 2 participants

        if failed_entities:
            logger.info(
                f"Creating N-ary relationship '{extracted.relationship_type}' with {len(participants)} participants. "
                f"Skipped unresolved entities: {failed_entities}"
            )

        return NaryRelationship(
            id=str(uuid.uuid4()),
            relationship_type=extracted.relationship_type,
            confidence=extracted.confidence,
            strength=extracted.confidence,  # Use confidence as initial strength
            participants=participants,
            properties={
                "evidence": extracted.evidence,
                "category": extracted.category,
                "extraction_reasoning": "Extracted via N-ary schema system",
            },
            source_trigger_id=source_trigger_id,
            category=extracted.category,  # Use category instead of old pattern_type
        )
