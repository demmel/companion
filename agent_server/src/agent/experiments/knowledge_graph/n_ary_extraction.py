#!/usr/bin/env python3
"""
N-ary Relationship Extraction

Extracts relationships with multiple participants and semantic roles from text,
replacing the binary-only extraction system.
"""

import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass
class ExistingEntityData:
    name: str
    type: str
    description: str


from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.experiments.knowledge_graph.n_ary_relationship import (
    NaryRelationship,
)
from agent.experiments.knowledge_graph.relationship_schema_bank import (
    RelationshipSchemaBank,
)


class ExtractedParticipant(BaseModel):
    """A participant in an N-ary relationship"""

    entity_name: str
    semantic_role: str = Field(
        description="Semantic role like 'agent', 'patient', 'object'"
    )
    entity_type: str = Field(description="Type like 'person', 'object', 'concept'")
    description: str = Field(description="Brief description of the entity")


class ExtractedNaryRelationship(BaseModel):
    """An N-ary relationship extracted from text"""

    relationship_type: str = Field(
        description="Type of relationship like 'gave', 'prefers', 'used'"
    )
    category: str = Field(
        description="Semantic category: 'state', 'action', 'transfer', 'comparative', etc."
    )
    participants: List[ExtractedParticipant] = Field(
        description="All participants with their semantic roles"
    )
    confidence: float = Field(description="Confidence in this relationship (0-1)")
    evidence: str = Field(description="Text evidence supporting this relationship")


class NaryRelationshipExtraction(BaseModel):
    """Collection of extracted N-ary relationships"""

    relationships: List[ExtractedNaryRelationship]
    extraction_reasoning: str = Field(
        description="Explanation of the extraction process"
    )


class NaryRelationshipExtractor:
    """Extracts N-ary relationships from text using LLM with relationship schema bank"""

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel,
        relationship_bank: Optional[RelationshipSchemaBank] = None,
    ):
        self.llm = llm
        self.model = model
        self.relationship_bank = relationship_bank

    def extract_nary_relationships(
        self,
        text: str,
        context: str = "",
        existing_entities: Optional[List[ExistingEntityData]] = None,
    ) -> List[ExtractedNaryRelationship]:
        """Extract N-ary relationships from text using dynamic schema proposal"""

        # Build existing entities context
        entities_context = ""
        if existing_entities:
            entities_list = [
                f"- {e.name} ({e.type}): {e.description}"
                for e in existing_entities[:10]
            ]
            entities_context = f"\n\nEXISTING ENTITIES:\n" + "\n".join(entities_list)

        # Build existing schemas context if we have a relationship bank
        schemas_context = ""
        if self.relationship_bank and self.relationship_bank.relationship_schemas:
            schema_examples = []
            for name, entry in list(
                self.relationship_bank.relationship_schemas.items()
            )[:5]:
                roles_text = ", ".join(entry.semantic_roles)
                schema_examples.append(
                    f"- {name} ({entry.category}): {roles_text} - {entry.description}"
                )
            schemas_context = f"\n\nEXISTING RELATIONSHIP SCHEMAS:\n" + "\n".join(
                schema_examples
            )

        prompt = f"""Extract N-ary relationships from text. For each relationship, propose both the relationship type AND a semantic category.

TEXT TO ANALYZE:
{text}

CONTEXT:
{context}{entities_context}{schemas_context}

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

4. **Assign Semantic Roles**: Create role names that clearly describe each participant's function:
   - Common roles: agent, patient, object, beneficiary, instrument, purpose, preferred, compared_to
   - Feel free to create specific roles that better capture the relationship meaning

5. **Extract Evidence**: Include the text that supports this relationship

EXAMPLES:
- "David gave me the penthouse" → gave (action): agent=David, beneficiary=me, object=penthouse
- "David prefers chocolate over vanilla" → prefers (state): agent=David, preferred=chocolate, compared_to=vanilla  
- "I used the search tool to find anime" → used (instrumental): agent=me, instrument=search_tool, purpose=find_anime

Extract all meaningful relationships with their categories and role structures:"""

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
                if self.relationship_bank:
                    # Use relationship bank to get or create schema
                    schema_roles = [p.semantic_role for p in rel.participants]
                    examples = [
                        f"{rel.relationship_type}({', '.join([f'{p.semantic_role}={p.entity_name}' for p in rel.participants])}"
                    ]

                    schema_match = self.relationship_bank.get_or_create_relationship_schema(
                        proposed_type=rel.relationship_type,
                        proposed_roles=schema_roles,
                        description=f"Relationship involving {', '.join(schema_roles)}",
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
            print(f"❌ N-ary relationship extraction failed: {e}")
            return []

    def convert_to_nary_relationship(
        self,
        extracted: ExtractedNaryRelationship,
        source_trigger_id: str,
        entity_resolver,
    ) -> Optional[NaryRelationship]:
        """Convert extracted relationship to NaryRelationship with proper node IDs"""

        participants = {}

        # Map entity names to node IDs using entity resolver
        for participant in extracted.participants:
            # Use the entity resolver to find the node ID for this entity
            entity_node_id = entity_resolver.resolve_entity_to_node_id(
                participant.entity_name,
                participant.entity_type,
                participant.description,
            )

            if entity_node_id:
                participants[participant.semantic_role] = entity_node_id
            else:
                # Entity doesn't exist yet - would need to be created first
                print(
                    f"⚠️  Entity not found for N-ary relationship: {participant.entity_name} ({participant.entity_type})"
                )
                return None

        if len(participants) < 2:
            return None  # Need at least 2 participants

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
            pattern=extracted.category,  # Use category instead of old pattern_type
        )
