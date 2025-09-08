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
    RELATIONSHIP_PATTERNS,
    identify_relationship_pattern,
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
    participants: List[ExtractedParticipant] = Field(
        description="All participants with their semantic roles"
    )
    confidence: float = Field(description="Confidence in this relationship (0-1)")
    evidence: str = Field(description="Text evidence supporting this relationship")
    pattern_type: Optional[str] = Field(
        description="Identified pattern like 'transfer', 'preference', 'instrumental_use'"
    )


class NaryRelationshipExtraction(BaseModel):
    """Collection of extracted N-ary relationships"""

    relationships: List[ExtractedNaryRelationship]
    extraction_reasoning: str = Field(
        description="Explanation of the extraction process"
    )


class NaryRelationshipExtractor:
    """Extracts N-ary relationships from text using LLM pattern recognition"""

    def __init__(self, llm: LLM, model: SupportedModel):
        self.llm = llm
        self.model = model

    def extract_nary_relationships(
        self,
        text: str,
        context: str = "",
        existing_entities: Optional[List[ExistingEntityData]] = None,
    ) -> List[ExtractedNaryRelationship]:
        """Extract N-ary relationships from text"""

        # Build pattern descriptions for the LLM
        pattern_descriptions = []
        for name, pattern in RELATIONSHIP_PATTERNS.items():
            roles_desc = ", ".join([role.value for role in pattern.required_roles])
            optional_roles_desc = (
                ", ".join([role.value for role in pattern.optional_roles])
                if pattern.optional_roles
                else "none"
            )

            pattern_descriptions.append(
                f"- {name.upper()}: {pattern.description}\n"
                f"  Required roles: {roles_desc}\n"
                f"  Optional roles: {optional_roles_desc}\n"
                f'  Example: "{pattern.examples[0]}"'
            )

        patterns_text = "\n".join(pattern_descriptions)

        # Build existing entities context
        entities_context = ""
        if existing_entities:
            entities_list = [
                f"- {e.name} ({e.type}): {e.description}"
                for e in existing_entities[:10]
            ]
            entities_context = f"\n\nEXISTING ENTITIES:\n" + "\n".join(entities_list)

        prompt = f"""I need to extract N-ary relationships from text. N-ary relationships involve multiple participants with specific semantic roles, going beyond simple binary relationships.

TEXT TO ANALYZE:
{text}

CONTEXT:
{context}{entities_context}

RELATIONSHIP PATTERNS TO IDENTIFY:
{patterns_text}

SEMANTIC ROLES AVAILABLE:
- agent: The one who performs the action (David in "David gave me penthouse")
- patient: The one affected by the action (me in "David gave me penthouse")
- beneficiary: The one who benefits (me in "David gave me penthouse")
- object: The direct object (penthouse in "David gave me penthouse")
- preferred: The preferred option (chocolate in "David prefers chocolate over vanilla")
- compared_to: The comparison baseline (vanilla in "David prefers chocolate over vanilla")
- instrument: The tool used (search tool in "I used search tool to find anime")
- purpose: The goal/purpose (find anime in "I used search tool to find anime")
- subject, target, source, destination, location, time: Generic roles

EXTRACTION GUIDELINES:
1. Look for relationships that involve 3 or more entities with distinct roles
2. Identify the relationship type (gave, prefers, used, causes, creates, etc.)
3. Use GENERIC relationship types - NEVER include specific names (e.g., use 'prefers' not 'David_prefers')
4. Assign semantic roles to each participant
5. Match against the predefined patterns when possible
6. Extract evidence text that supports each relationship
7. Focus on meaningful relationships that capture the full semantic structure
8. AVOID backwards causation - emotions/feelings cannot cause physical objects or people

EXAMPLES:
- "David gave me the penthouse" → gave(agent: David, beneficiary: me, object: penthouse)
- "David prefers chocolate over vanilla" → prefers(agent: David, preferred: chocolate, compared_to: vanilla)
- "I used the search tool to find anime" → used(agent: me, instrument: search tool, purpose: find anime)

Extract all N-ary relationships with their participants and semantic roles:"""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=NaryRelationshipExtraction,
                model=self.model,
                llm=self.llm,
                caller="nary_relationship_extraction",
                temperature=0.2,
            )

            # Post-process to identify patterns
            for rel in result.relationships:
                if not rel.pattern_type:
                    # Create participants dict for pattern matching
                    participants = {
                        p.semantic_role: p.entity_name for p in rel.participants
                    }
                    pattern = identify_relationship_pattern(
                        rel.relationship_type, participants
                    )
                    if pattern:
                        rel.pattern_type = pattern.pattern_name

            return result.relationships

        except Exception as e:
            print(f"❌ N-ary relationship extraction failed: {e}")
            return []

    def convert_to_nary_relationship(
        self,
        extracted: ExtractedNaryRelationship,
        entity_name_to_id: Dict[str, str],
        source_trigger_id: str,
    ) -> Optional[NaryRelationship]:
        """Convert extracted relationship to NaryRelationship with proper node IDs"""

        participants = {}

        # Map entity names to node IDs
        for participant in extracted.participants:
            # Try to find existing entity ID
            entity_key = self._find_entity_key(
                participant.entity_name, participant.entity_type, entity_name_to_id
            )
            if entity_key:
                participants[participant.semantic_role] = entity_name_to_id[entity_key]
            else:
                # Entity doesn't exist yet - would need to be created first
                print(
                    f"⚠️  Entity not found for N-ary relationship: {participant.entity_name}"
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
                "extraction_reasoning": "Extracted via N-ary pattern recognition",
            },
            source_trigger_id=source_trigger_id,
            pattern=extracted.pattern_type,
        )

    def _find_entity_key(
        self, entity_name: str, entity_type: str, entity_name_to_id: Dict[str, str]
    ) -> Optional[str]:
        """Find entity key in the name-to-ID mapping"""

        # Try exact match first
        exact_key = f"{entity_name.lower()}|{entity_type.lower()}"
        if exact_key in entity_name_to_id:
            return exact_key

        # Try partial matches
        for key in entity_name_to_id.keys():
            if entity_name.lower() in key.lower():
                return key

        return None

    def extract_comparative_relationships(
        self, text: str
    ) -> List[ExtractedNaryRelationship]:
        """Specialized extraction for comparative relationships like 'prefers X over Y'"""

        comparative_prompt = f"""Extract comparative relationships from this text. Look specifically for:

TEXT: {text}

COMPARATIVE PATTERNS:
- "prefers X over Y" → prefers(agent: someone, preferred: X, compared_to: Y)
- "likes X better than Y" → likes(agent: someone, preferred: X, compared_to: Y)  
- "chooses X instead of Y" → chooses(agent: someone, preferred: X, compared_to: Y)
- "X is better than Y" → comparative_evaluation(subject: X, object: Y)

Focus only on relationships that involve comparison between options."""

        try:
            result = direct_structured_llm_call(
                prompt=comparative_prompt,
                response_model=NaryRelationshipExtraction,
                model=self.model,
                llm=self.llm,
                caller="comparative_extraction",
                temperature=0.1,
            )
            return result.relationships
        except Exception as e:
            print(f"❌ Comparative extraction failed: {e}")
            return []

    def extract_transfer_relationships(
        self, text: str
    ) -> List[ExtractedNaryRelationship]:
        """Specialized extraction for transfer relationships like 'X gave Y to Z'"""

        transfer_prompt = f"""Extract transfer relationships from this text. Look specifically for:

TEXT: {text}

TRANSFER PATTERNS:
- "X gave Y to Z" → gave(agent: X, object: Y, beneficiary: Z)
- "X handed Y the Z" → handed(agent: X, beneficiary: Y, object: Z)
- "X provided Y with Z" → provided(agent: X, beneficiary: Y, object: Z)
- "X sent Y to Z" → sent(agent: X, object: Y, destination: Z)

Focus only on relationships involving transfer of objects, information, or benefits."""

        try:
            result = direct_structured_llm_call(
                prompt=transfer_prompt,
                response_model=NaryRelationshipExtraction,
                model=self.model,
                llm=self.llm,
                caller="transfer_extraction",
                temperature=0.1,
            )
            return result.relationships
        except Exception as e:
            print(f"❌ Transfer extraction failed: {e}")
            return []

    def extract_instrumental_relationships(
        self, text: str
    ) -> List[ExtractedNaryRelationship]:
        """Specialized extraction for instrumental relationships like 'X used Y to do Z'"""

        instrumental_prompt = f"""Extract instrumental relationships from this text. Look specifically for:

TEXT: {text}

INSTRUMENTAL PATTERNS:
- "X used Y to do Z" → used(agent: X, instrument: Y, purpose: Z)
- "X employed Y for Z" → employed(agent: X, instrument: Y, purpose: Z)
- "X utilized Y to achieve Z" → utilized(agent: X, instrument: Y, purpose: Z)
- "With Y, X accomplished Z" → accomplished(agent: X, instrument: Y, object: Z)

Focus only on relationships involving the use of tools, methods, or means to achieve goals."""

        try:
            result = direct_structured_llm_call(
                prompt=instrumental_prompt,
                response_model=NaryRelationshipExtraction,
                model=self.model,
                llm=self.llm,
                caller="instrumental_extraction",
                temperature=0.1,
            )
            return result.relationships
        except Exception as e:
            print(f"❌ Instrumental extraction failed: {e}")
            return []
