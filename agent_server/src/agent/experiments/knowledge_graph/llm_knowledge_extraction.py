#!/usr/bin/env python3
"""
LLM-Based Knowledge Extraction

Implements actual LLM calls to extract meaningful knowledge, concepts, relationships,
and emotional context from trigger history entries. This replaces the regex-based
placeholder extraction with real semantic understanding.
"""

import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass
import logging

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.chain_of_action.trigger import UserInputTrigger
from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.prompts import format_single_trigger_entry, format_section
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.state import State, build_agent_state_description

logger = logging.getLogger(__name__)


@dataclass
class EntityValidation:
    entity: str
    type: str
    confidence: float
    found_in_text: bool
    evidence_provided: bool


@dataclass
class RelationshipValidation:
    relationship: str
    confidence: float
    source_exists: bool
    target_exists: bool
    evidence_provided: bool


@dataclass
class ExtractionValidation:
    trigger_id: str
    entities_count: int
    relationships_count: int
    has_emotional_context: bool
    themes_count: int
    factual_claims_count: int
    commitments_count: int
    questions_count: int
    entity_validation: List[EntityValidation]
    relationship_validation: List[RelationshipValidation]


class ExtractedEntity(BaseModel):
    """An entity extracted from trigger content"""

    name: str = Field(description="Name or identifier of the entity")
    type: str = Field(
        description="Type of entity: person, concept, emotion, goal, object, etc."
    )
    description: str = Field(
        description="Brief description of what this entity represents"
    )
    confidence: float = Field(description="Confidence in this extraction (0.0 to 1.0)")
    evidence: str = Field(description="Text evidence that supports this extraction")


class ExtractedRelationship(BaseModel):
    """A relationship between entities extracted from trigger content"""

    source_entity: str = Field(description="Name of the source entity")
    target_entity: str = Field(description="Name of the target entity")
    relationship_type: str = Field(
        description="Type of relationship: caused, enabled, involves, similar_to, etc."
    )
    description: str = Field(
        description="Description of how these entities are related"
    )
    confidence: float = Field(
        description="Confidence in this relationship (0.0 to 1.0)"
    )
    evidence: str = Field(description="Text evidence that supports this relationship")


class ExtractedEmotionalContext(BaseModel):
    """Emotional context extracted from trigger content"""

    agent_emotions: List[str] = Field(
        default_factory=list, description="Emotions the agent expressed or felt"
    )
    user_emotions: List[str] = Field(
        default_factory=list, description="Emotions the user seemed to express"
    )
    emotional_triggers: List[str] = Field(
        default_factory=list, description="Things that triggered emotional responses"
    )
    confidence: float = Field(
        description="Confidence in emotional analysis (0.0 to 1.0)"
    )


class KnowledgeExtraction(BaseModel):
    """Complete knowledge extraction from a trigger history entry"""

    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)
    emotional_context: Optional[ExtractedEmotionalContext] = None
    key_themes: List[str] = Field(
        default_factory=list, description="Main themes or topics discussed"
    )
    factual_claims: List[str] = Field(
        default_factory=list,
        description="Factual statements that could be important later",
    )
    commitments_or_promises: List[str] = Field(
        default_factory=list, description="Any commitments or promises made"
    )
    questions_or_unknowns: List[str] = Field(
        default_factory=list,
        description="Questions raised or things that remain unclear",
    )


def build_knowledge_extraction_prompt(
    trigger: TriggerHistoryEntry, state: State, recent_nodes: Optional[List] = None
) -> str:
    """Build prompt for LLM to extract knowledge from a trigger using proven formatting"""

    # Use proven trigger formatting from chain_of_action/prompts.py
    trigger_text = format_single_trigger_entry(trigger, use_summary=False)
    state_desc = build_agent_state_description(state)
    
    sections = []
    
    # Add state context
    sections.append(format_section("CURRENT CONTEXT", state_desc))
    
    # Add the trigger experience using proven formatting
    sections.append(format_section("MY RECENT EXPERIENCE", trigger_text))
    
    # Add recent knowledge context if available
    if recent_nodes:
        recent_knowledge_text = "\n".join([
            f"- {node.get('name', 'Unknown')}: {node.get('description', '')}"
            for node in recent_nodes[:10]
        ])
        sections.append(format_section("RECENT KNOWLEDGE I'VE BEEN BUILDING", recent_knowledge_text))

    prompt = f"""I am {state.name}, {state.role}. I need to extract meaningful knowledge from my recent experience to build my personal memory and understanding. This is me reflecting on what I learned, felt, promised, or experienced during this interaction.

{"\n".join(sections)}

From my first-person perspective, extract knowledge using "I/me/my" language ONLY:

1. ENTITIES - People, concepts, topics, emotions, goals, objects from MY experience
   - Describe entities from MY viewpoint using first-person language
   - Example: "David" = "The person I'm talking to" NOT "The person Chloe is talking to"
   - Example: "warmth" = "The emotion I'm feeling" NOT "The emotion Chloe is feeling"
   - ENTITY TYPES: Choose the most specific type:
     * "person" - People I interact with
     * "emotion" - My emotional states  
     * "concept" - Abstract ideas, topics, skills, activities (anime, gaming, trust)
     * "object" - Physical things, places, tools, clothing, items (penthouse, dress, boots, web search tool)
     * "goal" - Intentions or objectives I have
   - Use MY exact words/phrases from the conversation
   - High confidence (0.8-1.0) for things I directly experienced
   - Medium confidence (0.5-0.7) for things I clearly implied

2. RELATIONSHIPS - How things connect in MY understanding (within this exchange AND to my recent knowledge)
   - Describe relationships using natural language from MY perspective
   - Look for connections to things I mentioned recently (see RECENT KNOWLEDGE above)
   - Use descriptive relationship types like: "reminds_me_of", "makes_me_feel", "is_similar_to", "David_prefers", "I_promised_to", "conflicts_with", "builds_on", etc.
   - Example: "anime" -> "reminds_me_of" -> "Dan Da Dan" (if I mentioned both recently)
   - Example: "trust choker" -> "symbolizes_my_bond_with" -> "David"
   - Include both new connections within this exchange AND connections to recent knowledge

3. EMOTIONAL CONTEXT - MY emotional experience during this exchange
   - What emotions did I express or feel?
   - What emotions did the user show that I should remember?
   - Describe using "I felt..." not "Chloe felt..."

4. KEY THEMES - Topics I was discussing or thinking about

5. FACTUAL CLAIMS - Facts I stated, learned, or should remember (from MY perspective)

6. COMMITMENTS/PROMISES - What I committed to do ("I will...", "I promise...", etc.)

7. QUESTIONS/UNKNOWNS - Things I wondered about or need to follow up on

CRITICAL REQUIREMENT: Use ONLY first-person language (I/me/my/mine). Never use third-person references to {state.name} or describe yourself from outside. This is MY personal memory being created from MY lived experience.
"""

    return prompt


class LLMKnowledgeExtractor:
    """Extracts knowledge from triggers using LLM calls"""

    def __init__(self, llm: LLM, model: SupportedModel):
        self.llm = llm
        self.model = model

    def extract_knowledge(
        self,
        trigger: TriggerHistoryEntry,
        state: State,
        recent_nodes: Optional[List] = None,
    ) -> Optional[KnowledgeExtraction]:
        """Extract knowledge from a single trigger using LLM"""

        prompt = build_knowledge_extraction_prompt(trigger, state, recent_nodes)

        try:
            extraction = direct_structured_llm_call(
                prompt=prompt,
                response_model=KnowledgeExtraction,
                model=self.model,
                llm=self.llm,
                caller="knowledge_extraction",
            )

            logger.info(
                f"Extracted {len(extraction.entities)} entities, {len(extraction.relationships)} relationships from trigger {trigger.entry_id}"
            )
            return extraction

        except Exception as e:
            logger.error(
                f"Knowledge extraction failed for trigger {trigger.entry_id}: {e}"
            )
            return None

    def validate_extraction(
        self, extraction: KnowledgeExtraction, trigger: TriggerHistoryEntry
    ) -> ExtractionValidation:
        """Validate the quality of an extraction"""

        # Get original text for validation
        user_input = (
            trigger.trigger.content
            if isinstance(trigger.trigger, UserInputTrigger)
            else ""
        )
        action_texts = []
        if trigger.actions_taken:
            for action in trigger.actions_taken:
                if action.type == ActionType.THINK:
                    focus = action.input.focus
                    if focus:
                        action_texts.append(focus)
                elif action.type == ActionType.SPEAK:
                    # Use actual spoken text from results for validation
                    if action.result and action.result.type == "success":
                        response = action.result.content.response
                        if response:
                            action_texts.append(response)

        all_text = " ".join([user_input] + action_texts).lower()

        entity_validations = []
        relationship_validations = []

        # Validate entities - check if entity names appear in original text
        for entity in extraction.entities:
            entity_in_text = entity.name.lower() in all_text
            entity_validations.append(
                EntityValidation(
                    entity=entity.name,
                    type=entity.type,
                    confidence=entity.confidence,
                    found_in_text=entity_in_text,
                    evidence_provided=len(entity.evidence) > 0,
                )
            )

        # Validate relationships - check if both entities exist and make sense
        entity_names = {e.name.lower() for e in extraction.entities}
        for rel in extraction.relationships:
            source_exists = (
                rel.source_entity.lower() in entity_names
                or rel.source_entity.lower() in all_text
            )
            target_exists = (
                rel.target_entity.lower() in entity_names
                or rel.target_entity.lower() in all_text
            )

            relationship_validations.append(
                RelationshipValidation(
                    relationship=f"{rel.source_entity} -> {rel.relationship_type} -> {rel.target_entity}",
                    confidence=rel.confidence,
                    source_exists=source_exists,
                    target_exists=target_exists,
                    evidence_provided=len(rel.evidence) > 0,
                )
            )

        return ExtractionValidation(
            trigger_id=trigger.entry_id,
            entities_count=len(extraction.entities),
            relationships_count=len(extraction.relationships),
            has_emotional_context=extraction.emotional_context is not None,
            themes_count=len(extraction.key_themes),
            factual_claims_count=len(extraction.factual_claims),
            commitments_count=len(extraction.commitments_or_promises),
            questions_count=len(extraction.questions_or_unknowns),
            entity_validation=entity_validations,
            relationship_validation=relationship_validations,
        )


def test_knowledge_extraction():
    """Test the knowledge extraction on sample triggers"""

    logging.basicConfig(level=logging.INFO)

    from agent.conversation_persistence import ConversationPersistence
    from agent.llm import create_llm, SupportedModel

    # Load baseline conversation data
    persistence = ConversationPersistence()
    trigger_history, state, initial_exchange = persistence.load_conversation("baseline")

    if state is None:
        print("❌ Could not load baseline state")
        return

    print(f"✅ Loaded baseline: {len(trigger_history.get_all_entries())} triggers")

    # Create LLM
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    # Test extraction on first 2 triggers for quick demo
    extractor = LLMKnowledgeExtractor(llm, model)

    all_triggers = trigger_history.get_all_entries()
    test_triggers = all_triggers[:2]  # Test just first 2 for speed

    print(f"\n🧪 Testing knowledge extraction on {len(test_triggers)} triggers...")

    extractions = []
    validations = []

    for i, trigger in enumerate(test_triggers):
        print(f"\nProcessing trigger {i+1}/{len(test_triggers)}: {trigger.entry_id}")

        extraction = extractor.extract_knowledge(trigger, state, [])
        if extraction:
            validation = extractor.validate_extraction(extraction, trigger)

            extractions.append(extraction)
            validations.append(validation)

            print(
                f"  ✅ Extracted: {validation.entities_count} entities, {validation.relationships_count} relationships"
            )
            print(
                f"     Themes: {validation.themes_count}, Facts: {validation.factual_claims_count}"
            )

            # Show some extracted entities
            if extraction.entities:
                print(
                    f"     Entities: {', '.join([f'{e.name}({e.type})' for e in extraction.entities[:3]])}..."
                )

            # Show validation issues
            invalid_entities = [
                e for e in validation.entity_validation if not e.found_in_text
            ]
            if invalid_entities:
                print(f"     ⚠️  Entities not found in text: {len(invalid_entities)}")

            invalid_rels = [
                r
                for r in validation.relationship_validation
                if not (r.source_exists and r.target_exists)
            ]
            if invalid_rels:
                print(f"     ⚠️  Invalid relationships: {len(invalid_rels)}")
        else:
            print(f"  ❌ Extraction failed")

    # Summary statistics
    print(f"\n📊 Extraction Summary:")
    print(f"  Successful extractions: {len(extractions)}/{len(test_triggers)}")

    if extractions:
        total_entities = sum(len(e.entities) for e in extractions)
        total_relationships = sum(len(e.relationships) for e in extractions)
        total_themes = sum(len(e.key_themes) for e in extractions)
        total_facts = sum(len(e.factual_claims) for e in extractions)
        total_commitments = sum(len(e.commitments_or_promises) for e in extractions)

        print(f"  Average entities per trigger: {total_entities/len(extractions):.1f}")
        print(
            f"  Average relationships per trigger: {total_relationships/len(extractions):.1f}"
        )
        print(f"  Average themes per trigger: {total_themes/len(extractions):.1f}")
        print(f"  Average facts per trigger: {total_facts/len(extractions):.1f}")
        print(
            f"  Average commitments per trigger: {total_commitments/len(extractions):.1f}"
        )

        # Validation statistics
        if validations:
            all_entity_validations = [
                e for v in validations for e in v["entity_validation"]
            ]
            valid_entities = sum(
                1 for e in all_entity_validations if e["found_in_text"]
            )
            print(
                f"  Entity validation rate: {valid_entities}/{len(all_entity_validations)} ({valid_entities/len(all_entity_validations)*100:.1f}%)"
            )

            all_rel_validations = [
                r for v in validations for r in v["relationship_validation"]
            ]
            valid_rels = sum(
                1
                for r in all_rel_validations
                if r["source_exists"] and r["target_exists"]
            )
            if all_rel_validations:
                print(
                    f"  Relationship validation rate: {valid_rels}/{len(all_rel_validations)} ({valid_rels/len(all_rel_validations)*100:.1f}%)"
                )

    # Save detailed results
    results = {
        "extractions": [e.model_dump() for e in extractions],
        "validations": validations,
        "summary": {
            "total_triggers_tested": len(test_triggers),
            "successful_extractions": len(extractions),
            "total_entities": sum(len(e.entities) for e in extractions),
            "total_relationships": sum(len(e.relationships) for e in extractions),
        },
    }

    with open("knowledge_extraction_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to knowledge_extraction_results.json")
    print(f"✅ Knowledge extraction testing completed!")


if __name__ == "__main__":
    test_knowledge_extraction()
