#!/usr/bin/env python3
"""
Knowledge Graph Builder with Validated LLM Extraction

Builds a proper knowledge+experience graph using the validated LLM extraction
system, creating meaningful nodes and relationships with confidence scoring.
"""

import time
import uuid
from typing import Dict, Optional, Any
from datetime import datetime
import logging

from agent.experiments.knowledge_graph.knowledge_graph_prototype import (
    KnowledgeExperienceGraph,
    GraphNode,
    GraphRelationship,
    NodeType,
)
from agent.experiments.knowledge_graph.llm_knowledge_extraction import (
    LLMKnowledgeExtractor,
    KnowledgeExtraction,
)
from agent.experiments.knowledge_graph.relationship_type_bank import (
    RelationshipTypeBank,
)
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.chain_of_action.trigger import (
    UserInputTrigger,
    WakeupTrigger,
)
from agent.chain_of_action.action.action_types import ActionType
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.state import State
from agent.memory.embedding_service import get_embedding_service
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RelationshipValidation(BaseModel):
    """Result of relationship semantic validation"""

    is_valid: bool = Field(description="Whether this relationship makes semantic sense")
    reasoning: str = Field(description="Why this relationship is valid or invalid")
    suggested_fix: Optional[str] = Field(
        default=None,
        description="If invalid, suggest a better relationship or approach",
    )


class EntitySimilarityMatch(BaseModel):
    """Result of entity similarity matching"""

    is_same_entity: bool = Field(
        description="Whether the proposed entity is the same as an existing entity"
    )
    existing_entity_name: str = Field(
        default="",
        description="The normalized name of the existing entity if it's the same, empty otherwise",
    )
    reasoning: str = Field(
        description="Explanation of why they are the same or different"
    )


class ValidatedKnowledgeGraphBuilder:
    """Builds knowledge graph using validated LLM extraction"""

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel,
        state: State,
        relationship_bank: Optional[RelationshipTypeBank] = None,
    ):
        self.graph = KnowledgeExperienceGraph()
        self.extractor = LLMKnowledgeExtractor(llm, model)
        self.llm = llm
        self.model = model
        # Track historical state progression - will be initialized with starting state
        self.state: State = state

        # Use provided relationship bank or create new one
        self.relationship_bank = relationship_bank or RelationshipTypeBank(
            llm, model, state
        )

        # Track entities across triggers for deduplication
        self.entity_name_to_node_id: Dict[str, str] = {}

        # Track statistics for analysis
        self.entity_evolution_count = 0

        # Embedding service for node similarity and deduplication
        self.embedding_service = get_embedding_service()

    def apply_action_effects_to_state(self, trigger: TriggerHistoryEntry) -> None:
        """Apply the effects of actions in this trigger to the current historical state"""
        if not self.state or not trigger.actions_taken:
            return

        for action in trigger.actions_taken:
            if action.result and action.result.type == "success":
                # Apply state changes based on action results
                if action.type == ActionType.UPDATE_MOOD:
                    # Extract mood change from result
                    mood_result = action.result.content
                    if hasattr(mood_result, "new_mood"):
                        old_mood = self.state.current_mood
                        self.state.current_mood = mood_result.new_mood
                        if hasattr(mood_result, "new_intensity"):
                            self.state.mood_intensity = mood_result.new_intensity
                        logger.debug(
                            f"Applied mood change: {old_mood} -> {mood_result.new_mood}"
                        )

                elif action.type == ActionType.UPDATE_APPEARANCE:
                    # Extract appearance change from result
                    appearance_result = action.result.content
                    if hasattr(appearance_result, "new_appearance"):
                        old_appearance = self.state.current_appearance
                        self.state.current_appearance = appearance_result.new_appearance
                        logger.debug(
                            f"Applied appearance change: {old_appearance[:50]}... -> {appearance_result.new_appearance[:50]}..."
                        )

                # Note: Other action types like SPEAK, THINK don't modify core state
                # but could be extended here if needed

    def process_trigger_incremental(
        self,
        trigger: TriggerHistoryEntry,
        previous_trigger: Optional[TriggerHistoryEntry] = None,
    ) -> bool:
        """Process a single trigger incrementally, building up the knowledge graph"""

        if trigger.entry_id in self.graph.processed_triggers:
            logger.debug(f"Trigger {trigger.entry_id} already processed, skipping")
            return True

        time_start = time.time()

        logger.info(f"Processing trigger: {trigger.entry_id} at {trigger.timestamp}")

        # Always create experience node first
        experience_node = self._create_experience_node(trigger)
        self.graph.add_node(experience_node)

        # Add temporal relationship to previous experience
        if previous_trigger:
            self._add_temporal_relationship(trigger, previous_trigger, experience_node)

        # Get recent nodes for context (last 10 nodes for cross-trigger relationships)
        recent_nodes = []
        if len(self.graph.nodes) > 0:
            # Get most recently added nodes
            sorted_nodes = sorted(
                self.graph.nodes.values(), key=lambda n: n.created_at, reverse=True
            )
            recent_nodes = [
                {
                    "name": n.name,
                    "description": n.description,
                    "type": n.node_type.value,
                }
                for n in sorted_nodes[:10]
            ]

        # Extract knowledge using historical state (not final state)
        state_to_use = self.state
        extraction = self.extractor.extract_knowledge(
            trigger, state_to_use, recent_nodes
        )
        if extraction:
            # Validate extraction quality
            validation = self.extractor.validate_extraction(extraction, trigger)
            logger.info(
                f"Extracted {validation['entities_count']} entities, {validation['relationships_count']} relationships"
            )

            # Only use high-quality extractions
            entity_validation_rate = sum(
                1 for e in validation["entity_validation"] if e["found_in_text"]
            ) / max(1, len(validation["entity_validation"]))

            if (
                entity_validation_rate >= 0.0
            ):  # Ultra low threshold to test evolution mechanism
                self._build_knowledge_nodes(extraction, experience_node, trigger)
                self._build_knowledge_relationships(
                    extraction, experience_node, trigger
                )
                logger.info(
                    f"Added knowledge nodes (validation rate: {entity_validation_rate:.2f})"
                )
            else:
                logger.warning(
                    f"Skipping low-quality extraction (validation rate: {entity_validation_rate:.2f})"
                )
        else:
            logger.warning(
                f"Knowledge extraction failed for trigger {trigger.entry_id}"
            )

        # Mark as processed
        self.graph.processed_triggers.add(trigger.entry_id)

        # Apply action effects to historical state AFTER processing
        self.apply_action_effects_to_state(trigger)

        time_end = time.time()
        logger.info(
            f"Finished processing trigger {trigger.entry_id} in {time_end - time_start:.2f} seconds"
        )

        return True

    def add_trigger(self, trigger: TriggerHistoryEntry) -> bool:
        """Simple interface to add a trigger to the graph (wrapper around process_trigger_incremental)"""
        return self.process_trigger_incremental(trigger)

    def _create_experience_node(self, trigger: TriggerHistoryEntry) -> GraphNode:
        """Create experience node preserving full trigger context"""

        # Build meaningful name and description
        description_parts = []

        if isinstance(trigger.trigger, UserInputTrigger):
            name = (
                f"User said: {trigger.trigger.content[:50]}..."
                if len(trigger.trigger.content) > 50
                else f"User: {trigger.trigger.content}"
            )
        elif isinstance(trigger.trigger, WakeupTrigger):
            name = "Wakeup at " + trigger.timestamp.strftime("%H:%M")
        else:
            name = "Experience at " + trigger.timestamp.strftime("%H:%M")

        description = trigger.compressed_summary or "No summary available."

        experience_node = GraphNode(
            id=f"exp_{trigger.entry_id}",
            node_type=NodeType.EXPERIENCE,
            name=name,
            description=description,
            properties=trigger.model_dump(
                mode="json", include={"trigger", "actions_taken", "timestamp"}
            ),
            source_trigger_id=trigger.entry_id,
        )

        # Generate embedding for the experience node
        self._generate_node_embedding(experience_node)

        return experience_node

    def _generate_node_embedding(self, node: GraphNode) -> None:
        """Generate and set embedding for a node based on name and description"""
        try:
            # Create a comprehensive text for embedding that captures the node's semantics
            embedding_text = f"{node.name}: {node.description}"

            # Add node type context for better semantic clustering
            embedding_text = f"[{node.node_type.value}] {embedding_text}"

            # Generate embedding
            embedding = self.embedding_service.encode(embedding_text)
            node.embedding = embedding

            logger.debug(
                f"Generated embedding for node: {node.name} ({len(embedding)} dimensions)"
            )

        except Exception as e:
            logger.warning(f"Failed to generate embedding for node {node.name}: {e}")
            node.embedding = None

    def _add_temporal_relationship(
        self,
        current_trigger: TriggerHistoryEntry,
        previous_trigger: TriggerHistoryEntry,
        experience_node: GraphNode,
    ) -> None:
        """Add high-confidence temporal relationship"""

        prev_exp_id = f"exp_{previous_trigger.entry_id}"
        if prev_exp_id in self.graph.nodes:
            rel = GraphRelationship(
                id=str(uuid.uuid4()),
                source_node_id=prev_exp_id,
                target_node_id=experience_node.id,
                relationship_type="happened_before",
                confidence=1.0,  # Perfect confidence from timestamps
                strength=1.0,
                properties={
                    "time_diff_seconds": (
                        current_trigger.timestamp - previous_trigger.timestamp
                    ).total_seconds()
                },
                source_trigger_id=current_trigger.entry_id,
            )
            self.graph.add_relationship(rel)

    def _build_knowledge_nodes(
        self,
        extraction: KnowledgeExtraction,
        experience_node: GraphNode,
        trigger: TriggerHistoryEntry,
    ) -> None:
        """Build knowledge nodes from validated extraction"""

        for entity in extraction.entities:
            # Map entity types to node types
            node_type = self._map_entity_type_to_node_type(entity.type)

            # Check if entity already exists using semantic similarity
            existing_entity_name = self._find_similar_entity(
                entity.name, entity.type, entity.description, entity.evidence
            )

            if existing_entity_name:
                # Entity exists, evolve its description with new information
                existing_node_id = self.entity_name_to_node_id[existing_entity_name]
                existing_node = self.graph.nodes[existing_node_id]

                # Evolve the description by combining old and new insights
                evolved_description = self._evolve_entity_description(
                    existing_node.description,
                    entity.description,
                    existing_node.access_count + 1,
                    trigger,
                )

                existing_node.description = evolved_description
                existing_node.access_count += 1
                existing_node.last_accessed = datetime.now()
                existing_node.importance = max(
                    existing_node.importance, entity.confidence
                )

                # Track entity evolution for statistics
                self.entity_evolution_count += 1

                # Regenerate embedding with updated description
                self._generate_node_embedding(existing_node)

                # Add this trigger to sources
                if "source_triggers" not in existing_node.properties:
                    existing_node.properties["source_triggers"] = []
                existing_node.properties["source_triggers"].append(trigger.entry_id)

                knowledge_node = existing_node
            else:
                # Create new knowledge node
                normalized_name = entity.name.lower().strip()
                knowledge_node = GraphNode(
                    id=f"{node_type.value}_{normalized_name.replace(' ', '_')}_{str(uuid.uuid4())[:8]}",
                    node_type=node_type,
                    name=entity.name,
                    description=entity.description,
                    properties={
                        "confidence": entity.confidence,
                        "evidence": entity.evidence,
                        "source_triggers": [trigger.entry_id],
                        "entity_type": entity.type,
                    },
                    importance=entity.confidence,  # Use extraction confidence as importance
                    source_trigger_id=trigger.entry_id,
                )

                # Generate embedding for the new knowledge node
                self._generate_node_embedding(knowledge_node)

                self.graph.add_node(knowledge_node)
                self.entity_name_to_node_id[normalized_name] = knowledge_node.id

            # Create relationship between experience and knowledge node
            # For experience -> entity relationships, strength is based on entity importance
            experience_strength = min(
                entity.confidence + 0.1, 1.0
            )  # Slightly higher than confidence

            rel = GraphRelationship(
                id=str(uuid.uuid4()),
                source_node_id=experience_node.id,
                target_node_id=knowledge_node.id,
                relationship_type="involves",
                confidence=entity.confidence,
                strength=experience_strength,
                properties={"extraction_evidence": entity.evidence},
                source_trigger_id=trigger.entry_id,
            )
            self.graph.add_relationship(rel)

    def _build_knowledge_relationships(
        self,
        extraction: KnowledgeExtraction,
        experience_node: GraphNode,
        trigger: TriggerHistoryEntry,
    ) -> None:
        """Build relationships between knowledge nodes from extraction"""

        for rel_extraction in extraction.relationships:
            # Find the source and target nodes
            source_name = rel_extraction.source_entity.lower().strip()
            target_name = rel_extraction.target_entity.lower().strip()

            source_node_id = self.entity_name_to_node_id.get(source_name)
            target_node_id = self.entity_name_to_node_id.get(target_name)

            if source_node_id and target_node_id:
                # Use relationship bank to get or create relationship type
                rel_type, should_flip = (
                    self.relationship_bank.get_or_create_relationship_type(
                        proposed_type=rel_extraction.relationship_type,
                        description=rel_extraction.description,
                        source_entity=rel_extraction.source_entity,
                        target_entity=rel_extraction.target_entity,
                        context=f"From trigger {trigger.entry_id}: {rel_extraction.evidence}",
                    )
                )

                # Handle direction flipping if needed
                final_source_id = target_node_id if should_flip else source_node_id
                final_target_id = source_node_id if should_flip else target_node_id
                final_source_name = (
                    rel_extraction.target_entity
                    if should_flip
                    else rel_extraction.source_entity
                )
                final_target_name = (
                    rel_extraction.source_entity
                    if should_flip
                    else rel_extraction.target_entity
                )

                if should_flip:
                    logger.info(
                        f"Flipping relationship direction: {rel_extraction.source_entity} -> {rel_extraction.target_entity} "
                        f"becomes {rel_extraction.target_entity} -> {rel_extraction.source_entity} for relationship '{rel_type}'"
                    )

                # Validate relationship semantics
                source_node = self.graph.nodes[final_source_id]
                target_node = self.graph.nodes[final_target_id]

                validation = self._validate_relationship_semantics(
                    source_entity_name=final_source_name,
                    target_entity_name=final_target_name,
                    relationship_type=rel_type,
                    description=rel_extraction.description,
                    source_node_type=source_node.node_type.value,
                    target_node_type=target_node.node_type.value,
                )

                if not validation.is_valid:
                    logger.warning(
                        f"Skipping invalid relationship: {final_source_name} --[{rel_type}]--> {final_target_name}. "
                        f"Reason: {validation.reasoning}"
                    )
                    if validation.suggested_fix:
                        logger.info(f"Suggestion: {validation.suggested_fix}")
                    continue  # Skip this relationship

                # Calculate proper strength based on relationship characteristics
                relationship_strength = self._calculate_relationship_strength(
                    rel_extraction, rel_type, trigger
                )

                # Create relationship
                rel = GraphRelationship(
                    id=str(uuid.uuid4()),
                    source_node_id=final_source_id,
                    target_node_id=final_target_id,
                    relationship_type=rel_type,
                    confidence=rel_extraction.confidence,
                    strength=relationship_strength,
                    properties={
                        "description": rel_extraction.description,
                        "extraction_evidence": rel_extraction.evidence,
                        "original_rel_type": rel_extraction.relationship_type,
                        "direction_flipped": should_flip,
                    },
                    source_trigger_id=trigger.entry_id,
                )
                self.graph.add_relationship(rel)

                logger.debug(
                    f"Added relationship: {rel_extraction.source_entity} -> {rel_type} -> {rel_extraction.target_entity}"
                )
            else:
                logger.debug(
                    f"Skipping relationship - nodes not found: {rel_extraction.source_entity} -> {rel_extraction.target_entity}"
                )

    def _map_entity_type_to_node_type(self, entity_type: str) -> NodeType:
        """Map extracted entity type to graph node type"""

        type_mapping = {
            "person": NodeType.PERSON,
            "emotion": NodeType.EMOTION,
            "concept": NodeType.CONCEPT,
            "goal": NodeType.GOAL,
            "object": NodeType.OBJECT,  # Physical objects get their own type
            "tool": NodeType.OBJECT,  # Tools are objects
            "place": NodeType.OBJECT,  # Places are objects
            "item": NodeType.OBJECT,  # Items are objects
            "topic": NodeType.CONCEPT,
            "theme": NodeType.CONCEPT,
        }

        return type_mapping.get(entity_type.lower(), NodeType.CONCEPT)

    def _map_extracted_relationship_type(self, rel_type: str) -> str:
        """Keep extracted relationship type as free-form string (no more enum mapping)"""

        # Clean up the relationship type but keep it dynamic
        cleaned = rel_type.lower().strip().replace(" ", "_")

        # Some basic normalization but preserve richness
        return cleaned if cleaned else "relates_to"

    def _evolve_entity_description(
        self,
        existing_description: str,
        new_description: str,
        encounter_count: int,
        trigger: TriggerHistoryEntry,
    ) -> str:
        """Evolve entity description by combining old and new insights using LLM"""

        if encounter_count <= 1:
            return new_description  # First encounter, use new description

        # Use LLM to intelligently combine descriptions
        evolution_prompt = f"""I am {self.state.name}. I'm updating my understanding of an entity based on new information.

ENTITY EVOLUTION:
Current Understanding: {existing_description}
New Information: {new_description}
Total Encounters: {encounter_count}

Create a natural description of this entity from my perspective as {self.state.name}:
1. Combines insights from both descriptions
2. Shows cumulative understanding over {encounter_count} encounters
3. Uses natural language like I'm describing someone/something to a friend:
   - Good: "David is someone who consistently greets me warmly"
   - Good: "The trust choker is jewelry that symbolizes my bond with David" 
   - Bad: "Chloe's user is someone who..." (too clinical)
   - Bad: "I am someone who..." (wrong perspective)
4. Keep it concise but richer than either individual description
5. Focus on what makes this entity meaningful to me personally

Description:"""

        try:
            from agent.structured_llm import direct_structured_llm_call
            from pydantic import BaseModel

            class EvolvedDescription(BaseModel):
                description: str

            result = direct_structured_llm_call(
                prompt=evolution_prompt,
                response_model=EvolvedDescription,
                model=self.model,
                llm=self.llm,
                caller="entity_evolution",
                temperature=0.3,
            )

            return result.description

        except Exception as e:
            logger.warning(f"Entity description evolution failed: {e}")
            # Fallback: simple combination
            return f"{existing_description} {new_description}".strip()

    def _find_similar_entity(
        self,
        entity_name: str,
        entity_type: str,
        entity_description: str,
        entity_evidence: str,
    ) -> str:
        """Find if there's an existing entity that represents the same thing using embedding-based similarity"""

        if not self.entity_name_to_node_id:
            return ""  # No existing entities

        # Get entities of the same type for comparison
        candidate_entities = []
        for existing_name, node_id in self.entity_name_to_node_id.items():
            existing_node = self.graph.nodes[node_id]
            # Only compare entities of the same type to avoid false matches
            if existing_node.node_type.value == entity_type:
                candidate_entities.append((existing_name, existing_node))

        if not candidate_entities:
            return ""  # No entities of this type exist yet

        # Generate embedding for the new entity
        new_entity_text = f"[{entity_type}] {entity_name}: {entity_description}"
        try:
            new_entity_embedding = self.embedding_service.encode(new_entity_text)
        except Exception as e:
            logger.warning(
                f"Failed to generate embedding for entity similarity check: {e}"
            )
            # Fallback to word-based filtering
            return self._find_similar_entity_fallback(
                entity_name,
                entity_type,
                entity_description,
                entity_evidence,
                candidate_entities,
            )

        # Calculate embedding similarities with all candidates
        embedding_matches = []
        for existing_name, existing_node in candidate_entities:
            if existing_node.embedding is None:
                # Generate missing embedding on the fly
                self._generate_node_embedding(existing_node)

            if existing_node.embedding is not None:
                similarity = self.embedding_service.cosine_similarity(
                    new_entity_embedding, existing_node.embedding
                )
                embedding_matches.append((similarity, existing_name, existing_node))

        # Sort by similarity score (highest first)
        embedding_matches.sort(key=lambda x: x[0], reverse=True)

        # Only consider high-similarity candidates for LLM review
        high_similarity_threshold = (
            0.85  # Cosine similarity > 0.85 suggests likely match
        )
        potential_matches = [
            (name, node)
            for sim, name, node in embedding_matches
            if sim >= high_similarity_threshold
        ]

        if not potential_matches:
            return ""  # No high-similarity matches found

        logger.debug(
            f"Embedding similarity found {len(potential_matches)} candidates for '{entity_name}' (max similarity: {embedding_matches[0][0]:.3f})"
        )

        # Use LLM to check semantic similarity for top embedding matches only
        for existing_name, existing_node in potential_matches[
            :3
        ]:  # Limit to top 3 for efficiency
            try:
                similarity_match = self._check_entity_similarity(
                    entity_name,
                    entity_description,
                    entity_evidence,
                    entity_type,
                    existing_name,
                    existing_node.description,
                )

                if similarity_match.is_same_entity:
                    logger.info(
                        f"Found existing entity match: '{entity_name}' -> '{existing_name}' "
                        f"(reason: {similarity_match.reasoning})"
                    )
                    return existing_name

            except Exception as e:
                logger.warning(
                    f"Entity similarity check failed for {entity_name} vs {existing_name}: {e}"
                )
                continue

        return ""  # No similar entity found

    def _find_similar_entity_fallback(
        self,
        entity_name: str,
        entity_type: str,
        entity_description: str,
        entity_evidence: str,
        candidate_entities,
    ) -> str:
        """Fallback word-based similarity when embeddings fail"""
        potential_matches = []
        entity_words = set(entity_name.lower().split())

        for existing_name, existing_node in candidate_entities:
            existing_words = set(existing_name.split())  # Already normalized
            # Check if there's word overlap or very similar length
            word_overlap = len(entity_words.intersection(existing_words))
            if word_overlap > 0 or abs(len(entity_name) - len(existing_name)) < 5:
                potential_matches.append((existing_name, existing_node))

        # Use LLM to check the potential matches
        for existing_name, existing_node in potential_matches:
            try:
                similarity_match = self._check_entity_similarity(
                    entity_name,
                    entity_description,
                    entity_evidence,
                    entity_type,
                    existing_name,
                    existing_node.description,
                )

                if similarity_match.is_same_entity:
                    logger.info(
                        f"Found existing entity match (fallback): '{entity_name}' -> '{existing_name}'"
                    )
                    return existing_name

            except Exception as e:
                logger.warning(f"Fallback entity similarity check failed: {e}")
                continue

        return ""  # No similar entity found in fallback

    def _validate_relationship_semantics(
        self,
        source_entity_name: str,
        target_entity_name: str,
        relationship_type: str,
        description: str,
        source_node_type: str,
        target_node_type: str,
    ) -> RelationshipValidation:
        """Validate whether a relationship makes semantic sense using LLM"""

        prompt = f"""I need to validate whether a proposed relationship makes semantic sense.

PROPOSED RELATIONSHIP:
{source_entity_name} --[{relationship_type}]--> {target_entity_name}

DETAILS:
- Source entity: "{source_entity_name}" (type: {source_node_type})
- Target entity: "{target_entity_name}" (type: {target_node_type}) 
- Relationship: "{relationship_type}"
- Description: "{description}"

VALIDATION CRITERIA:
1. Does this relationship make logical/semantic sense?
2. Are the entity types compatible with this relationship?
3. Is the direction correct (A relates to B vs B relates to A)?

COMMON INVALID PATTERNS:
- Physical objects "expressing" people (clothes can express emotions, not people)
- Emotions "causing" physical objects (emotions don't create tools)
- Backwards causation (effects don't cause their causes)
- Category confusion (abstract concepts in physical relationships)

Validate this relationship and explain your reasoning. If invalid, suggest a better approach."""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=RelationshipValidation,
                model=self.model,
                llm=self.llm,
                caller="relationship_validation",
                temperature=0.1,  # Low temperature for consistent validation
            )

            return result

        except Exception as e:
            logger.warning(f"Relationship validation failed: {e}")
            # Fallback: assume valid to avoid blocking legitimate relationships
            return RelationshipValidation(
                is_valid=True,
                reasoning="LLM validation failed, assuming valid to avoid false negatives",
            )

    def _check_entity_similarity(
        self,
        new_entity_name: str,
        new_entity_description: str,
        new_entity_evidence: str,
        entity_type: str,
        existing_entity_name: str,
        existing_entity_description: str,
    ) -> EntitySimilarityMatch:
        """Use LLM to determine if two entities represent the same thing"""

        prompt = f"""I am analyzing whether two entities represent the same thing.

NEW ENTITY: "{new_entity_name}" (type: {entity_type})
Description: {new_entity_description}
Evidence: {new_entity_evidence}

EXISTING ENTITY: "{existing_entity_name}" (type: {entity_type})
Description: {existing_entity_description}

Are these the same entity? Consider:
1. Do they refer to the same specific thing, person, concept, or emotion?
2. Do the descriptions describe the same underlying entity?
3. Could minor name variations (word order, synonyms) represent the same entity?
4. Are they genuinely different things that happen to have similar names?

Examples of SAME entities:
- "search web tool" and "web search tool" with similar descriptions about web searching
- "David" and "david" referring to the same person
- "gaming setup" and "gaming rig" both describing gaming equipment
- "excitement" and "excitement" with similar emotional contexts

Examples of DIFFERENT entities:
- "trust" (emotion) and "trust" (legal concept) - different descriptions/contexts
- "home" (physical place) and "home" (feeling of belonging) - different descriptions
- "apple" (fruit description) and "apple" (company description) - different contexts

Compare both the names AND descriptions to determine if they represent the same real-world entity.
If they are the same entity, provide the existing entity's normalized name."""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=EntitySimilarityMatch,
                model=self.model,
                llm=self.llm,
                caller="entity_similarity_matching",
                temperature=0.2,
            )
            return result
        except Exception as e:
            logger.error(f"Entity similarity matching failed: {e}")
            return EntitySimilarityMatch(
                is_same_entity=False,
                existing_entity_name="",
                reasoning="LLM similarity matching failed, treating as different entities",
            )

    def _calculate_relationship_strength(
        self, rel_extraction, rel_type: str, trigger: TriggerHistoryEntry
    ) -> float:
        """Calculate relationship strength based on characteristics beyond just confidence"""

        base_strength = rel_extraction.confidence

        # Adjust strength based on relationship type importance
        high_importance_types = [
            "loves",
            "creates",
            "owns",
            "is",
            "causes",
            "makes_me_feel",
            "symbolizes",
            "represents",
            "depends_on",
        ]
        medium_importance_types = [
            "prefers",
            "likes",
            "uses",
            "has",
            "contains",
            "connects_to",
            "influences",
            "affects",
        ]

        if any(
            important_type in rel_type.lower()
            for important_type in high_importance_types
        ):
            strength_modifier = 0.15  # Boost high-importance relationships
        elif any(
            medium_type in rel_type.lower() for medium_type in medium_importance_types
        ):
            strength_modifier = 0.05  # Small boost for medium importance
        else:
            strength_modifier = -0.05  # Small penalty for generic relationships

        # Adjust based on evidence length/detail
        evidence_length = len(rel_extraction.evidence)
        if evidence_length > 100:
            strength_modifier += (
                0.05  # Detailed evidence suggests stronger relationship
            )
        elif evidence_length < 30:
            strength_modifier -= 0.05  # Weak evidence suggests weaker relationship

        # Adjust based on description quality
        if any(
            word in rel_extraction.description.lower()
            for word in ["always", "never", "extremely", "very", "deeply"]
        ):
            strength_modifier += 0.1  # Strong qualifiers suggest strong relationships

        final_strength = min(max(base_strength + strength_modifier, 0.1), 1.0)

        logger.debug(
            f"Relationship strength calculation: {rel_type} = {base_strength:.2f} + {strength_modifier:.2f} = {final_strength:.2f}"
        )

        return final_strength

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        return self.graph.get_stats()

    def save_graph(self, filename: str) -> None:
        """Save the built graph"""
        self.graph.save_to_file(filename)


def test_validated_graph_builder():
    """Test the validated knowledge graph builder"""

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

    # Build graph with first 5 triggers (for testing entity evolution)
    builder = ValidatedKnowledgeGraphBuilder(llm, model, state)
    all_triggers = trigger_history.get_all_entries()[:5]

    print(f"\n🏗️  Building knowledge graph from {len(all_triggers)} triggers...")

    previous_trigger = None
    successful_count = 0

    for i, trigger in enumerate(all_triggers):
        print(f"  Processing trigger {i+1}/{len(all_triggers)}: {trigger.entry_id}")

        success = builder.process_trigger_incremental(trigger, previous_trigger)
        if success:
            successful_count += 1

        previous_trigger = trigger

        # Progress update every 5 triggers
        if (i + 1) % 5 == 0:
            stats = builder.get_stats()
            print(
                f"    Current stats: {stats['total_nodes']} nodes, {stats['total_relationships']} relationships"
            )

    # Final statistics
    print(f"\n📊 Final Graph Statistics:")
    stats = builder.get_stats()

    for key, value in stats.items():
        if isinstance(value, dict) and value:
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        elif not isinstance(value, dict):
            print(f"  {key}: {value}")

    print(
        f"\n  Successfully processed: {successful_count}/{len(all_triggers)} triggers"
    )

    # Save the validated graph
    builder.save_graph("validated_knowledge_graph.json")
    print(f"\n💾 Validated knowledge graph saved to validated_knowledge_graph.json")

    # Show some sample nodes and relationships
    print(f"\n🔍 Sample Knowledge Nodes:")
    knowledge_nodes = [
        node
        for node in builder.graph.nodes.values()
        if node.node_type != NodeType.EXPERIENCE
    ][:5]

    for node in knowledge_nodes:
        print(
            f"  [{node.node_type.value}] {node.name} (importance: {node.importance:.2f})"
        )
        print(f"    {node.description}")
        source_count = len(node.properties.get("source_triggers", []))
        print(f"    Sources: {source_count} triggers, Access: {node.access_count}")

    print(f"\n🔗 Sample Relationships:")
    sample_rels = list(builder.graph.relationships.values())[:5]
    for rel in sample_rels:
        source_name = builder.graph.nodes[rel.source_node_id].name
        target_name = builder.graph.nodes[rel.target_node_id].name
        print(
            f"  {source_name} --[{rel.relationship_type}]--> {target_name} (conf: {rel.confidence:.2f})"
        )

    print(f"\n✅ Validated knowledge graph building completed!")


if __name__ == "__main__":
    test_validated_graph_builder()
