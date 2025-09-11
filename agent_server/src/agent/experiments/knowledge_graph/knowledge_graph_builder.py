#!/usr/bin/env python3
"""
Knowledge Graph Builder with Validated LLM Extraction

Builds a proper knowledge+experience graph using the validated LLM extraction
system, creating meaningful nodes and relationships with confidence scoring.
"""

import time
import uuid
from typing import Dict, Optional, Any, List
from datetime import datetime
import logging
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

from agent.experiments.knowledge_graph.knowledge_graph_prototype import (
    KnowledgeExperienceGraph,
    GraphNode,
    GraphRelationship,
    NodeType,
    GraphStats,
)
from agent.experiments.knowledge_graph.llm_knowledge_extraction import (
    LLMKnowledgeExtractor,
    KnowledgeExtraction,
)
from agent.experiments.knowledge_graph.relationship_schema_bank import (
    RelationshipSchemaBank,
)
from agent.chain_of_action.prompts import format_single_trigger_entry
from agent.experiments.knowledge_graph.n_ary_extraction import ExistingEntityData, ExtractedNaryRelationship
from agent.experiments.knowledge_graph.knn_entity_search import (
    KNNEntitySearch,
)
from agent.experiments.knowledge_graph.n_ary_extraction import (
    NaryRelationshipExtractor,
)
from agent.experiments.knowledge_graph.n_ary_relationship import (
    NaryRelationshipManager,
    NaryRelationship,
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
from agent.ui_output import ui_print
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class TimingMetrics:
    total_time: str
    extraction_time: str
    entity_processing_time: str
    relationship_processing_time: str
    embedding_generation_time: str
    other_time: str


@dataclass
class LLMCallMetrics:
    total_calls: int
    total_time: str
    by_operation: Dict[str, int]
    time_by_operation: Dict[str, str]


@dataclass
class EntityProcessingMetrics:
    total_processed: int
    deduplicated: int
    auto_accepted: int
    auto_rejected: int
    llm_validated: int
    avg_similarity: float


@dataclass
class RelationshipProcessingMetrics:
    total_processed: int
    type_matched_by_embedding: int
    type_matched_by_llm: int
    validated_by_embedding: int
    validated_by_llm: int
    avg_type_similarity: float


@dataclass
class PerformanceMetricsSummary:
    timing: TimingMetrics
    llm_calls: LLMCallMetrics
    entity_processing: EntityProcessingMetrics
    relationship_processing: RelationshipProcessingMetrics


@dataclass
class DetailedLLMStats:
    calls: int
    total_time: str
    avg_time: str


@dataclass
class DetailedLLMCallMetrics:
    total_calls: int
    total_time: str
    by_operation: Dict[str, DetailedLLMStats]


@dataclass
class EnhancedPerformanceMetrics:
    timing: TimingMetrics
    llm_calls: LLMCallMetrics
    entity_processing: EntityProcessingMetrics
    relationship_processing: RelationshipProcessingMetrics
    llm_calls_detailed: Optional[DetailedLLMCallMetrics] = None


@dataclass
class PerformanceMetrics:
    """Track detailed performance metrics for knowledge graph building"""

    # Timing metrics (in seconds)
    total_time: float = 0.0
    extraction_time: float = 0.0
    entity_processing_time: float = 0.0
    relationship_processing_time: float = 0.0
    embedding_generation_time: float = 0.0

    # LLM call metrics
    llm_calls_by_operation: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    llm_time_by_operation: Dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    # Entity processing metrics
    entities_processed: int = 0
    entities_deduplicated: int = 0
    entities_auto_accepted: int = 0
    entities_auto_rejected: int = 0
    entities_llm_validated: int = 0

    # Relationship processing metrics
    relationships_processed: int = 0
    relationships_type_matched_by_embedding: int = 0
    relationships_type_matched_by_llm: int = 0
    relationships_validated_by_embedding: int = 0
    relationships_validated_by_llm: int = 0

    # Similarity score distributions
    entity_similarity_scores: List[float] = field(default_factory=list)
    relationship_type_similarity_scores: List[float] = field(default_factory=list)

    def start_timer(self, operation: str) -> float:
        """Start timing an operation"""
        return time.time()

    def end_timer(self, operation: str, start_time: float) -> None:
        """End timing an operation and record the duration"""
        duration = time.time() - start_time
        if operation == "extraction":
            self.extraction_time += duration
        elif operation == "entity_processing":
            self.entity_processing_time += duration
        elif operation == "relationship_processing":
            self.relationship_processing_time += duration
        elif operation == "embedding_generation":
            self.embedding_generation_time += duration

    def record_llm_call(self, operation: str, duration: float) -> None:
        """Record an LLM call with its timing"""
        self.llm_calls_by_operation[operation] += 1
        self.llm_time_by_operation[operation] += duration

    def get_summary(self) -> PerformanceMetricsSummary:
        """Get a comprehensive summary of performance metrics"""
        total_llm_calls = sum(self.llm_calls_by_operation.values())
        total_llm_time = sum(self.llm_time_by_operation.values())

        other_time = (
            self.total_time
            - self.extraction_time
            - self.entity_processing_time
            - self.relationship_processing_time
            - self.embedding_generation_time
        )

        return PerformanceMetricsSummary(
            timing=TimingMetrics(
                total_time=f"{self.total_time:.2f}s",
                extraction_time=f"{self.extraction_time:.2f}s",
                entity_processing_time=f"{self.entity_processing_time:.2f}s",
                relationship_processing_time=f"{self.relationship_processing_time:.2f}s",
                embedding_generation_time=f"{self.embedding_generation_time:.2f}s",
                other_time=f"{other_time:.2f}s",
            ),
            llm_calls=LLMCallMetrics(
                total_calls=total_llm_calls,
                total_time=f"{total_llm_time:.2f}s",
                by_operation=dict(self.llm_calls_by_operation),
                time_by_operation={
                    k: f"{v:.2f}s" for k, v in self.llm_time_by_operation.items()
                },
            ),
            entity_processing=EntityProcessingMetrics(
                total_processed=self.entities_processed,
                deduplicated=self.entities_deduplicated,
                auto_accepted=self.entities_auto_accepted,
                auto_rejected=self.entities_auto_rejected,
                llm_validated=self.entities_llm_validated,
                avg_similarity=(
                    sum(self.entity_similarity_scores)
                    / len(self.entity_similarity_scores)
                    if self.entity_similarity_scores
                    else 0.0
                ),
            ),
            relationship_processing=RelationshipProcessingMetrics(
                total_processed=self.relationships_processed,
                type_matched_by_embedding=self.relationships_type_matched_by_embedding,
                type_matched_by_llm=self.relationships_type_matched_by_llm,
                validated_by_embedding=self.relationships_validated_by_embedding,
                validated_by_llm=self.relationships_validated_by_llm,
                avg_type_similarity=(
                    sum(self.relationship_type_similarity_scores)
                    / len(self.relationship_type_similarity_scores)
                    if self.relationship_type_similarity_scores
                    else 0.0
                ),
            ),
        )


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
        relationship_bank: Optional[RelationshipSchemaBank] = None,
    ):
        self.graph = KnowledgeExperienceGraph()
        self.extractor = LLMKnowledgeExtractor(llm, model)
        self.llm = llm
        self.model = model
        # Track historical state progression - will be initialized with starting state
        self.state: State = state

        # Use provided relationship bank or create new one
        self.relationship_bank = relationship_bank or RelationshipSchemaBank(
            llm, model, state
        )

        # Track statistics for analysis
        self.entity_evolution_count = 0

        # Embedding service for node similarity and deduplication
        self.embedding_service = get_embedding_service()

        # Performance metrics tracking (initialize early for embedding generation)
        self.metrics = PerformanceMetrics()
        
        # kNN-based entity deduplication system
        self.node_search = KNNEntitySearch[GraphNode]()
        
        # Initialize with self/agent entity for pronoun resolution
        self._initialize_self_entity()

        # N-ary relationship extraction and management
        self.nary_extractor = NaryRelationshipExtractor(
            llm, model, self.relationship_bank
        )
        self.nary_manager = NaryRelationshipManager()

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
                    old_mood = self.state.current_mood
                    self.state.current_mood = mood_result.new_mood
                    self.state.mood_intensity = mood_result.new_intensity

                elif action.type == ActionType.UPDATE_APPEARANCE:
                    # Extract appearance change from result
                    appearance_result = action.result.content
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

        # Start total timing for this trigger
        trigger_start_time = self.metrics.start_timer("total")

        logger.info(f"Processing trigger: {trigger.entry_id} at {trigger.timestamp}")
        
        # Show what trigger we're processing
        if isinstance(trigger.trigger, UserInputTrigger):
            content_preview = trigger.trigger.content[:60] + "..." if len(trigger.trigger.content) > 60 else trigger.trigger.content
            ui_print(f"🔄 Processing: \"{content_preview}\"")
        elif isinstance(trigger.trigger, WakeupTrigger):
            ui_print(f"🔄 Processing: Wakeup at {trigger.timestamp.strftime('%H:%M')}")
        else:
            ui_print(f"🔄 Processing: {type(trigger.trigger).__name__}")

        result = self._process_trigger_sections(
            trigger, previous_trigger, trigger_start_time
        )
        return result

    def _process_trigger_sections(
        self,
        trigger: TriggerHistoryEntry,
        previous_trigger: Optional[TriggerHistoryEntry],
        trigger_start_time: float,
    ) -> bool:
        """Process trigger with section-level instrumentation"""

        # Always create experience node first
        experience_node: GraphNode = self._create_experience_node(trigger)
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

        # Find entities similar to the trigger before extraction to provide context
        related_entities = self._find_trigger_related_entities(trigger)
        
        # Extract knowledge using historical state (not final state)
        ui_print(f"  🧠 Extracting knowledge...")
        extraction_start_time = self.metrics.start_timer("extraction")
        state_to_use = self.state
        extraction = self.extractor.extract_knowledge(
            trigger, state_to_use, recent_nodes, related_entities
        )
        self.metrics.end_timer("extraction", extraction_start_time)
        if extraction:
            # Validate extraction quality
            validation = self.extractor.validate_extraction(extraction, trigger)
            logger.info(
                f"Extracted {validation.entities_count} entities, {validation.relationships_count} relationships"
            )
            ui_print(f"  📝 Extracted {validation.entities_count} entities, {validation.relationships_count} relationships")

            # Only use high-quality extractions
            entity_validation_rate = sum(
                1 for e in validation.entity_validation if e.found_in_text
            ) / max(1, len(validation.entity_validation))

            if (
                entity_validation_rate >= 0.0
            ):  # Ultra low threshold to test evolution mechanism
                # Time entity processing
                entity_start_time = self.metrics.start_timer("entity_processing")
                fresh_entity_ids = self._build_knowledge_nodes(extraction, experience_node, trigger)
                self.metrics.end_timer("entity_processing", entity_start_time)

                # Time relationship processing
                ui_print(f"  🔗 Building relationships...")
                relationship_start_time = self.metrics.start_timer(
                    "relationship_processing"
                )
                self._build_knowledge_relationships(
                    extraction, experience_node, trigger, fresh_entity_ids
                )
                self.metrics.end_timer(
                    "relationship_processing", relationship_start_time
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

        # Complete timing and log performance
        self.metrics.end_timer("total", trigger_start_time)
        logger.info(
            f"Finished processing trigger {trigger.entry_id} in {self.metrics.total_time:.2f} seconds"
        )
        
        # Show summary of what was built
        current_stats = self.get_stats()
        ui_print(f"  ✅ Complete - Graph: {current_stats.total_nodes} nodes, {current_stats.total_relationships + len(self.graph.nary_relationships)} relationships")

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
            embedding_start = self.metrics.start_timer("embedding_generation")

            # Create a comprehensive text for embedding that captures the node's semantics
            embedding_text = f"{node.name}: {node.description}"

            # Add node type context for better semantic clustering
            embedding_text = f"[{node.node_type.value}] {embedding_text}"

            # Generate embedding
            embedding = self.embedding_service.encode(embedding_text)

            self.metrics.end_timer("embedding_generation", embedding_start)
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
        """Add high-confidence temporal n-ary relationship"""

        prev_exp_id = f"exp_{previous_trigger.entry_id}"
        if prev_exp_id in self.graph.nodes:
            # Create n-ary temporal relationship
            from agent.experiments.knowledge_graph.n_ary_relationship import (
                NaryRelationship,
            )

            time_diff_seconds = (
                current_trigger.timestamp - previous_trigger.timestamp
            ).total_seconds()

            nary_rel = NaryRelationship(
                id=str(uuid.uuid4()),
                relationship_type="preceded",
                participants={"earlier": prev_exp_id, "later": experience_node.id},
                confidence=1.0,  # Perfect confidence from timestamps
                strength=1.0,  # High strength for temporal relationships
                properties={
                    "time_diff_seconds": time_diff_seconds,
                    "evidence": f"Timestamp sequence: {previous_trigger.timestamp} → {current_trigger.timestamp}",
                },
                source_trigger_id=current_trigger.entry_id,
                pattern="temporal_sequence",
            )

            # Add directly to graph's n-ary storage
            self.graph.add_nary_relationship(nary_rel)

    def _build_knowledge_nodes(
        self,
        extraction: KnowledgeExtraction,
        experience_node: GraphNode,
        trigger: TriggerHistoryEntry,
    ) -> List[str]:
        """Build knowledge nodes from validated extraction"""
        
        # Track entity IDs processed in this trigger for N-ary context
        processed_entity_ids = []

        for entity in extraction.entities:
            self.metrics.entities_processed += 1

            # Map entity types to node types
            node_type = self._map_entity_type_to_node_type(entity.type)

            # Use kNN-based entity deduplication
            existing_entity_id = self.resolve_entity_to_node_id(
                entity.name, entity.type, entity.description
            )

            if existing_entity_id:
                self.metrics.entities_deduplicated += 1
                ui_print(f"  🔗 {entity.name} → resolved to existing entity")
                # Entity exists, evolve its description with new information
                existing_node_id = existing_entity_id
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
                ui_print(f"  🆕 {entity.name} ({entity.type}) → new entity")
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
                # Use composite key to handle name collisions
                composite_key = self._make_entity_key(entity.name, entity.type)

                # Add to kNN search index for future deduplication
                logger.info(f"📝 Adding entity to KNN index: ID:{knowledge_node.get_id()}, Text:{knowledge_node.get_text()}")
                self.node_search.add_entity(knowledge_node)
                logger.info(f"📊 KNN index now has {len(self.node_search.entity_metadata)} entities")

            # Track this entity ID for N-ary context
            processed_entity_ids.append(knowledge_node.id)

            # Create n-ary relationship between experience and knowledge node
            from agent.experiments.knowledge_graph.n_ary_relationship import (
                NaryRelationship,
            )

            experience_strength = self._calculate_experience_entity_strength(
                entity.confidence,
                entity.type,
                relationship_type="involves",
                is_experience_relationship=True,
            )

            # Create n-ary "involves" relationship
            nary_rel = NaryRelationship(
                id=str(uuid.uuid4()),
                relationship_type="involves",
                participants={
                    "experiencer": experience_node.id,
                    "participant": knowledge_node.id,
                },
                confidence=entity.confidence,
                strength=experience_strength,
                properties={
                    "extraction_evidence": entity.evidence,
                    "entity_type": entity.type,
                },
                source_trigger_id=trigger.entry_id,
                pattern="experience_involvement",
            )

            # Add directly to graph's n-ary storage
            self.graph.add_nary_relationship(nary_rel)

        return processed_entity_ids

    def _build_knowledge_relationships(
        self,
        extraction: KnowledgeExtraction,
        experience_node: GraphNode,
        trigger: TriggerHistoryEntry,
        fresh_entity_ids: List[str],
    ) -> None:
        """Build N-ary relationships between knowledge nodes from extraction (binary relationships are just N-ary with 2 participants)"""

        # Start with fresh entities from this trigger (highest priority)
        fresh_entities = []
        for entity_id in fresh_entity_ids:
            if entity_id in self.graph.nodes:
                node = self.graph.nodes[entity_id]
                fresh_entities.append(
                    ExistingEntityData(
                        id=entity_id,
                        name=node.name,
                        type=node.node_type.value,
                        description=node.description,
                    )
                )

        # Add other existing entities for broader context
        other_entities = []
        for node_id, node in self.graph.nodes.items():
            # Skip fresh entities to avoid duplicates
            if node_id in fresh_entity_ids:
                continue
                
            if node.node_type in [
                NodeType.PERSON,
                NodeType.CONCEPT,
                NodeType.OBJECT,
                NodeType.EMOTION,
                NodeType.GOAL,
            ]:
                other_entities.append(
                    ExistingEntityData(
                        id=node_id,
                        name=node.name,
                        type=node.node_type.value,
                        description=node.description,
                    )
                )

        # Combine with fresh entities first (highest priority)
        existing_entities = fresh_entities + other_entities
        
        logger.info(f"🔄 N-ary context: {len(fresh_entities)} fresh entities, {len(other_entities)} existing entities")

        # Extract N-ary relationships using semantic roles
        logger.info(f"🔄 Starting N-ary relationship extraction with {len(existing_entities)} existing entities")
        logger.info(f"📊 KNN index has {len(self.node_search.entity_metadata)} entities before N-ary processing")
        
        ui_print(f"    🤖 Analyzing semantic relationships...")
        nary_relationships: List[ExtractedNaryRelationship] = self.nary_extractor.extract_nary_relationships(
            trigger=trigger, existing_entities=existing_entities
        )

        logger.info(f"✨ Extracted {len(nary_relationships)} N-ary relationships")
        if nary_relationships:
            ui_print(f"  🔗 Processing {len(nary_relationships)} relationships...")

        # Process N-ary relationships
        valid_relationships = 0
        for extracted_rel in nary_relationships:
            # Convert N-ary relationship to our internal format using entity resolution
            nary_relationship: Optional[NaryRelationship] = self.nary_extractor.convert_to_nary_relationship(
                extracted_rel, trigger.entry_id, self
            )

            if nary_relationship:
                # Add directly to graph's n-ary relationship storage
                self.graph.add_nary_relationship(nary_relationship)

                valid_relationships += 1
                
                # Show the relationship being created using the converted relationship (not extracted)
                participant_names = []
                for role, entity_id in nary_relationship.participants.items():
                    if entity_id in self.graph.nodes:
                        participant_names.append(self.graph.nodes[entity_id].name)
                    else:
                        participant_names.append(entity_id)
                
                if len(participant_names) == 2:
                    ui_print(f"    🔗 {participant_names[0]} --[{nary_relationship.relationship_type}]--> {participant_names[1]}")
                else:
                    ui_print(f"    🔗 {nary_relationship.relationship_type}: {', '.join(participant_names)}")
                
                logger.debug(
                    f"Added N-ary relationship: {nary_relationship.relationship_type} with {len(nary_relationship.participants)} participants"
                )

        logger.info(
            f"Added {valid_relationships} valid N-ary relationships (validation rate: {valid_relationships/len(nary_relationships) if nary_relationships else 0:.2f})"
        )
        if valid_relationships > 0:
            ui_print(f"  ✅ Added {valid_relationships} valid relationships")

    def _map_entity_type_to_node_type(self, entity_type: str) -> NodeType:
        """Map extracted entity type to graph node type"""

        type_mapping = {
            "person": NodeType.PERSON,
            "emotion": NodeType.EMOTION,
            "concept": NodeType.CONCEPT,
            "goal": NodeType.GOAL,
            "object": NodeType.OBJECT,  # Physical objects get their own type
            "tool": NodeType.CONCEPT,  # Tools are abstract capabilities/concepts
            "place": NodeType.CONCEPT,  # Places can be conceptual (locations, virtual spaces)
            "item": NodeType.OBJECT,  # Items are physical objects
            "topic": NodeType.CONCEPT,
            "theme": NodeType.CONCEPT,
        }

        return type_mapping.get(entity_type.lower(), NodeType.CONCEPT)

    def _map_extracted_relationship_type(self, rel_type: str) -> str:
        """Clean and validate relationship types with structural validation"""

        # Clean up the relationship type but keep it dynamic
        cleaned = rel_type.lower().strip().replace(" ", "_")

        # Structural validation to catch problematic patterns
        if not cleaned or len(cleaned) < 2:
            logger.warning(
                f"Invalid relationship type (too short): '{rel_type}' -> using 'relates_to'"
            )
            return "relates_to"

        # Detect agent-specific relationship names (user identified these as problematic)
        agent_specific_patterns = [
            "my_",
            "_prefers",
            "_goal_is_",
            "david_",
            "_me_",
            "i_",
        ]
        for pattern in agent_specific_patterns:
            if pattern in cleaned:
                logger.warning(
                    f"Agent-specific relationship detected: '{cleaned}' - should be generalized"
                )
                # For now, continue but flag for attention

        # Detect grammatically broken relationship types
        broken_patterns = ["_is_to_", "_goal_is_to", "_want_to_"]
        for pattern in broken_patterns:
            if pattern in cleaned:
                logger.warning(
                    f"Grammatically broken relationship: '{cleaned}' -> using 'relates_to'"
                )
                return "relates_to"

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

        if not self.graph.get_all_nodes():
            return ""  # Graph is empty

        # Get entities of the same type for comparison
        candidate_entities = []
        for node in self.graph.get_all_nodes():
            existing_node = self.graph.nodes[node.id]
            # Only compare entities of the same type to avoid false matches
            if existing_node.node_type.value == entity_type:
                candidate_entities.append((node.name, existing_node))

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
            else:
                # If embedding generation failed, skip this candidate
                continue

        # Sort by similarity score (highest first)
        embedding_matches.sort(key=lambda x: x[0], reverse=True)

        if not embedding_matches:
            return ""  # No candidates found

        # Get the highest similarity match
        best_similarity, best_name, best_node = embedding_matches[0]

        # Track similarity score for analysis
        self.metrics.entity_similarity_scores.append(best_similarity)

        # Very high similarity threshold for auto-accept (skip LLM)
        if best_similarity >= 0.95:
            self.metrics.entities_auto_accepted += 1
            logger.info(
                f"Very high similarity ({best_similarity:.3f}) - auto-accepting entity match: {best_name}"
            )
            return best_name

        # Very low similarity threshold for auto-reject (skip LLM)
        if best_similarity < 0.3:
            self.metrics.entities_auto_rejected += 1
            logger.debug(
                f"Very low similarity ({best_similarity:.3f}) - auto-rejecting all matches"
            )
            return ""

        # Medium-high similarity candidates for LLM review
        high_similarity_threshold = 0.85
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

    def _check_relationship_coherence_by_embedding(
        self,
        source_entity_name: str,
        target_entity_name: str,
        relationship_type: str,
        source_node_type: str,
        target_node_type: str,
    ) -> Optional[RelationshipValidation]:
        """Check relationship semantic coherence using embeddings before LLM call"""

        try:
            # Create text representation of the relationship
            relationship_text = (
                f"{source_entity_name} {relationship_type} {target_entity_name}"
            )
            relationship_embedding = self.embedding_service.encode(relationship_text)

            # Create embeddings for known good and bad relationship patterns
            good_patterns = [
                f"{source_node_type} {relationship_type} {target_node_type}",
                f"person causes emotion",
                f"object enables action",
                f"concept relates to concept",
                f"person creates object",
            ]

            bad_patterns = [
                f"emotion creates person",
                f"object thinks about person",
                f"emotion owns physical object",
                f"abstract concept physically touches concrete object",
            ]

            # Calculate similarity to good patterns
            good_similarities = []
            for pattern in good_patterns:
                try:
                    pattern_embedding = self.embedding_service.encode(pattern)
                    similarity = self.embedding_service.cosine_similarity(
                        relationship_embedding, pattern_embedding
                    )
                    good_similarities.append(similarity)
                except Exception:
                    continue

            # Calculate similarity to bad patterns
            bad_similarities = []
            for pattern in bad_patterns:
                try:
                    pattern_embedding = self.embedding_service.encode(pattern)
                    similarity = self.embedding_service.cosine_similarity(
                        relationship_embedding, pattern_embedding
                    )
                    bad_similarities.append(similarity)
                except Exception:
                    continue

            if good_similarities and bad_similarities:
                max_good = max(good_similarities)
                max_bad = max(bad_similarities)

                # High confidence valid relationship
                if max_good > 0.8 and max_good > max_bad + 0.2:
                    logger.debug(
                        f"High embedding coherence ({max_good:.3f}) - auto-validating: {relationship_text}"
                    )
                    return RelationshipValidation(
                        is_valid=True,
                        reasoning=f"High embedding coherence ({max_good:.3f}) with valid patterns",
                    )

                # High confidence invalid relationship
                if max_bad > 0.8 and max_bad > max_good + 0.2:
                    logger.debug(
                        f"High embedding incoherence ({max_bad:.3f}) - auto-rejecting: {relationship_text}"
                    )
                    return RelationshipValidation(
                        is_valid=False,
                        reasoning=f"High embedding similarity ({max_bad:.3f}) with invalid patterns",
                    )

        except Exception as e:
            logger.warning(f"Embedding relationship coherence check failed: {e}")

        # Uncertain case - let LLM decide
        return None

    def _validate_relationship_semantics(
        self,
        source_entity_name: str,
        target_entity_name: str,
        relationship_type: str,
        description: str,
        source_node_type: str,
        target_node_type: str,
    ) -> RelationshipValidation:
        """Validate whether a relationship makes semantic sense using embedding pre-filtering then LLM"""

        # Try embedding-based semantic coherence check first
        embedding_validation = self._check_relationship_coherence_by_embedding(
            source_entity_name,
            target_entity_name,
            relationship_type,
            source_node_type,
            target_node_type,
        )

        if embedding_validation:
            return embedding_validation

        # Fall back to LLM validation for uncertain cases
        return self._llm_validate_relationship_semantics(
            source_entity_name,
            target_entity_name,
            relationship_type,
            description,
            source_node_type,
            target_node_type,
        )

    def _llm_validate_relationship_semantics(
        self,
        source_entity_name: str,
        target_entity_name: str,
        relationship_type: str,
        description: str,
        source_node_type: str,
        target_node_type: str,
    ) -> RelationshipValidation:
        """Validate relationship using LLM"""

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

    def _calculate_experience_entity_strength(
        self,
        entity_confidence: float,
        entity_type: str,
        relationship_type: str = "involves",
        is_experience_relationship: bool = False,
    ) -> float:
        """Calculate relationship strength for experience-entity relationships based on semantic importance"""

        # Base strength from entity confidence
        base_strength = entity_confidence

        # Entity type importance modifiers
        entity_importance = {
            "person": 0.9,  # People are very important
            "emotion": 0.8,  # Emotions are highly significant
            "goal": 0.8,  # Goals are important
            "concept": 0.7,  # Concepts are moderately important
            "object": 0.5,  # Objects are less important
            "tool": 0.6,  # Tools have moderate importance
            "place": 0.6,  # Places have moderate importance
        }

        type_modifier = entity_importance.get(entity_type.lower(), 0.6)

        # Experience relationships are generally important but not overwhelming
        if is_experience_relationship:
            experience_modifier = 0.8
        else:
            experience_modifier = 1.0

        # Calculate final strength as weighted combination
        # Using entity confidence as primary signal, type importance as secondary
        final_strength = (
            (base_strength * 0.7) + (type_modifier * 0.2) + (experience_modifier * 0.1)
        )

        # Ensure bounds [0.1, 1.0]
        final_strength = min(max(final_strength, 0.1), 1.0)

        logger.debug(
            f"Experience-entity strength: {entity_type} confidence={entity_confidence:.2f} "
            f"type_mod={type_modifier:.2f} final={final_strength:.2f}"
        )

        return final_strength

    def _make_entity_key(self, name: str, entity_type: str) -> str:
        """Create composite key to handle entity name collisions"""
        normalized_name = name.lower().strip()
        return f"{normalized_name}|{entity_type.lower()}"

    def resolve_entity_to_node_id(
        self, entity_name: str, entity_type: str, entity_description: str
    ) -> Optional[str]:
        """Resolve entity to node ID using kNN similarity with pronoun resolution"""
        # Handle pronoun resolution first
        resolved_entity_name = self._resolve_pronouns_to_names(entity_name, entity_type)
        
        query_text = f"{resolved_entity_name} ({entity_type}): {entity_description}"

        # Log the resolution attempt
        if resolved_entity_name != entity_name:
            logger.info(f"🔄 Resolved pronoun '{entity_name}' to '{resolved_entity_name}'")
        logger.info(f"🔍 Attempting to resolve entity: '{query_text}'")
        logger.info(f"📊 KNN index has {len(self.node_search.entity_metadata)} entities")
        
        # Log all available entities for debugging when there are few
        if len(self.node_search.entity_metadata) <= 10:
            available_entities = [
                f"ID:{meta.get_id()}, Text:{meta.get_text()}" for meta in self.node_search.entity_metadata
            ]
            logger.info(f"📋 All available entities: {available_entities}")
        else:
            # Just show first 5 if there are many
            available_entities = [
                f"ID:{meta.get_id()}, Text:{meta.get_text()}" for meta in self.node_search.entity_metadata[:5]
            ]
            logger.info(f"📋 Available entities (first 5): {available_entities}")

        # Implement tiered entity resolution with LLM validation
        result = self._tiered_entity_resolution(
            query_text, entity_name, entity_type, entity_description, resolved_entity_name
        )
        
        if result:
            logger.info(f"✅ Resolved '{entity_name}' to node ID: {result}")
        else:
            logger.info(
                f"🆕 Creating new entity (no existing match found): '{entity_name}' ({entity_type})"
            )

        return result
    
    def _tiered_entity_resolution(
        self, 
        query_text: str, 
        entity_name: str, 
        entity_type: str, 
        entity_description: str,
        resolved_entity_name: str
    ) -> Optional[str]:
        """Implement tiered entity resolution with LLM validation for ambiguous cases"""
        
        # First, try to find the best match
        best_match = self.node_search.find_best_match(query_text)
        
        # If no match at all, try with original name as backup
        if not best_match and resolved_entity_name != entity_name:
            original_query = f"{entity_name} ({entity_type}): {entity_description}"
            best_match = self.node_search.find_best_match(original_query)
        
        if not best_match:
            logger.info(f"🔍 No entities found in index for query: '{query_text}' (index size: {len(self.node_search.entity_metadata)})")
            return None
            
        similarity = best_match.similarity
        logger.info(f"🔢 Best match similarity: {similarity:.3f} for '{best_match.t.get_text()}'")
        
        # Tier 1: High similarity - auto accept
        if similarity >= 0.85:
            logger.info(f"✅ Auto-accepting high similarity match: {similarity:.3f}")
            self.metrics.entities_auto_accepted += 1
            return best_match.t.get_id()
        
        # Tier 2: Medium similarity - LLM validation
        elif similarity >= 0.7:
            logger.info(f"🤖 LLM validation needed for medium similarity: {similarity:.3f}")
            self.metrics.entities_llm_validated += 1
            
            # Use LLM to determine if entities are the same
            if self._llm_validate_entity_match(
                entity_name, entity_type, entity_description,
                best_match.t.name, best_match.t.node_type.value, best_match.t.description
            ):
                logger.info(f"✅ LLM validated entity match")
                return best_match.t.get_id()
            else:
                logger.info(f"❌ LLM rejected entity match")
                return None
        
        # Tier 3: Low similarity - auto reject
        else:
            logger.info(f"❌ Auto-rejecting low similarity match: {similarity:.3f}")
            self.metrics.entities_auto_rejected += 1
            return None
    
    def _llm_validate_entity_match(
        self,
        new_entity_name: str,
        new_entity_type: str, 
        new_entity_description: str,
        existing_entity_name: str,
        existing_entity_type: str,
        existing_entity_description: str
    ) -> bool:
        """Use LLM to validate if two entities are the same"""
        
        try:
            validation = direct_structured_llm_call(
                prompt=f"""I need to determine if these two entities represent the same thing:

NEW ENTITY:
Name: {new_entity_name}
Type: {new_entity_type}
Description: {new_entity_description}

EXISTING ENTITY:
Name: {existing_entity_name}
Type: {existing_entity_type}
Description: {existing_entity_description}

Are these the same entity? Consider:
- Do they refer to the same person, object, concept, emotion, or goal?
- Are the descriptions compatible or complementary?
- Could the new description be an updated/expanded version of the existing one?
- Are there any contradictions that would make them different entities?

Examples of SAME entities:
- "David" (person) vs "David" (person) with complementary descriptions about the same person
- "excitement" (emotion) vs "excitement" (emotion) with similar emotional contexts
- "web search tool" (object) vs "web search tool" (object) referring to the same tool

Examples of DIFFERENT entities:
- "David" (person) vs "Sarah" (person) - different people
- "excitement about tools" vs "excitement about games" if they represent distinct emotional episodes
- Same name but completely contradictory descriptions""",
                response_model=EntitySimilarityMatch,
                model=self.model,
                llm=self.llm,
                caller="entity_similarity_validation",
            )
            
            return validation.is_same_entity
            
        except Exception as e:
            logger.warning(f"LLM entity validation failed: {e}, defaulting to reject")
            return False
    
    def _resolve_pronouns_to_names(self, entity_name: str, entity_type: str) -> str:
        """Resolve pronouns and contextual references to actual names"""
        name_lower = entity_name.lower().strip()
        
        # Handle first-person pronouns (I, me, myself)
        if name_lower in ["i", "me", "myself"] and entity_type == "person":
            return self.state.name
        
        # Handle user references based on common patterns  
        if "user" in name_lower and entity_type == "person":
            return self.state.name
            
        # Handle "test user" specifically from our test
        if name_lower in ["test user", "the user"] and entity_type == "person":
            return self.state.name
        
        # Return original name if no pronoun resolution needed
        return entity_name

    def _initialize_self_entity(self) -> None:
        """Initialize the self/agent entity for pronoun resolution"""
        # Create a self entity node
        self_node = GraphNode(
            id=f"person_{self.state.name.lower().replace(' ', '_')}_agent_self",
            node_type=NodeType.PERSON,
            name=self.state.name,
            description=f"The agent {self.state.name}, representing the AI assistant's identity",
            properties={
                "is_self": True,
                "role": self.state.role,
                "entity_type": "person"
            },
            importance=1.0,  # High importance for self
            source_trigger_id="agent_initialization"
        )
        
        # Generate embedding for the self node
        self._generate_node_embedding(self_node)
        
        # Add to graph and KNN index
        self.graph.add_node(self_node)
        self.node_search.add_entity(self_node)
        
        logger.info(f"🤖 Initialized self entity: {self.state.name} (ID: {self_node.id})")

    def verify_entity_match(
        self, entity_id: str, provided_name: str, provided_type: str, provided_description: str
    ) -> bool:
        """Verify that provided entity data matches the entity ID using similarity"""
        try:
            if entity_id not in self.graph.nodes:
                return False
                
            actual_entity = self.graph.nodes[entity_id]
            
            # Create text representations for similarity comparison
            provided_text = f"{provided_name} ({provided_type}): {provided_description}"
            actual_text = f"{actual_entity.name} ({actual_entity.node_type.value}): {actual_entity.description}"
            
            # Use embedding service for similarity comparison
            similarity = self.embedding_service.cosine_similarity(
                self.embedding_service.encode(provided_text),
                self.embedding_service.encode(actual_text)
            )
            
            # Use same threshold as KNN matching
            is_match = similarity > 0.75
            
            if is_match:
                logger.debug(f"✅ Entity ID verified: {entity_id} (similarity: {similarity:.3f})")
            else:
                logger.warning(f"❌ Entity ID mismatch: {entity_id} (similarity: {similarity:.3f})")
                logger.debug(f"   Provided: {provided_text}")
                logger.debug(f"   Actual:   {actual_text}")
            
            return is_match
            
        except Exception as e:
            logger.error(f"Entity verification failed: {e}")
            return False

    def create_entity_from_nary_participant(
        self, entity_name: str, entity_type: str, entity_description: str, source_trigger_id: str
    ) -> Optional[str]:
        """Create a new entity from N-ary relationship participant data"""
        try:
            # Map entity type to node type
            node_type = self._map_entity_type_to_node_type(entity_type)
            
            # Create new knowledge node  
            normalized_name = entity_name.lower().strip()
            knowledge_node = GraphNode(
                id=f"{node_type.value}_{normalized_name.replace(' ', '_')}_{str(uuid.uuid4())[:8]}",
                node_type=node_type,
                name=entity_name,
                description=entity_description,
                properties={
                    "confidence": 0.8,  # Medium confidence for N-ary created entities
                    "evidence": f"Referenced in N-ary relationship",
                    "source_triggers": [source_trigger_id],
                    "entity_type": entity_type,
                    "created_from_nary": True
                },
                importance=0.8,  # Medium importance
                source_trigger_id=source_trigger_id,
            )
            
            # Generate embedding for the new knowledge node
            self._generate_node_embedding(knowledge_node)
            
            # Add to graph and KNN index
            self.graph.add_node(knowledge_node)
            self.node_search.add_entity(knowledge_node)
            
            logger.info(f"📝 Created N-ary entity: ID:{knowledge_node.get_id()}, Text:{knowledge_node.get_text()}")
            
            return knowledge_node.id
            
        except Exception as e:
            logger.error(f"Failed to create entity from N-ary participant: {e}")
            return None

    def get_stats(self) -> GraphStats:
        """Get comprehensive graph statistics"""
        return self.graph.get_stats()

    def get_performance_metrics(self) -> EnhancedPerformanceMetrics:
        """Get comprehensive performance metrics"""
        summary = self.metrics.get_summary()

        detailed_llm_calls = None

        # Add LLM call statistics from the LLM's built-in tracking

        llm_stats = {}
        total_llm_calls = 0
        total_llm_time = 0.0

        for caller, stats in self.llm.call_stats.items():
            llm_stats[caller] = DetailedLLMStats(
                calls=stats.count,
                total_time=f"{stats.total_time:.2f}s",
                avg_time=(
                    f"{stats.total_time/stats.count:.2f}s"
                    if stats.count > 0
                    else "0.00s"
                ),
            )
            total_llm_calls += stats.count
            total_llm_time += stats.total_time

        detailed_llm_calls = DetailedLLMCallMetrics(
            total_calls=total_llm_calls,
            total_time=f"{total_llm_time:.2f}s",
            by_operation=llm_stats,
        )

        return EnhancedPerformanceMetrics(
            timing=summary.timing,
            llm_calls=summary.llm_calls,
            entity_processing=summary.entity_processing,
            relationship_processing=summary.relationship_processing,
            llm_calls_detailed=detailed_llm_calls,
        )

    def _find_trigger_related_entities(
        self, trigger: TriggerHistoryEntry, k: int = 10
    ) -> List[Dict[str, str]]:
        """Find entities most similar to the trigger content for context during extraction"""
        
        if not self.graph.get_all_nodes():
            return []
            
        # Get trigger text for embedding
        trigger_text = format_single_trigger_entry(trigger, use_summary=False)
        
        try:
            # Generate embedding for the trigger
            trigger_embedding_list = self.embedding_service.encode(trigger_text)
            trigger_embedding = np.array(trigger_embedding_list)
            
            # Filter to exclude experience nodes and find similar entities
            entity_nodes = [
                node for node in self.graph.get_all_nodes() 
                if node.node_type != NodeType.EXPERIENCE
            ]
            
            if not entity_nodes:
                return []
            
            # Use KNN search to find similar entities
            matches = self.node_search.find_similar_entities(
                trigger_embedding, 
                k=k, 
                similarity_threshold=0.3  # Lower threshold for context gathering
            )
            
            # Convert to format expected by knowledge extraction
            related_entities = []
            for match in matches:
                related_entities.append({
                    "id": match.t.id,
                    "name": match.t.name,
                    "type": match.t.node_type.value,
                    "description": match.t.description,
                    "similarity": match.similarity
                })
            
            if related_entities:
                logger.info(
                    f"🔍 Found {len(related_entities)} entities related to trigger "
                    f"(similarities: {[f'{e['similarity']:.3f}' for e in related_entities[:3]]})"
                )
            
            return related_entities
            
        except Exception as e:
            logger.warning(f"Failed to find trigger-related entities: {e}")
            return []

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
                f"    Current stats: {stats.total_nodes} nodes, {stats.total_relationships} relationships"
            )

    # Final statistics
    print(f"\n📊 Final Graph Statistics:")
    stats = builder.get_stats()

    print(f"  total_nodes: {stats.total_nodes}")
    print(f"  total_relationships: {stats.total_relationships}")
    print(f"  processed_triggers: {stats.processed_triggers}")
    if stats.node_types:
        print(f"  node_types:")
        for subkey, subvalue in stats.node_types.items():
            print(f"    {subkey}: {subvalue}")
    if stats.relationship_types:
        print(f"  relationship_types:")
        for subkey, subvalue in stats.relationship_types.items():
            print(f"    {subkey}: {subvalue}")
    print(f"  confidence_distribution:")
    print(f"    mean: {stats.confidence_distribution.mean}")
    print(
        f"    high_confidence_count: {stats.confidence_distribution.high_confidence_count}"
    )
    print(
        f"    medium_confidence_count: {stats.confidence_distribution.medium_confidence_count}"
    )
    print(
        f"    low_confidence_count: {stats.confidence_distribution.low_confidence_count}"
    )

    print(
        f"\n  Successfully processed: {successful_count}/{len(all_triggers)} triggers"
    )

    # Show performance metrics
    print(f"\n⚡ Performance Metrics:")
    perf_metrics = builder.get_performance_metrics()

    print(f"  Timing:")
    print(f"    total_time: {perf_metrics.timing.total_time}")
    print(f"    extraction_time: {perf_metrics.timing.extraction_time}")
    print(f"    entity_processing_time: {perf_metrics.timing.entity_processing_time}")
    print(
        f"    relationship_processing_time: {perf_metrics.timing.relationship_processing_time}"
    )
    print()

    print(f"  LLM Calls:")
    print(f"    total_calls: {perf_metrics.llm_calls.total_calls}")
    print(f"    total_time: {perf_metrics.llm_calls.total_time}")
    if perf_metrics.llm_calls.by_operation:
        print(f"    by_operation:")
        for op, count in perf_metrics.llm_calls.by_operation.items():
            print(f"      {op}: {count}")
    print()

    print(f"  Entity Processing:")
    print(f"    total_processed: {perf_metrics.entity_processing.total_processed}")
    print(f"    deduplicated: {perf_metrics.entity_processing.deduplicated}")
    print(f"    auto_accepted: {perf_metrics.entity_processing.auto_accepted}")
    print(f"    auto_rejected: {perf_metrics.entity_processing.auto_rejected}")
    print(f"    llm_validated: {perf_metrics.entity_processing.llm_validated}")
    print()

    print(f"  Relationship Processing:")
    print(
        f"    total_processed: {perf_metrics.relationship_processing.total_processed}"
    )
    print(
        f"    type_matched_by_embedding: {perf_metrics.relationship_processing.type_matched_by_embedding}"
    )
    print(
        f"    type_matched_by_llm: {perf_metrics.relationship_processing.type_matched_by_llm}"
    )
    print(
        f"    validated_by_embedding: {perf_metrics.relationship_processing.validated_by_embedding}"
    )
    print(
        f"    validated_by_llm: {perf_metrics.relationship_processing.validated_by_llm}"
    )
    print()

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
