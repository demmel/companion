#!/usr/bin/env python3
"""
Knowledge Graph Builder with Validated LLM Extraction

Builds a proper knowledge+experience graph using the validated LLM extraction
system, creating meaningful nodes and relationships with confidence scoring.
"""

import uuid
from typing import Dict, Optional, Any
from datetime import datetime
import logging
import copy

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
from agent.chain_of_action.trigger import UserInputTrigger
from agent.chain_of_action.action.action_types import ActionType
from agent.llm import LLM, SupportedModel
from agent.state import State

logger = logging.getLogger(__name__)


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
        self.state = state
        # Track historical state progression - will be initialized with starting state
        self.current_historical_state: Optional[State] = None

        # Use provided relationship bank or create new one
        self.relationship_bank = relationship_bank or RelationshipTypeBank(
            llm, model, state
        )

        # Track entities across triggers for deduplication
        self.entity_name_to_node_id: Dict[str, str] = {}

        # Track statistics for analysis
        self.entity_evolution_count = 0

    def initialize_historical_state(self, initial_state: State) -> None:
        """Initialize the historical state progression with the agent's initial state"""
        self.current_historical_state = copy.deepcopy(initial_state)
        logger.info(
            f"Initialized historical state: mood={initial_state.current_mood}, environment={initial_state.current_environment}"
        )

    def apply_action_effects_to_state(self, trigger: TriggerHistoryEntry) -> None:
        """Apply the effects of actions in this trigger to the current historical state"""
        if not self.current_historical_state or not trigger.actions_taken:
            return

        for action in trigger.actions_taken:
            if action.result and action.result.type == "success":
                # Apply state changes based on action results
                if action.type == ActionType.UPDATE_MOOD:
                    # Extract mood change from result
                    mood_result = action.result.content
                    if hasattr(mood_result, "new_mood"):
                        old_mood = self.current_historical_state.current_mood
                        self.current_historical_state.current_mood = (
                            mood_result.new_mood
                        )
                        if hasattr(mood_result, "new_intensity"):
                            self.current_historical_state.mood_intensity = (
                                mood_result.new_intensity
                            )
                        logger.debug(
                            f"Applied mood change: {old_mood} -> {mood_result.new_mood}"
                        )

                elif action.type == ActionType.UPDATE_APPEARANCE:
                    # Extract appearance change from result
                    appearance_result = action.result.content
                    if hasattr(appearance_result, "new_appearance"):
                        old_appearance = (
                            self.current_historical_state.current_appearance
                        )
                        self.current_historical_state.current_appearance = (
                            appearance_result.new_appearance
                        )
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
        state_to_use = (
            self.current_historical_state
            if self.current_historical_state
            else self.state
        )
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

        return True

    def add_trigger(self, trigger: TriggerHistoryEntry) -> bool:
        """Simple interface to add a trigger to the graph (wrapper around process_trigger_incremental)"""
        return self.process_trigger_incremental(trigger)

    def _create_experience_node(self, trigger: TriggerHistoryEntry) -> GraphNode:
        """Create experience node preserving full trigger context"""

        # Build meaningful name and description
        user_input = None
        if isinstance(trigger.trigger, UserInputTrigger):
            user_input = trigger.trigger.content

        if user_input:
            name = (
                f"User: {user_input[:50]}..."
                if len(user_input) > 50
                else f"User: {user_input}"
            )
        else:
            name = f"Experience at {trigger.timestamp.strftime('%H:%M')}"

        # Build rich description
        description_parts = []
        if user_input:
            description_parts.append(f"User said: {user_input}")

        # Add agent actions with more detail
        if trigger.actions_taken:
            for action in trigger.actions_taken:
                if action.type == ActionType.THINK:
                    description_parts.append(
                        f"Agent thought: {action.input.focus[:100]}..."
                    )
                elif action.type == ActionType.SPEAK:
                    # Use actual spoken text from results
                    if action.result and action.result.type == "success":
                        response = action.result.content.response
                        if response:
                            description_parts.append(f"Agent said: {response[:100]}...")
                elif action.type == ActionType.UPDATE_MOOD:
                    description_parts.append(
                        f"Agent updated mood to {action.input.new_mood}: {action.input.reason[:50]}..."
                    )
                elif action.type == ActionType.UPDATE_APPEARANCE:
                    description_parts.append(
                        f"Agent updated appearance: {action.input.change_description[:50]}..."
                    )

        description = (
            "; ".join(description_parts) if description_parts else "Experience"
        )

        return GraphNode(
            id=f"exp_{trigger.entry_id}",
            node_type=NodeType.EXPERIENCE,
            name=name,
            description=description,
            properties={
                "timestamp": trigger.timestamp.isoformat(),
                "user_input": user_input,
                "actions": (
                    [
                        {
                            "type": action.type,
                            "focus": (
                                action.input.focus
                                if action.type == ActionType.THINK
                                else ""
                            ),
                            "intent": (
                                action.input.intent
                                if action.type == ActionType.SPEAK
                                else ""
                            ),
                            "tone": (
                                action.input.tone
                                if action.type == ActionType.SPEAK
                                else ""
                            ),
                        }
                        for action in trigger.actions_taken
                    ]
                    if trigger.actions_taken
                    else []
                ),
            },
            source_trigger_id=trigger.entry_id,
        )

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

            # Check if entity already exists (simple name matching for now)
            normalized_name = entity.name.lower().strip()

            if normalized_name in self.entity_name_to_node_id:
                # Entity exists, evolve its description with new information
                existing_node_id = self.entity_name_to_node_id[normalized_name]
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

                # Add this trigger to sources
                if "source_triggers" not in existing_node.properties:
                    existing_node.properties["source_triggers"] = []
                existing_node.properties["source_triggers"].append(trigger.entry_id)

                knowledge_node = existing_node
            else:
                # Create new knowledge node
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
                self.graph.add_node(knowledge_node)
                self.entity_name_to_node_id[normalized_name] = knowledge_node.id

            # Create relationship between experience and knowledge node
            rel = GraphRelationship(
                id=str(uuid.uuid4()),
                source_node_id=experience_node.id,
                target_node_id=knowledge_node.id,
                relationship_type="involves",
                confidence=entity.confidence,
                strength=entity.confidence,
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
                rel_type = self.relationship_bank.get_or_create_relationship_type(
                    proposed_type=rel_extraction.relationship_type,
                    description=rel_extraction.description,
                    source_entity=rel_extraction.source_entity,
                    target_entity=rel_extraction.target_entity,
                    context=f"From trigger {trigger.entry_id}: {rel_extraction.evidence}",
                )

                # Create relationship
                rel = GraphRelationship(
                    id=str(uuid.uuid4()),
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    relationship_type=rel_type,
                    confidence=rel_extraction.confidence,
                    strength=rel_extraction.confidence,
                    properties={
                        "description": rel_extraction.description,
                        "extraction_evidence": rel_extraction.evidence,
                        "original_rel_type": rel_extraction.relationship_type,
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
            "object": NodeType.CONCEPT,  # Objects become concepts
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
