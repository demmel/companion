#!/usr/bin/env python3
"""
Comprehensive Graph Maintenance System

Performs advanced semantic analysis and refinement of knowledge graph segments,
including missing intermediate relationship detection, contradiction resolution,
and semantic refinement.
"""

import logging
from typing import Dict, List, Set, Any, Tuple, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from collections import defaultdict

from agent.experiments.knowledge_graph.knowledge_graph_prototype import (
    KnowledgeExperienceGraph,
    GraphNode,
    GraphRelationship,
    NodeType,
)
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.memory.embedding_service import get_embedding_service
from agent.state import State

logger = logging.getLogger(__name__)


class LogicalLeap(BaseModel):
    """Represents a detected logical leap in relationships"""

    source_relationship_id: str
    source_description: str
    missing_intermediate_steps: List[str] = Field(
        description="List of missing intermediate relationship descriptions"
    )
    confidence: float = Field(
        description="Confidence in this being a logical leap (0-1)"
    )
    reasoning: str


class Contradiction(BaseModel):
    """Represents a detected contradiction in the graph"""

    relationship_ids: List[str] = Field(description="IDs of conflicting relationships")
    contradiction_type: str = Field(
        description="Type of contradiction: temporal, logical, or semantic"
    )
    description: str
    resolution_suggestion: str
    confidence: float = Field(
        description="Confidence in this being a contradiction (0-1)"
    )


class SemanticRefinement(BaseModel):
    """Represents a suggested semantic refinement"""

    target_ids: List[str] = Field(description="IDs of nodes/relationships to refine")
    refinement_type: str = Field(
        description="Type: consolidate, split, upgrade, or clarify"
    )
    current_representation: str
    suggested_representation: str
    reasoning: str
    confidence: float


class MaintenanceReport(BaseModel):
    """Complete maintenance analysis report"""

    logical_leaps: List[LogicalLeap]
    contradictions: List[Contradiction]
    semantic_refinements: List[SemanticRefinement]
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    nodes_analyzed: int
    relationships_analyzed: int


class GraphMaintenanceSystem:
    """
    Comprehensive graph maintenance system for semantic analysis and refinement
    """

    def __init__(self, llm: LLM, model: SupportedModel, state: State):
        self.llm = llm
        self.model = model
        self.state = state
        self.embedding_service = get_embedding_service()

    def analyze_graph(self, graph: KnowledgeExperienceGraph) -> MaintenanceReport:
        """Perform comprehensive analysis of the graph"""

        logger.info("Starting comprehensive graph maintenance analysis...")

        all_nodes = graph.get_all_nodes()
        all_relationships = graph.get_all_relationships()

        # Filter out experience nodes for analysis (focus on knowledge relationships)
        knowledge_nodes = [n for n in all_nodes if n.node_type != NodeType.EXPERIENCE]
        knowledge_relationships = [
            r
            for r in all_relationships
            if graph.nodes.get(
                r.source_node_id, GraphNode("", NodeType.EXPERIENCE, "", "")
            ).node_type
            != NodeType.EXPERIENCE
            and graph.nodes.get(
                r.target_node_id, GraphNode("", NodeType.EXPERIENCE, "", "")
            ).node_type
            != NodeType.EXPERIENCE
        ]

        logger.info(
            f"Analyzing {len(knowledge_nodes)} knowledge nodes and {len(knowledge_relationships)} knowledge relationships"
        )

        # Detect logical leaps
        logical_leaps = self._detect_logical_leaps(graph, knowledge_relationships)
        logger.info(f"Found {len(logical_leaps)} logical leaps")

        # Detect contradictions
        contradictions = self._detect_contradictions(
            graph, knowledge_nodes, knowledge_relationships
        )
        logger.info(f"Found {len(contradictions)} contradictions")

        # Find semantic refinements
        semantic_refinements = self._find_semantic_refinements(
            graph, knowledge_nodes, knowledge_relationships
        )
        logger.info(f"Found {len(semantic_refinements)} semantic refinements")

        return MaintenanceReport(
            logical_leaps=logical_leaps,
            contradictions=contradictions,
            semantic_refinements=semantic_refinements,
            nodes_analyzed=len(knowledge_nodes),
            relationships_analyzed=len(knowledge_relationships),
        )

    def _detect_logical_leaps(
        self, graph: KnowledgeExperienceGraph, relationships: List[GraphRelationship]
    ) -> List[LogicalLeap]:
        """Detect relationships that skip intermediate logical steps"""

        logical_leaps = []

        # Focus on causal relationships which are most prone to logical leaps
        causal_types = ["causes", "caused", "enables", "results_in", "leads_to"]
        causal_relationships = [
            r
            for r in relationships
            if any(
                causal_type in r.relationship_type.lower()
                for causal_type in causal_types
            )
        ]

        for rel in causal_relationships:
            source_node = graph.nodes.get(rel.source_node_id)
            target_node = graph.nodes.get(rel.target_node_id)

            if not source_node or not target_node:
                continue

            # Check if this might be a logical leap using LLM
            leap_analysis = self._analyze_potential_logical_leap(
                source_node.name,
                target_node.name,
                rel.relationship_type,
                source_node.node_type.value,
                target_node.node_type.value,
            )

            if leap_analysis and leap_analysis.confidence > 0.7:
                logical_leaps.append(
                    LogicalLeap(
                        source_relationship_id=rel.id,
                        source_description=f"{source_node.name} --[{rel.relationship_type}]--> {target_node.name}",
                        missing_intermediate_steps=leap_analysis.missing_intermediate_steps,
                        confidence=leap_analysis.confidence,
                        reasoning=leap_analysis.reasoning,
                    )
                )

        return logical_leaps

    def _analyze_potential_logical_leap(
        self,
        source_name: str,
        target_name: str,
        relationship_type: str,
        source_type: str,
        target_type: str,
    ) -> Optional[LogicalLeap]:
        """Use LLM to analyze if a relationship represents a logical leap"""

        prompt = f"""I'm analyzing a relationship in a knowledge graph to detect logical leaps.

RELATIONSHIP TO ANALYZE:
{source_name} --[{relationship_type}]--> {target_name}

ENTITY TYPES:
- Source: "{source_name}" (type: {source_type})
- Target: "{target_name}" (type: {target_type})

LOGICAL LEAP DETECTION:
A logical leap occurs when a direct relationship skips important intermediate steps in reasoning.

EXAMPLES OF LOGICAL LEAPS:
- "neon pink minidress --[causes]--> David smiling" 
  Missing: minidress → attractiveness → positive reaction → smiling
  
- "anxiety --[creates]--> solutions"
  Missing: anxiety → problem awareness → motivation → solution seeking → solutions

ANALYSIS TASK:
1. Does this relationship skip intermediate logical steps?
2. If yes, what are the missing intermediate steps?
3. What's your confidence this is a logical leap (0-1)?

Focus on causal relationships that jump too many conceptual levels."""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=LogicalLeap,
                model=self.model,
                llm=self.llm,
                caller="logical_leap_analysis",
                temperature=0.2,
            )
            return result
        except Exception as e:
            logger.warning(
                f"Logical leap analysis failed for {source_name} -> {target_name}: {e}"
            )
            return None

    def _detect_contradictions(
        self,
        graph: KnowledgeExperienceGraph,
        nodes: List[GraphNode],
        relationships: List[GraphRelationship],
    ) -> List[Contradiction]:
        """Detect contradictory relationships or states in the graph"""

        contradictions = []

        # Group relationships by entities to find conflicts
        entity_relationships = defaultdict(list)
        for rel in relationships:
            entity_relationships[rel.source_node_id].append(rel)
            entity_relationships[rel.target_node_id].append(rel)

        # Look for contradictory relationships involving same entities
        for node_id, node_rels in entity_relationships.items():
            if len(node_rels) < 2:
                continue

            node = graph.nodes.get(node_id)
            if not node:
                continue

            # Check pairs of relationships for contradictions
            for i, rel1 in enumerate(node_rels):
                for rel2 in node_rels[i + 1 :]:
                    contradiction = self._analyze_relationship_contradiction(
                        graph, rel1, rel2
                    )
                    if contradiction and contradiction.confidence > 0.6:
                        contradictions.append(contradiction)

        return contradictions

    def _analyze_relationship_contradiction(
        self,
        graph: KnowledgeExperienceGraph,
        rel1: GraphRelationship,
        rel2: GraphRelationship,
    ) -> Optional[Contradiction]:
        """Analyze two relationships for potential contradictions"""

        # Get the nodes involved
        rel1_source = graph.nodes.get(rel1.source_node_id)
        rel1_target = graph.nodes.get(rel1.target_node_id)
        rel2_source = graph.nodes.get(rel2.source_node_id)
        rel2_target = graph.nodes.get(rel2.target_node_id)

        if not all([rel1_source, rel1_target, rel2_source, rel2_target]):
            return None

        # Safe access to node names with null checks
        rel1_desc = f"{rel1_source.name if rel1_source else 'Unknown'} --[{rel1.relationship_type}]--> {rel1_target.name if rel1_target else 'Unknown'}"
        rel2_desc = f"{rel2_source.name if rel2_source else 'Unknown'} --[{rel2.relationship_type}]--> {rel2_target.name if rel2_target else 'Unknown'}"

        prompt = f"""I'm analyzing two relationships for potential contradictions.

RELATIONSHIP 1: {rel1_desc}
RELATIONSHIP 2: {rel2_desc}

CONTRADICTION TYPES:
1. TEMPORAL: Same entity causing conflicting states at same time
2. LOGICAL: Relationships that cannot both be true
3. SEMANTIC: Relationships with conflicting meanings

EXAMPLES:
- "David makes me feel excited" + "David makes me feel anxious" = Temporal (context-dependent)
- "David owns the apartment" + "Someone else owns the apartment" = Logical contradiction
- "Tool is physical object" + "Tool is abstract concept" = Semantic contradiction

ANALYSIS:
Do these relationships contradict each other? If yes:
- What type of contradiction is it?
- How confident are you (0-1)?
- What's a suggested resolution?"""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=Contradiction,
                model=self.model,
                llm=self.llm,
                caller="contradiction_analysis",
                temperature=0.2,
            )

            # Set the relationship IDs
            result.relationship_ids = [rel1.id, rel2.id]
            result.description = f"Conflict between: {rel1_desc} AND {rel2_desc}"

            return result
        except Exception as e:
            logger.warning(f"Contradiction analysis failed: {e}")
            return None

    def _find_semantic_refinements(
        self,
        graph: KnowledgeExperienceGraph,
        nodes: List[GraphNode],
        relationships: List[GraphRelationship],
    ) -> List[SemanticRefinement]:
        """Find opportunities for semantic refinement of concepts and relationships"""

        refinements = []

        # Look for nodes that might need consolidation (similar concepts)
        consolidation_candidates = self._find_consolidation_candidates(nodes)
        refinements.extend(consolidation_candidates)

        # Look for vague relationships that could be more specific
        relationship_refinements = self._find_relationship_refinements(
            graph, relationships
        )
        refinements.extend(relationship_refinements)

        return refinements

    def _find_consolidation_candidates(
        self, nodes: List[GraphNode]
    ) -> List[SemanticRefinement]:
        """Find nodes that represent similar concepts and could be consolidated"""

        refinements = []

        # Group nodes by type for comparison
        nodes_by_type = defaultdict(list)
        for node in nodes:
            nodes_by_type[node.node_type].append(node)

        # Check each type group for similar concepts
        for node_type, type_nodes in nodes_by_type.items():
            if len(type_nodes) < 2:
                continue

            # Use embedding similarity to find potential duplicates
            try:
                for i, node1 in enumerate(type_nodes):
                    for node2 in type_nodes[i + 1 :]:
                        if not node1.embedding or not node2.embedding:
                            continue

                        similarity = self.embedding_service.cosine_similarity(
                            node1.embedding, node2.embedding
                        )

                        # High similarity suggests potential consolidation
                        if similarity > 0.8:
                            refinement = SemanticRefinement(
                                target_ids=[node1.id, node2.id],
                                refinement_type="consolidate",
                                current_representation=f"Separate: '{node1.name}' and '{node2.name}'",
                                suggested_representation=f"Consolidated concept combining both",
                                reasoning=f"High semantic similarity ({similarity:.3f}) suggests overlapping concepts",
                                confidence=min(similarity, 0.9),
                            )
                            refinements.append(refinement)

            except Exception as e:
                logger.warning(f"Consolidation analysis failed for {node_type}: {e}")

        return refinements

    def _find_relationship_refinements(
        self, graph: KnowledgeExperienceGraph, relationships: List[GraphRelationship]
    ) -> List[SemanticRefinement]:
        """Find relationships that could be made more specific or meaningful"""

        refinements = []

        # Look for vague relationship types
        vague_types = ["reminds_me_of", "relates_to", "involves", "connected_to"]

        for rel in relationships:
            if rel.relationship_type.lower() in vague_types:
                source_node = graph.nodes.get(rel.source_node_id)
                target_node = graph.nodes.get(rel.target_node_id)

                if source_node and target_node:
                    specific_suggestion = self._suggest_specific_relationship(
                        source_node.name,
                        target_node.name,
                        rel.relationship_type,
                        source_node.node_type.value,
                        target_node.node_type.value,
                    )

                    if specific_suggestion:
                        refinement = SemanticRefinement(
                            target_ids=[rel.id],
                            refinement_type="upgrade",
                            current_representation=f"{source_node.name} --[{rel.relationship_type}]--> {target_node.name}",
                            suggested_representation=f"{source_node.name} --[{specific_suggestion}]--> {target_node.name}",
                            reasoning=f"Replace vague '{rel.relationship_type}' with more specific relationship",
                            confidence=0.7,
                        )
                        refinements.append(refinement)

        return refinements

    def _suggest_specific_relationship(
        self,
        source_name: str,
        target_name: str,
        current_type: str,
        source_type: str,
        target_type: str,
    ) -> Optional[str]:
        """Suggest a more specific relationship type"""

        # Simple heuristic-based suggestions (could be enhanced with LLM)
        type_mapping = {
            ("person", "emotion"): "feels",
            ("object", "person"): "belongs_to",
            ("concept", "concept"): "relates_to",
            ("person", "object"): "uses",
            ("emotion", "person"): "affects",
        }

        key = (source_type, target_type)
        return type_mapping.get(key)


if __name__ == "__main__":
    # Test the maintenance system
    logging.basicConfig(level=logging.INFO)

    from agent.llm import create_llm, SupportedModel
    from agent.conversation_persistence import ConversationPersistence

    # Load state for testing
    persistence = ConversationPersistence()
    _, state, _ = persistence.load_conversation("baseline")

    if state is None:
        print("❌ Could not load baseline state")
        exit(1)

    # Create maintenance system
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    maintenance = GraphMaintenanceSystem(llm, model, state)

    print("✅ Graph maintenance system test completed!")
