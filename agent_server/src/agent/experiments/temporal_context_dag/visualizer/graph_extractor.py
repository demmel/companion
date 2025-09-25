"""
Graph state extractor for DAG memory visualization.

Extracts and formats memory graph and context data for web visualization,
including visual styling information based on memory types, confidence levels,
and context status.
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from ..models import MemoryGraph, ContextGraph, MemoryElement, MemoryEdge
from ..memory_types import MemoryType
from ..edge_types import GraphEdgeType
from .action_processor import GraphState

logger = logging.getLogger(__name__)


@dataclass
class NodeData:
    """Visualization data for a memory node."""

    id: str
    content: str
    content_preview: str  # Truncated version for display
    evidence: str
    memory_type: str
    confidence_level: str
    emotional_significance: float
    timestamp: str
    in_context: bool
    tokens: Optional[int] = None  # Only if in context

    # Visual styling
    color: str = ""
    outline_style: str = ""
    pattern: str = ""

    # Change highlighting
    is_new: bool = False
    is_modified: bool = False


@dataclass
class EdgeData:
    """Visualization data for a memory edge."""

    id: str
    source: str
    target: str
    edge_type: str
    created_at: str
    in_context: bool = False

    # Visual styling
    color: str = ""
    line_style: str = ""
    width: int = 2

    # Change highlighting
    is_new: bool = False


@dataclass
class GraphVisualizationData:
    """Complete visualization data for a graph state."""

    step_index: int
    action_type: str
    action_description: str
    timestamp: str

    nodes: List[NodeData]
    edges: List[EdgeData]

    # Summary statistics
    total_memories: int
    context_memories: int
    total_edges: int
    context_edges: int
    total_tokens: int


class GraphExtractor:
    """Extracts visualization data from memory graph states."""

    # Color scheme for memory types
    MEMORY_TYPE_COLORS = {
        MemoryType.COMMITMENT.value: "#FF4444",  # Red
        MemoryType.IDENTITY.value: "#8844FF",  # Purple
        MemoryType.EMOTIONAL.value: "#FF88CC",  # Pink
        MemoryType.PREFERENCE.value: "#FF8844",  # Orange
        MemoryType.FACTUAL.value: "#4488FF",  # Blue
        MemoryType.PROCEDURAL.value: "#44FF88",  # Green
    }

    # Edge type styling
    EDGE_TYPE_STYLES = {
        GraphEdgeType.FOLLOWED_BY.value: {"color": "#666666", "style": "solid"},
        GraphEdgeType.CAUSED.value: {"color": "#4488FF", "style": "solid"},
        GraphEdgeType.EXPLAINS.value: {"color": "#22AA22", "style": "solid"},
        GraphEdgeType.EXPLAINED_BY.value: {"color": "#66BB66", "style": "solid"},
        GraphEdgeType.CONTRADICTED_BY.value: {"color": "#FF2222", "style": "dashed"},
        GraphEdgeType.RETRACTED_BY.value: {"color": "#CC4444", "style": "dashed"},
        GraphEdgeType.SUPERSEDED_BY.value: {"color": "#FF6666", "style": "dashed"},
        GraphEdgeType.CLARIFIED_BY.value: {"color": "#FF8844", "style": "dotted"},
    }

    def extract_visualization_data(self, state: GraphState) -> GraphVisualizationData:
        """Extract complete visualization data from a graph state."""

        # Get context memory IDs for quick lookup
        context_memory_ids = {elem.memory.id for elem in state.context_graph.elements}
        context_edge_ids = {edge.id for edge in state.context_graph.edges}

        # Create token lookup for context memories
        context_tokens = {}
        for elem in state.context_graph.elements:
            context_tokens[elem.memory.id] = elem.tokens

        # Extract node data
        nodes = []
        for memory_id, memory in state.memory_graph.elements.items():
            node_data = self._extract_node_data(
                memory,
                in_context=memory_id in context_memory_ids,
                tokens=context_tokens.get(memory_id),
                is_new=memory_id in (state.added_memories or set()),
                is_modified=memory_id in (state.modified_memories or set()),
            )
            nodes.append(node_data)

        # Extract edge data
        edges = []
        for edge_id, edge in state.memory_graph.edges.items():
            edge_data = self._extract_edge_data(
                edge,
                in_context=edge_id in context_edge_ids,
                is_new=edge_id in (state.added_edges or set()),
            )
            edges.append(edge_data)

        # Calculate statistics
        total_tokens = sum(elem.tokens for elem in state.context_graph.elements)

        return GraphVisualizationData(
            step_index=state.step_index,
            action_type=state.action.action_type,
            action_description=self._generate_action_description(state.action),
            timestamp=state.timestamp.isoformat(),
            nodes=nodes,
            edges=edges,
            total_memories=len(state.memory_graph.elements),
            context_memories=len(state.context_graph.elements),
            total_edges=len(state.memory_graph.edges),
            context_edges=len(state.context_graph.edges),
            total_tokens=total_tokens,
        )

    def _extract_node_data(
        self,
        memory: MemoryElement,
        in_context: bool,
        tokens: Optional[int] = None,
        is_new: bool = False,
        is_modified: bool = False,
    ) -> NodeData:
        """Extract visualization data for a single memory node."""

        # Create content preview
        content_preview = memory.content
        if len(content_preview) > 60:
            content_preview = content_preview[:57] + "..."

        # Get styling
        color = self.MEMORY_TYPE_COLORS.get(memory.memory_type.value, "#CCCCCC")
        outline_style = "thick-solid" if in_context else "thin-dashed"
        pattern = self._get_confidence_pattern(memory.confidence_level.value)

        return NodeData(
            id=memory.id,
            content=memory.content,
            content_preview=content_preview,
            evidence=memory.evidence,
            memory_type=memory.memory_type.value,
            confidence_level=memory.confidence_level.value,
            emotional_significance=memory.emotional_significance,
            timestamp=memory.timestamp.isoformat(),
            in_context=in_context,
            tokens=tokens,
            color=color,
            outline_style=outline_style,
            pattern=pattern,
            is_new=is_new,
            is_modified=is_modified,
        )

    def _extract_edge_data(
        self, edge: MemoryEdge, in_context: bool = False, is_new: bool = False
    ) -> EdgeData:
        """Extract visualization data for a single memory edge."""

        # Get styling
        style_info = self.EDGE_TYPE_STYLES.get(
            edge.edge_type.value, {"color": "#999999", "style": "solid"}
        )

        # Adjust width for context edges
        width = 3 if in_context else 2

        return EdgeData(
            id=edge.id,
            source=edge.source_id,
            target=edge.target_id,
            edge_type=edge.edge_type.value,
            created_at=edge.created_at.isoformat(),
            in_context=in_context,
            color=style_info["color"],
            line_style=style_info["style"],
            width=width,
            is_new=is_new,
        )

    def _get_confidence_pattern(self, confidence_level: str) -> str:
        """Get pattern style for confidence level."""
        match confidence_level:
            case "user_confirmed":
                return "solid"
            case "strong_inference":
                return "diagonal-stripes"
            case "reasonable_assumption":
                return "dots"
            case "speculative":
                return "transparent"
            case "likely_error":
                return "error-outline"
            case "known_false":
                return "false-overlay"
            case _:
                return "solid"

    def _generate_action_description(self, action) -> str:
        """Generate human-readable description of an action."""
        match action.action_type:
            case "add_memory":
                preview = (
                    action.memory.content[:30] + "..."
                    if len(action.memory.content) > 30
                    else action.memory.content
                )
                return f"Added {action.memory.memory_type.value} memory: {preview}"
            case "add_connection":
                return f"Added {action.edge.edge_type.value} connection"
            case "add_to_context":
                return f"Added to context ({action.initial_tokens} tokens)"
            case "add_edge_to_context":
                return f"Added edge to context"
            case "remove_from_context":
                return f"Removed {len(action.memory_ids)} from context: {action.reason}"
            case "update_confidence":
                return f"Updated confidence to {action.new_confidence.value}: {action.reason}"
            case "add_container":
                return f"Added container with {len(action.element_ids)} elements"
            case "apply_token_decay":
                return f"Applied token decay (-{action.decay_amount})"
            case "checkpoint":
                return f"Checkpoint: {action.description}"
            case _:
                return f"Action: {action.action_type}"

    def get_legend_data(self) -> Dict:
        """Get legend data for the visualization."""
        return {
            "memory_types": [
                {
                    "name": "Commitment",
                    "color": self.MEMORY_TYPE_COLORS[MemoryType.COMMITMENT.value],
                },
                {
                    "name": "Identity",
                    "color": self.MEMORY_TYPE_COLORS[MemoryType.IDENTITY.value],
                },
                {
                    "name": "Emotional",
                    "color": self.MEMORY_TYPE_COLORS[MemoryType.EMOTIONAL.value],
                },
                {
                    "name": "Preference",
                    "color": self.MEMORY_TYPE_COLORS[MemoryType.PREFERENCE.value],
                },
                {
                    "name": "Factual",
                    "color": self.MEMORY_TYPE_COLORS[MemoryType.FACTUAL.value],
                },
                {
                    "name": "Procedural",
                    "color": self.MEMORY_TYPE_COLORS[MemoryType.PROCEDURAL.value],
                },
            ],
            "edge_types": [
                {"name": "Temporal", "color": "#666666", "style": "solid"},
                {"name": "Causal", "color": "#4488FF", "style": "solid"},
                {"name": "Explanatory", "color": "#44AA44", "style": "solid"},
                {"name": "Correction", "color": "#FF4444", "style": "dashed"},
                {"name": "Clarification", "color": "#FF8844", "style": "dotted"},
            ],
            "context_status": [
                {"name": "In Context", "outline": "thick-solid"},
                {"name": "Out of Context", "outline": "thin-dashed"},
            ],
            "confidence_levels": [
                {"name": "User Confirmed", "pattern": "solid"},
                {"name": "Strong Inference", "pattern": "diagonal-stripes"},
                {"name": "Reasonable Assumption", "pattern": "dots"},
                {"name": "Speculative", "pattern": "transparent"},
                {"name": "Likely Error", "pattern": "error-outline"},
                {"name": "Known False", "pattern": "false-overlay"},
            ],
        }
