"""
LLM-driven system for deciding connections between memory elements.
"""

from typing import List
import logging

from agent.memory.edge_types import (
    AgentControlledEdgeType,
    GraphEdgeType,
)


from .models import (
    MemoryGraph,
    MemoryEdge,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


def update_confidence_for_correction_edge(graph: MemoryGraph, edge: MemoryEdge) -> None:
    """
    Update confidence level of source memory when correction edge is created.

    Args:
        graph: Memory graph containing the memories
        edge: Correction edge that was just created
    """
    source_memory = graph.elements.get(edge.source_id)
    if source_memory:
        if edge.edge_type == GraphEdgeType.CONTRADICTED_BY:
            source_memory.confidence_level = ConfidenceLevel.LIKELY_ERROR
            logger.info(
                f"  Updated confidence of memory {edge.source_id[:8]} to LIKELY_ERROR due to contradiction"
            )
        elif edge.edge_type == GraphEdgeType.RETRACTED_BY:
            source_memory.confidence_level = ConfidenceLevel.KNOWN_FALSE
            logger.info(
                f"  Updated confidence of memory {edge.source_id[:8]} to KNOWN_FALSE due to retraction"
            )


def add_connections_to_graph(
    graph: MemoryGraph, connections: List[MemoryEdge]
) -> tuple[MemoryGraph, List[MemoryEdge]]:
    """
    Add agent-decided connections to the memory graph.

    Validates edge constraints before adding.

    Returns:
        Tuple of (updated_graph, successfully_added_edges)
    """
    added_count = 0
    successfully_added = []
    agent_controlled_edge_types = {e.value for e in AgentControlledEdgeType}
    for i, connection in enumerate(connections):
        logger.info(
            f"  Processing connection {i+1}: {connection.edge_type} from {connection.source_id[:8]} to {connection.target_id[:8]}"
        )

        # Validate that both memories exist in graph
        if (
            connection.source_id not in graph.elements
            or connection.target_id not in graph.elements
        ):
            logger.warning(f"  Skipping connection {i+1}: memory not found in graph")
            continue

        if connection.edge_type.value not in agent_controlled_edge_types:
            logger.warning(
                f"  Skipping connection {i+1}: edge type {connection.edge_type} not in agent-controlled set"
            )
            continue

        # Validate temporal constraints
        if not _validate_edge_temporal_constraints(
            graph, connection.source_id, connection.target_id, connection.edge_type
        ):
            logger.warning(
                f"  Skipping connection {i+1}: temporal constraint violation"
            )
            continue

        # Use the original edge object
        graph.edges[connection.id] = connection
        successfully_added.append(connection)

        # Update confidence levels for correction edges
        update_confidence_for_correction_edge(graph, connection)

        added_count += 1
        logger.info(f"  Successfully added connection {i+1}")

    logger.info(f"  Added {added_count} out of {len(connections)} connections to graph")
    return graph, successfully_added


def _validate_edge_temporal_constraints(
    graph: MemoryGraph, source_id: str, target_id: str, edge_type: GraphEdgeType
) -> bool:
    """
    Validate that edge follows temporal constraints for inter-container connections.

    Inter-container edges must go forward in time only.
    Intra-container edges are always allowed.
    """
    # Find containers for source and target
    source_container = None
    target_container = None

    for container in graph.containers.values():
        if source_id in container.element_ids:
            source_container = container
        if target_id in container.element_ids:
            target_container = container

    # If same container, always allow (intra-container edge)
    if (
        source_container
        and target_container
        and source_container.trigger.entry_id == target_container.trigger.entry_id
    ):
        return True

    # If different containers, must be forward in time
    if source_container and target_container:
        return source_container.trigger.timestamp < target_container.trigger.timestamp

    # If containers not found, allow (shouldn't happen in normal operation)
    return True
