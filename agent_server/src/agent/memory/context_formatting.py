"""
Functions for formatting context graphs for display in prompts.
"""

from collections import defaultdict
from agent.memory.edge_types import (
    REVERSE_MAPPING,
    EdgeType,
    get_edge_type_context_descrioptions,
)
from .models import ContextElement, ContextGraph, MemoryEdge, MemoryGraph
from agent.chain_of_action.trigger import format_trigger_for_prompt


def format_element(
    element: ContextElement,
    forward_edges: list[MemoryEdge],
    backward_edges: list[MemoryEdge],
) -> str:
    """
    Format a single context element (memory) with its edges for display.

    Args:
        element: The context element to format
        forward_edges: List of forward edges connected to this memory
        backward_edges: List of backward edges connected to this memory

    Returns:
        Formatted string representation of the context element
    """

    # Format memory header
    memory_id = element.memory.id[:8]  # Shortened ID for readability
    timestamp = element.memory.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(
        f"{memory_id}: [{timestamp}] (Confidence: {element.memory.confidence_level.value})"
    )
    lines.append(f"  Content: {element.memory.content}")

    # Format forward edges
    for edge in forward_edges:
        target_id = edge.target_id[:8]
        lines.append(f"  -[{edge.edge_type.value}]-> {target_id}")

    # Format backward edges (reverse the edge type wording for clarity)
    for edge in backward_edges:
        source_id = edge.source_id[:8]

        reversed_edge_type = REVERSE_MAPPING[EdgeType(edge.edge_type.value)]
        lines.append(f"  <-[{reversed_edge_type}]- {source_id}")

    return "\n".join(lines)


def format_container(container_id: str, graph: MemoryGraph) -> str:
    """
    Format all memories in a container for display.

    Args:
        container_id: ID of the memory container to format
        graph: The memory graph containing the container and its memories
    Returns:
        Formatted string representation of the container's memories
    """
    container = graph.containers.get(container_id)
    assert container is not None, "Container ID not found in graph"

    trigger_entry = container.trigger
    timestamp = trigger_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return f"""[{timestamp}]
{format_trigger_for_prompt(trigger_entry.trigger)}

{trigger_entry.compressed_summary}"""


def format_context(
    context: ContextGraph,
    memory_graph: MemoryGraph,
    use_individual_formatting: bool = False,
) -> str:
    """
    Format a context graph for display in agent prompts.

    When memory_graph is provided, memories are grouped by container and displayed
    using compressed summaries. Otherwise, individual memories are shown.

    Args:
        context: The context graph to format
        memory_graph: Optional memory graph to access containers for compressed display

    Returns:
        Formatted string representation of the context
    """
    if not context.elements:
        return "No memories currently in context."

    lines = []

    if use_individual_formatting:
        lines.extend(
            [
                "## Memory System",
                "",
                "This is your memory graph showing important memories and their relationships.",
                "Memories are ordered chronologically (oldest to newest).",
                "Edges show relationships: -[edge_type]-> points to later memories, <-[edge_type]- points to earlier memories.",
                "",
                "**Confidence Levels:**",
                "- `user_confirmed`: Direct user statements/confirmations - highest reliability",
                "- `strong_inference`: High-confidence deductions from user input",
                "- `reasonable_assumption`: Logical assumptions that could be wrong",
                "- `speculative`: Uncertain inferences - use with caution",
                "- `likely_error`: Probably incorrect but not confirmed",
                "- `known_false`: Definitively corrected/contradicted - treat as false",
                "",
                *get_edge_type_context_descrioptions(),
                "",
                "## Memories",
            ]
        )

        # Create edge lookups for O(1) access
        forward_edges_map = defaultdict(list)  # memory_id -> list of outgoing edges
        backward_edges_map = defaultdict(list)  # memory_id -> list of incoming edges

        for edge in context.edges:
            forward_edges_map[edge.source_id].append(edge)
            backward_edges_map[edge.target_id].append(edge)

        # Sort memories chronologically
        sorted_elements = sorted(context.elements, key=lambda e: e.memory.timestamp)
        for element in sorted_elements:
            forward_edges = forward_edges_map[element.memory.id]
            backward_edges = backward_edges_map[element.memory.id]
            formatted_element = format_element(element, forward_edges, backward_edges)
            lines.append(formatted_element)
            lines.append("")  # Blank line between memories

    else:
        # Group memories by container
        container_groups = defaultdict(list)
        for element in context.elements:
            container_groups[element.memory.container_id].append(element)

        # Sort containers chronologically
        sorted_containers = sorted(
            container_groups.items(),
            key=lambda item: memory_graph.containers[item[0]].trigger.timestamp,
        )

        for container_id, elements in sorted_containers:
            lines.append(format_container(container_id, memory_graph))
            lines.append("")  # Blank line between containers

    return "\n".join(lines)
