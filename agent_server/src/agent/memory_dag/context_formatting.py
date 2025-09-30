"""
Functions for formatting context graphs for display in prompts.
"""

from agent.memory_dag.edge_types import (
    REVERSE_MAPPING,
    EdgeType,
    get_edge_type_context_descrioptions,
)
from .models import ContextElement, ContextGraph, MemoryEdge


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
        f"{memory_id}: [{timestamp}] (Type: {element.memory.memory_type.value}) (Confidence: {element.memory.confidence_level.value})"
    )
    lines.append(f"  Content: {element.memory.content}")
    lines.append(f"  Evidence: {element.memory.evidence}")

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


def format_context(context: ContextGraph) -> str:
    """
    Format a context graph for display in agent prompts.
    Memories are presented chronologically with edges embedded directly.

    Args:
        context: The context graph to format

    Returns:
        Formatted string representation of the context
    """
    if not context.elements:
        return "No memories currently in context."

    lines = [
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

    # Create edge lookups for O(1) access
    from collections import defaultdict

    forward_edges_map = defaultdict(list)  # memory_id -> list of outgoing edges
    backward_edges_map = defaultdict(list)  # memory_id -> list of incoming edges

    for edge in context.edges:
        forward_edges_map[edge.source_id].append(edge)
        backward_edges_map[edge.target_id].append(edge)

    # Sort memories chronologically with sequence as secondary sort key
    sorted_memories = sorted(
        context.elements,
        key=lambda e: (e.memory.timestamp, e.memory.sequence_in_container),
    )

    for element in sorted_memories:
        # Get edges for this memory using O(1) lookups
        forward_edges = forward_edges_map[element.memory.id]
        backward_edges = backward_edges_map[element.memory.id]

        # Format and append the memory with its edges
        memory_str = format_element(element, forward_edges, backward_edges)
        lines.append(memory_str)
        lines.append("")  # Empty line between memories

    return "\n".join(lines)
