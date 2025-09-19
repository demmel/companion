"""
Functions for formatting context graphs for display in prompts.
"""

from .models import ContextElement, ContextGraph, MemoryEdge, MemoryEdgeType

# Mapping for reversing edge type wording in backward edges for clarity
EDGE_TYPE_REVERSAL = {
    MemoryEdgeType.EXPLAINED_BY: "explains",
    MemoryEdgeType.EXPLAINS: "explained_by",
    MemoryEdgeType.FOLLOWED_BY: "follows",
    MemoryEdgeType.UPDATED_BY: "updates",
    MemoryEdgeType.CAUSED: "caused_by"
}


def format_element(element: ContextElement) -> str:
    """Format a single context element for display."""
    return f"""- ID: {element.memory.id} Time: {element.memory.timestamp.isoformat()}
  - Content: {element.memory.content}
  - Significance: {element.memory.emotional_significance:.2f}
"""


def format_edge(edge: MemoryEdge) -> str:
    """Format a single context edge for display."""
    return (
        f"- Connection: {edge.source_id} --[{edge.edge_type.value}]--> {edge.target_id}"
    )


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
        "**Connection Types:**",
        "- `explained_by`: This memory is given context, background, or reasoning by another memory",
        "- `explains`: This memory provides context, background, or reasoning for another memory",
        "- `followed_by`: This memory happens before another memory in chronological sequence",
        "- `updated_by`: This memory is superseded, corrected, or refined by newer information",
        "- `caused`: This memory directly caused, triggered, or led to another memory",
        "- `follows`: This memory happens after another memory in chronological sequence",
        "- `updates`: This memory supersedes, corrects, or refines older information",
        "- `caused_by`: This memory was directly caused, triggered, or resulted from another memory",
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

    # Sort memories chronologically
    sorted_memories = sorted(context.elements, key=lambda e: e.memory.timestamp)

    for element in sorted_memories:
        # Format memory header
        memory_id = element.memory.id[:8]  # Shortened ID for readability
        timestamp = element.memory.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        lines.append(f"{memory_id}: [{timestamp}]")
        lines.append(f"  Content: {element.memory.content}")
        lines.append(f"  Evidence: {element.memory.evidence}")

        # Get edges for this memory using O(1) lookups
        forward_edges = forward_edges_map[element.memory.id]
        backward_edges = backward_edges_map[element.memory.id]

        # Format forward edges
        for edge in forward_edges:
            target_id = edge.target_id[:8]
            lines.append(f"  -[{edge.edge_type.value}]-> {target_id}")

        # Format backward edges (reverse the edge type wording for clarity)
        for edge in backward_edges:
            source_id = edge.source_id[:8]
            reversed_edge_type = EDGE_TYPE_REVERSAL.get(edge.edge_type, edge.edge_type.value)
            lines.append(f"  <-[{reversed_edge_type}]- {source_id}")

        lines.append("")  # Empty line between memories

    return "\n".join(lines)
