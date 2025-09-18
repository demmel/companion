"""
Functions for formatting context graphs for display in prompts.
"""

from .models import ContextElement, ContextGraph, MemoryEdge


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
        "You can create new connections between memories using these relationship types:",
        "",
        "**Connection Types:**",
        "- `relates_to`: Two memories are thematically related or share common elements",
        "- `explains`: One memory provides context, background, or explanation for another",
        "- `follows`: One memory is a direct chronological sequel or continuation of another",
        "- `updates`: One memory contains newer information that supersedes or refines another",
        "",
        "## Memories",
    ]

    for node in sorted(
        context.elements,
        key=lambda e: e.memory.timestamp,
    ):
        lines.append(format_element(node))

    lines.append("\n## Connections")
    for edge in context.edges:
        lines.append(format_edge(edge))

    return "\n".join(lines)
