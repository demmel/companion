"""
Action-based context management that emits actions instead of directly mutating state.
"""

import logging
from typing import List
from collections import defaultdict

from .models import ContextGraph
from .actions import RemoveFromContextAction
from .context_formatting import format_element

logger = logging.getLogger(__name__)


def prune_context_to_budget_as_actions(
    context: ContextGraph, budget: int
) -> List[RemoveFromContextAction]:
    """
    Determine which memories and edges to remove from context to fit budget and return as actions.

    Uses the same pruning logic as the original function but returns actions
    instead of directly mutating the context.

    Args:
        context: Current context graph
        budget: Token budget to fit within

    Returns:
        List of RemoveFromContextAction actions (may be empty if no pruning needed)
    """
    if not context.elements:
        return []

    # Calculate edge maps for token calculation
    forward_edges_map = defaultdict(list)
    backward_edges_map = defaultdict(list)

    for edge in context.edges:
        forward_edges_map[edge.source_id].append(edge)
        backward_edges_map[edge.target_id].append(edge)

    # Calculate LLM tokens used by each element
    llm_tokens_used_by_element = {}
    for e in context.elements:
        forward_edges = forward_edges_map[e.memory.id]
        backward_edges = backward_edges_map[e.memory.id]
        element_string = format_element(e, forward_edges, backward_edges)
        llm_tokens_used_by_element[e.memory.id] = len(element_string) / 3.4

    llm_tokens_used = sum(llm_tokens_used_by_element.values())

    logger.info(f"  LLM tokens used by context before pruning: {llm_tokens_used:.1f}")
    logger.info(f"  Context budget: {budget}")

    # Determine which memories to remove
    memories_to_remove = []
    remaining_elements = list(context.elements)

    while llm_tokens_used > budget and remaining_elements:
        # Find least valuable memory (same logic as original)
        least_valuable = min(remaining_elements, key=lambda e: e.tokens)

        memories_to_remove.append(least_valuable.memory.id)
        remaining_elements.remove(least_valuable)
        llm_tokens_used -= llm_tokens_used_by_element[least_valuable.memory.id]

    if memories_to_remove:
        # Determine which edges to remove (those that reference removed memories)
        memories_to_remove_set = set(memories_to_remove)
        edges_to_remove = [
            edge.id
            for edge in context.edges
            if edge.source_id in memories_to_remove_set
            or edge.target_id in memories_to_remove_set
        ]

        logger.info(
            f"  Determined {len(memories_to_remove)} memories and {len(edges_to_remove)} edges should be removed for budget"
        )
        return [
            RemoveFromContextAction(
                memory_ids=memories_to_remove,
                edge_ids=edges_to_remove,
                reason=f"Pruned to fit budget of {budget} tokens",
            )
        ]
    else:
        logger.info("  No pruning needed - context fits within budget")
        return []
