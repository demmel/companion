"""
Action-based context management that emits actions instead of directly mutating state.
"""

import logging
from typing import List
from collections import defaultdict

from agent.memory.action_log import MemoryGraph

from .models import ContextGraph
from .actions import RemoveFromContextAction
from .context_formatting import format_container, format_element

logger = logging.getLogger(__name__)


def prune_context_to_budget_as_actions(
    graph: MemoryGraph, context: ContextGraph, budget: int
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

    containers = {e.memory.container_id for e in context.elements}
    llm_tokens_used_by_container = {}
    for container_id in containers:
        container_string = format_container(container_id, graph)
        llm_tokens_used_by_container[container_id] = len(container_string) / 3.4

    llm_tokens_used = sum(llm_tokens_used_by_container.values())

    logger.info(f"  LLM tokens used by context before pruning: {llm_tokens_used:.1f}")
    logger.info(f"  Context budget: {budget}")

    # Determine which memories to remove
    memories_to_remove = []
    remaining_elements = list(context.elements)
    remaining_element_ids = {e.memory.id for e in remaining_elements}

    while llm_tokens_used > budget and remaining_elements:
        # Find least valuable memory (same logic as original)
        least_valuable = min(remaining_elements, key=lambda e: e.tokens)

        memories_to_remove.append(least_valuable.memory.id)
        remaining_elements.remove(least_valuable)
        remaining_element_ids.remove(least_valuable.memory.id)

        container_id = least_valuable.memory.container_id
        container = graph.containers[container_id]
        intersection = remaining_element_ids & {e for e in container.element_ids}
        if not intersection:
            # If this was the last memory in the container, remove container cost
            llm_tokens_used -= llm_tokens_used_by_container[container_id]
            logger.info(
                f"  Removing memory {least_valuable.memory.id[:8]} also removes container {container_id[:8]} cost"
            )

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
