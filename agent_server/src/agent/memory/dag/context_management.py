"""
Action-based context management that emits actions instead of directly mutating state.
"""

import logging
from typing import List
from collections import defaultdict

from agent.memory.dag.action_log import MemoryGraph
from agent.llm.router import LLM
from agent.llm.models import SupportedModel

from .models import ContextGraph
from .actions import RemoveFromContextAction
from .context_formatting import (
    format_container,
    format_element,
    format_forward_edge,
    format_backward_edge,
    format_individual_header,
)

logger = logging.getLogger(__name__)


def prune_context_to_budget_as_actions(
    graph: MemoryGraph,
    context: ContextGraph,
    budget: int,
    use_individual_formatting: bool,
    llm: LLM,
    model: SupportedModel,
) -> List[RemoveFromContextAction]:
    """
    Determine which memories and edges to remove from context to fit budget.

    Args:
        graph: Memory graph containing containers
        context: Current context graph
        budget: Token budget to fit within
        use_individual_formatting: If True, calculate costs for individual memory
            formatting. If False, use container-based compressed summary formatting.
        llm: LLM instance for token estimation
        model: Model to use for token estimation

    Returns:
        List of RemoveFromContextAction actions (may be empty if no pruning needed)
    """
    if not context.elements:
        return []

    if use_individual_formatting:
        return _prune_individual_mode(graph, context, budget, llm, model)
    else:
        return _prune_container_mode(graph, context, budget, llm, model)


def _prune_individual_mode(
    graph: MemoryGraph,
    context: ContextGraph,
    budget: int,
    llm: LLM,
    model: SupportedModel,
) -> List[RemoveFromContextAction]:
    """Prune using individual memory formatting costs."""
    # Build edge maps
    forward_edges_map: dict[str, list] = defaultdict(list)
    backward_edges_map: dict[str, list] = defaultdict(list)
    for edge in context.edges:
        forward_edges_map[edge.source_id].append(edge)
        backward_edges_map[edge.target_id].append(edge)
    # Calculate header cost
    header_tokens = llm.estimate_tokens(format_individual_header(), model)

    # Calculate node costs (memory content without edges)
    node_costs: dict[str, int] = {}
    for element in context.elements:
        formatted = format_element(element, [], [])
        node_costs[element.memory.id] = llm.estimate_tokens(formatted, model)

    # Calculate edge costs
    # Each edge is displayed twice: forward on source, backward on target
    edge_forward_costs: dict[str, int] = {}
    edge_backward_costs: dict[str, int] = {}

    for edge in context.edges:
        edge_forward_costs[edge.id] = llm.estimate_tokens(
            format_forward_edge(edge), model
        )
        edge_backward_costs[edge.id] = llm.estimate_tokens(
            format_backward_edge(edge), model
        )

    # Calculate blank line cost (one per memory)
    blank_line_tokens = llm.estimate_tokens("\n", model)

    # Total tokens = header + sum of node costs + all edge display costs + blank lines
    total_tokens = header_tokens
    total_tokens += sum(node_costs.values())
    for edge in context.edges:
        total_tokens += edge_forward_costs[edge.id]
        total_tokens += edge_backward_costs[edge.id]
    total_tokens += len(context.elements) * blank_line_tokens

    logger.info(f"  LLM tokens used by context before pruning: {total_tokens}")
    logger.info(f"  Context budget: {budget}")

    # Track which memories and edges remain
    remaining_elements = list(context.elements)
    remaining_memory_ids: set[str] = {e.memory.id for e in remaining_elements}
    remaining_edge_ids: set[str] = {e.id for e in context.edges}

    memories_to_remove: list[str] = []
    edges_to_remove: list[str] = []

    while total_tokens > budget and remaining_elements:
        # Find least valuable memory (lowest .tokens priority)
        least_valuable = min(remaining_elements, key=lambda e: e.tokens)
        memory_id = least_valuable.memory.id

        # Remove this memory
        memories_to_remove.append(memory_id)
        remaining_elements.remove(least_valuable)
        remaining_memory_ids.remove(memory_id)

        # Subtract node cost + blank line
        total_tokens -= node_costs[memory_id]
        total_tokens -= blank_line_tokens

        # Find and remove all edges involving this memory
        edges_involving_memory = []
        for edge in forward_edges_map[memory_id]:
            if edge.id in remaining_edge_ids:
                edges_involving_memory.append(edge)
        for edge in backward_edges_map[memory_id]:
            if edge.id in remaining_edge_ids:
                edges_involving_memory.append(edge)

        for edge in edges_involving_memory:
            edges_to_remove.append(edge.id)
            remaining_edge_ids.remove(edge.id)

            # Subtract edge costs (both forward and backward displays)
            total_tokens -= edge_forward_costs[edge.id]
            total_tokens -= edge_backward_costs[edge.id]

    if memories_to_remove:
        logger.info(
            f"  Determined {len(memories_to_remove)} memories and {len(edges_to_remove)} edges should be removed for budget"
        )
        logger.info(f"  Tokens after pruning: {total_tokens}")
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


def _prune_container_mode(
    graph: MemoryGraph,
    context: ContextGraph,
    budget: int,
    llm: LLM,
    model: SupportedModel,
) -> List[RemoveFromContextAction]:
    """Prune using container-based compressed summary formatting costs."""
    # Calculate per-container costs
    containers = {e.memory.container_id for e in context.elements}
    tokens_by_container: dict[str, int] = {}
    for container_id in containers:
        container_string = format_container(container_id, graph)
        tokens_by_container[container_id] = llm.estimate_tokens(container_string, model)

    total_tokens = sum(tokens_by_container.values())

    logger.info(f"  LLM tokens used by context before pruning: {total_tokens}")
    logger.info(f"  Context budget: {budget}")

    # Track remaining elements
    memories_to_remove: list[str] = []
    remaining_elements = list(context.elements)
    remaining_element_ids = {e.memory.id for e in remaining_elements}

    while total_tokens > budget and remaining_elements:
        # Find least valuable memory
        least_valuable = min(remaining_elements, key=lambda e: e.tokens)

        memories_to_remove.append(least_valuable.memory.id)
        remaining_elements.remove(least_valuable)
        remaining_element_ids.remove(least_valuable.memory.id)

        container_id = least_valuable.memory.container_id
        container = graph.containers[container_id]
        intersection = remaining_element_ids & set(container.element_ids)
        if not intersection:
            # Last memory in container - remove container cost
            total_tokens -= tokens_by_container[container_id]
            logger.info(
                f"  Removing memory {least_valuable.memory.id[:8]} also removes container {container_id[:8]} cost"
            )

    if memories_to_remove:
        # Determine which edges to remove
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
