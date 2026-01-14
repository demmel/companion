"""
Action-based context management that emits actions instead of directly mutating state.
"""

import logging
from typing import List

from agent.memory.dag.action_log import MemoryGraph
from agent.llm.router import LLM
from agent.llm.models import SupportedModel

from .models import ContextGraph
from .actions import RemoveFromContextAction
from .context_formatting import format_context

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

    remaining_elements = list(context.elements)
    remaining_edges = list(context.edges)

    def calculate_total_tokens() -> int:
        temp_context = ContextGraph(elements=remaining_elements, edges=remaining_edges)
        formatted = format_context(temp_context, graph, use_individual_formatting)
        return llm.estimate_tokens(formatted, model)

    total_tokens = calculate_total_tokens()

    logger.info(f"  LLM tokens used by context before pruning: {total_tokens}")
    logger.info(f"  Context budget: {budget}")

    memories_to_remove: list[str] = []
    edges_to_remove: list[str] = []

    while total_tokens > budget and remaining_elements:
        # Find least valuable memory (lowest .tokens priority)
        least_valuable = min(remaining_elements, key=lambda e: e.tokens)
        memory_id = least_valuable.memory.id

        # Remove this memory
        memories_to_remove.append(memory_id)
        remaining_elements.remove(least_valuable)

        # Find and remove all edges involving this memory
        new_remaining_edges = []
        for edge in remaining_edges:
            if edge.source_id == memory_id or edge.target_id == memory_id:
                edges_to_remove.append(edge.id)
            else:
                new_remaining_edges.append(edge)
        remaining_edges = new_remaining_edges

        # Recalculate total tokens
        total_tokens = calculate_total_tokens()

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
