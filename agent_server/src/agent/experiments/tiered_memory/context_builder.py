"""
Context builder for formatting tiered retrieval results for LLM prompts.

Formats retrieved memories hierarchically to match production memory formatting style.
"""

import logging
from typing import Dict
from datetime import datetime

from agent.memory.models import MemoryGraph
from agent.chain_of_action.trigger_history import TriggerHistoryEntry

from .models import (
    TieredRetrievalResults,
    MemoryTier,
    TieredMemoryGraph,
)

logger = logging.getLogger(__name__)


def format_for_llm_prompt(
    retrieval_results: TieredRetrievalResults,
    tiered_graph: TieredMemoryGraph,
    memory_graph: MemoryGraph,
    trigger_entries_dict: Dict[str, TriggerHistoryEntry],
    token_budget: int = 8000,
) -> str:
    """
    Format retrieval results hierarchically for LLM prompt, filling token budget.

    Greedily adds memories by relevance until token budget is exhausted.
    Shows higher-tier items with their summaries, and lower-tier children
    indented underneath. Matches production formatting style for tier 2.

    Args:
        retrieval_results: Retrieval results sorted by relevance
        tiered_graph: Tiered memory graph
        memory_graph: Memory graph
        trigger_entries_dict: Trigger entries by ID
        token_budget: Maximum tokens to use (default 8000)

    Returns:
        Formatted context string within token budget
    """
    from agent.chain_of_action.trigger import format_trigger_for_prompt

    if not retrieval_results.results:
        return "No relevant memories found."

    results_by_id = {r.element_id: r for r in retrieval_results.results}

    # Phase 1: Greedily select items by similarity until budget is full
    selected_items = []
    current_tokens = 0

    for result in retrieval_results.results:
        # Format item standalone (no children)
        item_text = _format_standalone_item(
            result, tiered_graph, memory_graph, trigger_entries_dict
        )
        item_tokens = estimate_context_tokens(item_text)

        # Check if it fits in budget
        if current_tokens + item_tokens > token_budget:
            logger.debug(
                f"Budget full: {current_tokens}/{token_budget} tokens, "
                f"cannot fit {result.tier.value} ({item_tokens} tokens)"
            )
            break

        # Add to selected items
        selected_items.append(result)
        current_tokens += item_tokens

        logger.debug(
            f"Selected {result.tier.value} (score: {result.score:.3f}, "
            f"tokens: {item_tokens}, total: {current_tokens}/{token_budget})"
        )

    logger.info(
        f"Selected {len(selected_items)} items using {current_tokens}/{token_budget} tokens"
    )

    if not selected_items:
        return "No memories fit within token budget."

    # Phase 2: Format selected items with hierarchy
    # Build parent-child relationships for selected items
    selected_ids = {item.element_id for item in selected_items}

    def get_parent_id(result):
        """Get parent ID if parent was also selected."""
        if result.tier == MemoryTier.ATOMIC:
            # Parent is the container (tier 2)
            mem = memory_graph.elements.get(result.element_id)
            if mem and mem.container_id in selected_ids:
                return mem.container_id
        elif result.tier == MemoryTier.TRIGGER_RESPONSE:
            # Check if parent conversation (tier 3) was selected
            for conv_id, conv in tiered_graph.conversations.items():
                if (
                    conv_id in selected_ids
                    and result.element_id in conv.trigger_entry_ids
                ):
                    return conv_id
        elif result.tier == MemoryTier.CONVERSATION:
            # Check if parent cluster (tier 4) was selected
            for cluster_id, cluster in tiered_graph.semantic_clusters.items():
                if (
                    cluster_id in selected_ids
                    and result.element_id in cluster.conversation_ids
                ):
                    return cluster_id
        return None

    def get_timestamp(result):
        """Get timestamp for sorting."""
        if result.tier == MemoryTier.SEMANTIC_CLUSTER:
            return datetime.max
        elif result.tier == MemoryTier.CONVERSATION:
            conv = tiered_graph.conversations.get(result.element_id)
            return conv.start_timestamp if conv else datetime.min
        elif result.tier == MemoryTier.TRIGGER_RESPONSE:
            entry = trigger_entries_dict.get(result.element_id)
            return entry.timestamp if entry else datetime.min
        elif result.tier == MemoryTier.ATOMIC:
            mem = memory_graph.elements.get(result.element_id)
            return mem.timestamp if mem else datetime.min
        return datetime.min

    # Group items by parent (None = top-level)
    children_by_parent = {}
    for item in selected_items:
        parent_id = get_parent_id(item)
        if parent_id not in children_by_parent:
            children_by_parent[parent_id] = []
        children_by_parent[parent_id].append(item)

    # Recursively format items with hierarchy
    shown = set()
    lines = []

    def format_with_children(result, indent=""):
        """Format item and its children that were also selected."""
        if result.element_id in shown:
            return
        shown.add(result.element_id)

        # Format this item
        item_lines = _format_standalone_item(
            result, tiered_graph, memory_graph, trigger_entries_dict, indent
        )
        lines.append(item_lines)

        # Format children that were also selected
        children = children_by_parent.get(result.element_id, [])
        if children:
            for child in sorted(children, key=get_timestamp):
                format_with_children(child, indent + "  ")

    # Format all top-level items (those without selected parents)
    top_level_items = children_by_parent.get(None, [])

    # Sort top-level items by tier (highest first), then timestamp
    tier_order = {
        MemoryTier.SEMANTIC_CLUSTER: 0,
        MemoryTier.CONVERSATION: 1,
        MemoryTier.TRIGGER_RESPONSE: 2,
        MemoryTier.ATOMIC: 3,
    }

    top_level_items.sort(
        key=lambda r: (
            tier_order[r.tier],
            (
                -get_timestamp(r).timestamp()
                if get_timestamp(r) != datetime.min and get_timestamp(r) != datetime.max
                else 0
            ),
        )
    )

    for item in top_level_items:
        format_with_children(item)

    return "\n".join(lines).rstrip()


def _format_standalone_item(
    result,
    tiered_graph: TieredMemoryGraph,
    memory_graph: MemoryGraph,
    trigger_entries_dict: Dict[str, TriggerHistoryEntry],
    indent: str = "",
) -> str:
    """Format a single item without children."""
    from agent.chain_of_action.trigger import format_trigger_for_prompt

    lines = []

    if result.tier == MemoryTier.SEMANTIC_CLUSTER:
        cluster = tiered_graph.semantic_clusters.get(result.element_id)
        if not cluster:
            return ""
        lines.append(f"{indent}**Topic: {cluster.cluster_topic}**")
        lines.append(f"{indent}{cluster.summary}")
        lines.append("")

    elif result.tier == MemoryTier.CONVERSATION:
        conv = tiered_graph.conversations.get(result.element_id)
        if not conv:
            return ""
        start = conv.start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        end = conv.end_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{indent}[Conversation: {start} to {end}]")
        lines.append(f"{indent}{conv.summary}")
        lines.append("")

    elif result.tier == MemoryTier.TRIGGER_RESPONSE:
        entry = trigger_entries_dict.get(result.element_id)
        if not entry:
            return ""
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        trigger_formatted = format_trigger_for_prompt(entry.trigger)

        lines.append(f"{indent}[{timestamp}]")
        if indent:
            for line in trigger_formatted.split("\n"):
                lines.append(f"{indent}{line}" if line.strip() else "")
        else:
            lines.append(trigger_formatted)
        lines.append("")

        if indent:
            for line in entry.compressed_summary.split("\n"):
                lines.append(f"{indent}{line}" if line.strip() else "")
        else:
            lines.append(entry.compressed_summary)
        lines.append("")

    elif result.tier == MemoryTier.ATOMIC:
        mem = memory_graph.elements.get(result.element_id)
        if not mem:
            return ""
        timestamp = mem.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{indent}[{timestamp}]")

        if indent:
            for line in mem.content.split("\n"):
                lines.append(f"{indent}{line}" if line.strip() else "")
        else:
            lines.append(mem.content)
        lines.append("")

    return "\n".join(lines)


def estimate_context_tokens(context_string: str) -> int:
    """
    Estimate token count for a context string.

    Uses rough approximation of 3.4 characters per token.

    Args:
        context_string: The context string

    Returns:
        Estimated token count
    """
    return int(len(context_string) / 3.4)
