"""
Iterative drill-down retrieval for tiered memory.

Instead of dumping all memories at once, the agent iteratively requests
more detail on specific conversations/clusters until it has enough context.
"""

import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from agent.memory.models import MemoryGraph
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.state import State, build_agent_state_description
from agent.chain_of_action.prompts import format_section

from .models import (
    TieredMemoryGraph,
    MemoryTier,
    RetrievalResult,
)
from .tiered_retrieval import (
    retrieve_from_tier_1,
    retrieve_from_tier_2,
    retrieve_from_tier_3,
    retrieve_from_tier_4,
)
from .context_builder import estimate_context_tokens

logger = logging.getLogger(__name__)


def build_lightweight_agent_state_description(state: State, max_priorities: int = 10) -> str:
    """Build a lightweight state description for memory retrieval with limited priorities.

    Copied from agent.state.build_agent_state_description but limits priorities to reduce token usage.
    """
    parts = ["## My Current State\n"]

    # Core identity
    mood_desc = f"{state.current_mood}"
    if state.mood_intensity != "neutral":
        mood_desc += f" ({state.mood_intensity})"
    parts.append(f"**Mood**: {mood_desc}")

    # Appearance and environment
    if state.current_appearance:
        parts.append(f"**Appearance:** {state.current_appearance}")
    if state.current_environment:
        parts.append(f"**Environment:** {state.current_environment}")

    # Values and preferences
    if state.core_values:
        parts.append("\n**Core Values:**")
        for value in state.core_values:
            parts.append(f"- {value.content}")

    # Current priorities (limited to first N)
    if state.current_priorities:
        priorities_to_show = state.current_priorities[:max_priorities]
        if len(state.current_priorities) > max_priorities:
            parts.append(f"\n**My Current Priorities (top {max_priorities} of {len(state.current_priorities)}):**")
        else:
            parts.append("\n**My Current Priorities:**")

        for priority in priorities_to_show:
            parts.append(f"- {priority.content} (id: {priority.id})")

    return "\n".join(parts)


class DrillDownRequest(BaseModel):
    """Request to drill down into a specific memory element."""

    tier: MemoryTier = Field(
        description="Which tier this element is from"
    )
    element_id: str = Field(
        description="ID of the element to drill down into"
    )
    reason: str = Field(
        description="Why I need more detail about this element"
    )


class DrillDownDecision(BaseModel):
    """Agent's decision about whether to drill down for more detail."""

    needs_more_detail: bool = Field(
        description="Whether I need more specific details to respond well"
    )
    drill_down_requests: List[DrillDownRequest] = Field(
        default_factory=list,
        description="Which elements I want to see in more detail (empty if needs_more_detail is false)"
    )
    reasoning: str = Field(
        description="My reasoning about what information I have and what I still need"
    )


def format_tier_results_for_review(
    results: List[RetrievalResult],
    tiered_graph: TieredMemoryGraph,
    tier: MemoryTier,
) -> str:
    """Format retrieval results at a specific tier for agent review."""
    lines = []

    for result in results:
        if result.tier != tier:
            continue

        lines.append(f"[ID: {result.element_id}]")
        lines.append(f"Relevance: {result.score:.3f}")

        if tier == MemoryTier.SEMANTIC_CLUSTER:
            cluster = tiered_graph.semantic_clusters.get(result.element_id)
            if cluster:
                lines.append(f"Topic: {cluster.cluster_topic}")
                lines.append(f"Summary: {cluster.summary}")
                lines.append(f"Contains: {len(cluster.conversation_ids)} conversations")

        elif tier == MemoryTier.CONVERSATION:
            conv = tiered_graph.conversations.get(result.element_id)
            if conv:
                start = conv.start_timestamp.strftime("%Y-%m-%d %H:%M")
                end = conv.end_timestamp.strftime("%Y-%m-%d %H:%M")
                lines.append(f"Time: {start} to {end}")
                lines.append(f"Summary: {conv.summary}")
                lines.append(f"Tags: {', '.join(conv.topic_tags)}")

        elif tier == MemoryTier.TRIGGER_RESPONSE:
            # For tier 2, just show summary
            lines.append(f"Summary: {result.summary}")

        elif tier == MemoryTier.ATOMIC:
            # For tier 1, show the content
            lines.append(f"Content: {result.summary}")

        lines.append("")  # Blank line between results

    return "\n".join(lines)


def iterative_drill_down_retrieval(
    query: str,
    tiered_graph: TieredMemoryGraph,
    memory_graph: MemoryGraph,
    trigger_entries: List[TriggerHistoryEntry],
    trigger_entries_dict: Dict[str, TriggerHistoryEntry],
    state: State,
    llm: LLM,
    model: SupportedModel,
    token_budget: int = 8000,
    max_iterations: int = 3,
    min_similarity: float = 0.3,
) -> Dict:
    """
    Iteratively retrieve memories by letting agent drill down for more detail.

    Args:
        query: The query to retrieve for
        tiered_graph: Tiered memory graph
        memory_graph: Base memory graph
        trigger_entries: All trigger entries
        trigger_entries_dict: Trigger entries by ID
        state: Agent state
        llm: LLM instance
        model: Model to use
        token_budget: Token budget for final context
        max_iterations: Maximum drill-down iterations
        min_similarity: Minimum similarity threshold

    Returns:
        Dictionary with final context and iteration details
    """
    logger.info(f"Starting iterative drill-down retrieval for: '{query}'")

    from agent.embedding_service import get_embedding_service
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.encode(query)

    # Track what we've shown across iterations
    shown_elements = set()
    accumulated_context = []
    iteration_log = []

    # Start with tier 3-4
    current_tier = MemoryTier.SEMANTIC_CLUSTER
    drill_down_filter = None  # IDs to filter to when drilling down
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Iteration {iteration}: Showing tier {current_tier.value}")

        # Get results for current tier
        if current_tier == MemoryTier.SEMANTIC_CLUSTER:
            clusters = list(tiered_graph.semantic_clusters.values())
            # Apply filter if drilling down
            if drill_down_filter:
                clusters = [c for c in clusters if c.id in drill_down_filter]
            tier_results = retrieve_from_tier_4(
                query_embedding, clusters, top_k=5, min_similarity=min_similarity
            )
        elif current_tier == MemoryTier.CONVERSATION:
            conversations = list(tiered_graph.conversations.values())
            # Apply filter if drilling down
            if drill_down_filter:
                conversations = [c for c in conversations if c.id in drill_down_filter]
            tier_results = retrieve_from_tier_3(
                query_embedding, conversations, top_k=10, min_similarity=min_similarity
            )
        elif current_tier == MemoryTier.TRIGGER_RESPONSE:
            # Apply filter if drilling down
            if drill_down_filter:
                filtered_entries = [e for e in trigger_entries if e.entry_id in drill_down_filter]
            else:
                filtered_entries = trigger_entries
            # Show fewer trigger entries since they're verbose
            tier_results = retrieve_from_tier_2(
                query_embedding, filtered_entries, top_k=5, min_similarity=min_similarity
            )
        else:  # ATOMIC
            tier_results = retrieve_from_tier_1(
                query_embedding, memory_graph, top_k=100, min_similarity=min_similarity
            )

        # Filter to only new elements
        new_results = [r for r in tier_results if r.element_id not in shown_elements]

        # Clear the drill-down filter after using it
        drill_down_filter = None

        if not new_results:
            logger.info(f"No new results at tier {current_tier.value}, stopping")
            break

        # Format for agent review
        formatted_results = format_tier_results_for_review(
            new_results, tiered_graph, current_tier
        )

        # Add to accumulated context
        accumulated_context.append(f"--- Tier {current_tier.value} ---")
        accumulated_context.append(formatted_results)

        # Mark as shown
        for result in new_results:
            shown_elements.add(result.element_id)

        # Ask agent if it needs more detail
        state_desc = build_agent_state_description(state)

        drill_down_prompt = f"""I am {state.name}, {state.role}. I'm retrieving memories to answer a query, and I need to decide if I have enough context or need more specific details.

{state_desc}

{format_section("QUERY I'M ANSWERING", query)}

{format_section("MEMORIES I'VE SEEN SO FAR", "\n".join(accumulated_context))}

I need to decide:
1. Do I have enough context to answer this query well, or do I need more specific details?
2. If I need more detail, which specific elements should I drill into?

Consider:
- If the high-level summaries give me enough context, I should stop here
- If I need specific facts, quotes, or details to answer well, I should drill down
- I can request specific conversations from clusters, or specific trigger entries from conversations

Current tier level: {current_tier.value}
Available drill-down: {"One tier deeper" if current_tier != MemoryTier.ATOMIC else "Already at most detailed level"}"""

        try:
            decision = direct_structured_llm_call(
                prompt=drill_down_prompt,
                response_model=DrillDownDecision,
                model=model,
                llm=llm,
                caller="iterative_drill_down_decision"
            )

            iteration_log.append({
                "iteration": iteration,
                "tier": current_tier.value,
                "results_shown": len(new_results),
                "decision": decision.model_dump(),
            })

            logger.info(f"Agent decision: needs_more_detail={decision.needs_more_detail}, "
                       f"requests={len(decision.drill_down_requests)}")

            if not decision.needs_more_detail:
                logger.info("Agent has enough context, stopping drill-down")
                break

            if not decision.drill_down_requests:
                logger.info("Agent wants more but didn't specify what, moving to next tier")
                # Move to next tier down
                if current_tier == MemoryTier.SEMANTIC_CLUSTER:
                    current_tier = MemoryTier.CONVERSATION
                elif current_tier == MemoryTier.CONVERSATION:
                    current_tier = MemoryTier.TRIGGER_RESPONSE
                elif current_tier == MemoryTier.TRIGGER_RESPONSE:
                    current_tier = MemoryTier.ATOMIC
                else:
                    break  # Already at atomic level
            else:
                # Agent requested specific elements - drill into those
                logger.info(f"Agent requested {len(decision.drill_down_requests)} specific elements")

                # Collect allowed IDs from the requested elements
                allowed_ids = set()
                for request in decision.drill_down_requests:
                    if request.tier == MemoryTier.SEMANTIC_CLUSTER:
                        # Drilling into a cluster - allow its conversations
                        cluster = tiered_graph.semantic_clusters.get(request.element_id)
                        if cluster:
                            allowed_ids.update(cluster.conversation_ids)
                    elif request.tier == MemoryTier.CONVERSATION:
                        # Drilling into a conversation - allow its trigger entries
                        conv = tiered_graph.conversations.get(request.element_id)
                        if conv:
                            allowed_ids.update(conv.trigger_entry_ids)
                    elif request.tier == MemoryTier.TRIGGER_RESPONSE:
                        # Drilling into a trigger - would need to get its memory elements
                        # For now, just allow this trigger ID
                        allowed_ids.add(request.element_id)

                if not allowed_ids:
                    logger.warning("No valid IDs found for drill-down requests, stopping")
                    break

                logger.info(f"Drilling down to {len(allowed_ids)} specific child elements")

                # Store the filter for next iteration
                drill_down_filter = allowed_ids

                # Move to next tier down
                if current_tier == MemoryTier.SEMANTIC_CLUSTER:
                    current_tier = MemoryTier.CONVERSATION
                elif current_tier == MemoryTier.CONVERSATION:
                    current_tier = MemoryTier.TRIGGER_RESPONSE
                elif current_tier == MemoryTier.TRIGGER_RESPONSE:
                    current_tier = MemoryTier.ATOMIC
                else:
                    break

        except Exception as e:
            logger.error(f"Drill-down decision failed: {e}")
            break

    # Final context is everything we've accumulated
    final_context = "\n\n".join(accumulated_context)
    final_tokens = estimate_context_tokens(final_context)

    logger.info(f"Iterative retrieval complete: {iteration} iterations, {final_tokens} tokens")

    return {
        "final_context": final_context,
        "final_tokens": final_tokens,
        "iterations": iteration,
        "iteration_log": iteration_log,
        "elements_shown": len(shown_elements),
    }
