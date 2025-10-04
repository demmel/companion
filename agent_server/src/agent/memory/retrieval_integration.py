"""
Main retrieval integration function for DAG memory system.

Orchestrates multi-query extraction, similarity scoring, contextual expansion,
and token allocation to generate actions for memory retrieval using existing
action types.
"""

import logging
from typing import List

from agent.chain_of_action.trigger import Trigger
from agent.llm import LLM, SupportedModel
from agent.state import State

from .models import MemoryGraph, ContextGraph
from .actions import MemoryAction, AddToContextAction, AddEdgeToContextAction
from .memory_retrieval import extract_memory_queries
from .similarity_scoring import retrieve_top_candidates

logger = logging.getLogger(__name__)


def retrieve_relevant_memories_as_actions(
    memory_graph: MemoryGraph,
    context_graph: ContextGraph,
    state: State,
    trigger: Trigger,
    llm: LLM,
    model: SupportedModel,
    max_retrieved_memories: int,
    max_queries: int,
    min_similarity_threshold: float,
) -> List[MemoryAction]:
    """
    Retrieve relevant memories and return them as actions using existing action types.

    This is the main orchestration function that:
    1. Extracts diverse queries from current context
    2. Scores all memories against those queries
    3. Expands top candidates with contextual dependencies
    4. Allocates strategic tokens
    5. Returns AddToContextAction and AddEdgeToContextAction objects

    Args:
        memory_graph: Complete memory graph to search
        context_graph: Current context graph
        state: Current agent state
        llm: LLM instance for query generation
        model: Model to use for LLM calls
        max_retrieved_memories: Maximum memories to retrieve
        max_queries: Maximum queries to generate
        min_similarity_threshold: Minimum similarity score to retrieve

    Returns:
        List of MemoryAction objects (AddToContextAction, AddEdgeToContextAction)
    """
    logger.info("Starting memory retrieval process")

    try:
        # Step 1: Extract diverse queries from current context
        logger.debug("Extracting memory queries from current context")
        query_result = extract_memory_queries(
            context=context_graph,
            state=state,
            llm=llm,
            model=model,
            max_queries=max_queries,
            trigger=trigger,
            memory_graph=memory_graph,
        )

        if not query_result.queries:
            logger.warning("No queries extracted from context, skipping retrieval")
            return []

        logger.info(
            f"Extracted {len(query_result.queries)} memory queries for retrieval"
        )

        # Step 2: Score all memories against queries
        logger.debug("Scoring memories against queries")
        candidates = retrieve_top_candidates(
            memory_graph=memory_graph,
            queries=query_result.queries,
            top_k=max_retrieved_memories,
            min_similarity_threshold=min_similarity_threshold,
            combination_strategy="weighted_max",
        )

        if not candidates:
            logger.warning(
                f"No candidates met similarity threshold {min_similarity_threshold}, skipping retrieval"
            )
            return []

        logger.info(
            f"Found {len(candidates)} memory candidates above threshold {min_similarity_threshold}"
        )

        # Step 3: Use update chain expansion instead of minimal spanning
        logger.debug("Expanding memories with update chains")
        from .update_chain_connector import UpdateChainConnector

        # Create update chain connector
        connector = UpdateChainConnector(memory_graph.elements, memory_graph.edges)

        # Extract target memories from candidates with weights
        target_memories = [c.memory.id for c in candidates]
        memory_weights: dict[str, float] = {
            c.memory.id: c.combined_score for c in candidates
        }

        # Expand target memories with their update chains
        update_chain_result = connector.expand_with_update_chains(
            target_memories=target_memories,
            token_budget=2000,
            memory_weights=memory_weights,
        )

        logger.info(
            f"Connected {len(update_chain_result.connected_memories)}/{len(candidates)} memories "
            f"using update chains with {update_chain_result.total_memories} total memories"
        )

        # Step 4: Generate actions from update chain result with dynamic token allocation
        logger.debug("Generating retrieval actions with dynamic token allocation")
        actions = []

        # Calculate min tokens among context elements NOT in update chains
        all_chain_nodes = update_chain_result.required_nodes
        non_chain_context_elements = [
            elem
            for elem in context_graph.elements
            if elem.memory.id not in all_chain_nodes
        ]

        if non_chain_context_elements:
            min_non_chain_tokens = min(
                elem.tokens for elem in non_chain_context_elements
            )
            safe_token_count = min_non_chain_tokens + 1

            logger.debug(
                f"Setting update chain nodes to {safe_token_count} tokens for pruning safety"
            )

            # Get currently in-context chain nodes
            in_context_chain_ids = {
                elem.memory.id
                for elem in context_graph.elements
                if elem.memory.id in all_chain_nodes
            }

            # Add all chain nodes that are NOT already in context
            for node_id in update_chain_result.required_nodes:
                if (
                    node_id in memory_graph.elements
                    and node_id not in in_context_chain_ids
                ):
                    actions.append(
                        AddToContextAction(
                            memory_id=node_id,
                            initial_tokens=safe_token_count,
                            reinforce_tokens=0,
                        )
                    )

            # Reinforce chain nodes that are already in context
            for elem in context_graph.elements:
                if elem.memory.id in all_chain_nodes:
                    needed_tokens = max(0, safe_token_count - elem.tokens)
                    if needed_tokens > 0:
                        actions.append(
                            AddToContextAction(
                                memory_id=elem.memory.id,
                                initial_tokens=0,  # Already in context
                                reinforce_tokens=needed_tokens,
                            )
                        )
        else:
            # All context elements are in the update chains - no need to adjust tokens
            logger.debug(
                "All context elements are in update chains, no token adjustments needed"
            )

        # Add all required edges
        for edge_id in update_chain_result.required_edges:
            if edge_id in memory_graph.edges:
                actions.append(
                    AddEdgeToContextAction(
                        edge_id=edge_id, should_boost_source_tokens=False
                    )
                )

        logger.info(
            f"Generated {len(actions)} retrieval actions: "
            f"{len(update_chain_result.required_nodes)} memories, "
            f"{len(update_chain_result.required_edges)} edges"
        )

        return actions

    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}", exc_info=True)
        return []
