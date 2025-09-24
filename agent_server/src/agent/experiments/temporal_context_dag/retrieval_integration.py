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
from .token_allocation import get_memory_tokens, get_reinforce_tokens

logger = logging.getLogger(__name__)


def retrieve_relevant_memories_as_actions(
    memory_graph: MemoryGraph,
    context_graph: ContextGraph,
    state: State,
    trigger: Trigger,
    llm: LLM,
    model: SupportedModel,
    max_retrieved_memories: int = 3,
    max_queries: int = 6,
    min_similarity_threshold: float = 0.4,
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

        # Step 3: Use minimal spanning to connect memories efficiently
        logger.debug("Connecting memories with minimal spanning tree")
        from .minimal_spanning import MinimalSpanningConnector

        # Get current context node IDs
        context_nodes = {elem.memory.id for elem in context_graph.elements}

        # Create minimal spanning connector
        connector = MinimalSpanningConnector(memory_graph.elements, memory_graph.edges)

        # Extract target memories from candidates with weights
        target_memories = [c.memory.id for c in candidates]
        memory_weights = {c.memory.id: c.combined_score for c in candidates}

        # Find minimal spanning connection
        spanning_result = connector.find_minimal_spanning_connection(
            target_memories=target_memories,
            context_nodes=context_nodes,
            token_budget=2000,
            memory_weights=memory_weights,
        )

        logger.info(
            f"Connected {len(spanning_result.connected_memories)}/{len(candidates)} memories "
            f"using minimal spanning with {spanning_result.total_cost:.2f} total cost"
        )

        # Step 4: Generate actions from spanning result
        logger.debug("Generating retrieval actions")
        actions = []

        # Add all required intermediate nodes
        for node_id in spanning_result.required_nodes:
            if node_id in memory_graph.elements:
                memory = memory_graph.elements[node_id]
                actions.append(
                    AddToContextAction(
                        memory_id=node_id,
                        initial_tokens=get_memory_tokens(
                            memory.emotional_significance, memory.memory_type
                        ),
                        reinforce_tokens=get_reinforce_tokens(
                            memory.memory_type, memory.emotional_significance
                        ),
                    )
                )

        # Add connected target memories
        for memory_id in spanning_result.connected_memories:
            if memory_id in memory_graph.elements:
                memory = memory_graph.elements[memory_id]
                actions.append(
                    AddToContextAction(
                        memory_id=memory_id,
                        initial_tokens=get_memory_tokens(
                            memory.emotional_significance, memory.memory_type
                        ),
                        reinforce_tokens=get_reinforce_tokens(
                            memory.memory_type, memory.emotional_significance
                        ),
                    )
                )

        # Add all required edges
        for edge_id in spanning_result.required_edges:
            if edge_id in memory_graph.edges:
                actions.append(
                    AddEdgeToContextAction(
                        edge_id=edge_id, should_boost_source_tokens=False
                    )
                )

        logger.info(
            f"Generated {len(actions)} retrieval actions: "
            f"{len(spanning_result.required_nodes) + len(spanning_result.connected_memories)} memories, "
            f"{len(spanning_result.required_edges)} edges"
        )

        return actions

    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}", exc_info=True)
        return []
