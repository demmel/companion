"""
Main retrieval integration function for DAG memory system.

Orchestrates multi-query extraction, similarity scoring, contextual expansion,
and token allocation to generate actions for memory retrieval using existing
action types.
"""

import logging
from typing import List

from agent.llm import LLM, SupportedModel
from agent.state import State

from .models import MemoryGraph, ContextGraph
from .actions import MemoryAction, AddToContextAction, AddEdgeToContextAction
from .memory_retrieval import extract_memory_queries
from .similarity_scoring import retrieve_top_candidates
from .contextual_expansion import ContextualExpander
from .token_allocation import TokenAllocator, create_context_elements_with_tokens

logger = logging.getLogger(__name__)


def retrieve_relevant_memories_as_actions(
    memory_graph: MemoryGraph,
    context_graph: ContextGraph,
    state: State,
    llm: LLM,
    model: SupportedModel,
    max_retrieved_memories: int = 8,
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
            top_k=max_retrieved_memories * 2,  # Get more candidates for expansion
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

        # Step 3: Expand candidates with contextual dependencies
        logger.debug("Expanding memories with contextual dependencies")
        expander = ContextualExpander(memory_graph, context_graph)

        # Limit candidates to requested amount before expansion
        top_candidates = candidates[:max_retrieved_memories]
        clusters = expander.expand_retrieved_memories(
            candidates=top_candidates, max_expansion_depth=2, max_cluster_size=6
        )

        # Find bridges to current context
        bridge_data = expander.find_bridges_to_context(
            clusters=clusters, max_bridge_distance=3
        )

        logger.info(
            f"Created {len(clusters)} memory clusters with contextual expansion"
        )

        # Step 4: Allocate strategic tokens
        logger.debug("Allocating strategic tokens")
        allocator = TokenAllocator()
        allocations = allocator.allocate_tokens_for_clusters(
            clusters=clusters, bridge_data=bridge_data, current_context=context_graph
        )

        # Step 5: Generate actions using existing action types
        logger.debug("Generating retrieval actions")
        actions = []

        # Collect all unique memories to add
        memories_to_add = []
        memory_ids_added = set()

        for cluster in clusters:
            # Add primary memory
            if cluster.primary_memory.id not in memory_ids_added:
                memories_to_add.append(cluster.primary_memory)
                memory_ids_added.add(cluster.primary_memory.id)

            # Add supporting memories
            for memory in cluster.supporting_memories:
                if memory.id not in memory_ids_added:
                    memories_to_add.append(memory)
                    memory_ids_added.add(memory.id)

        # Add bridge memories
        for _, bridge_memories, _ in bridge_data:
            for memory in bridge_memories:
                if memory.id not in memory_ids_added:
                    memories_to_add.append(memory)
                    memory_ids_added.add(memory.id)

        # Create context elements with strategic tokens
        context_elements = create_context_elements_with_tokens(
            memories=memories_to_add, allocations=allocations
        )

        for context_element in context_elements:
            actions.append(
                AddToContextAction(context_element=context_element, reinforce_tokens=20)
            )

        # Generate AddEdgeToContextAction for connecting edges
        edges_to_add = []
        edge_ids_added = set()

        # Add cluster connecting edges
        for cluster in clusters:
            for edge in cluster.connecting_edges:
                if (
                    edge.id not in edge_ids_added
                    and edge.source_id in memory_ids_added
                    and edge.target_id in memory_ids_added
                ):
                    edges_to_add.append(edge)
                    edge_ids_added.add(edge.id)

        # Add bridge edges
        for _, _, bridge_edges in bridge_data:
            for edge in bridge_edges:
                if edge.id not in edge_ids_added:
                    # Only add edge if both endpoints will be in context
                    source_in_context = edge.source_id in memory_ids_added or any(
                        elem.memory.id == edge.source_id
                        for elem in context_graph.elements
                    )
                    target_in_context = edge.target_id in memory_ids_added or any(
                        elem.memory.id == edge.target_id
                        for elem in context_graph.elements
                    )

                    if source_in_context and target_in_context:
                        edges_to_add.append(edge)
                        edge_ids_added.add(edge.id)

        # Generate AddEdgeToContextAction for each edge
        for edge in edges_to_add:
            actions.append(
                AddEdgeToContextAction(edge=edge, should_boost_source_tokens=False)
            )

        logger.info(
            f"Generated {len(actions)} retrieval actions: "
            f"{len(context_elements)} memories, {len(edges_to_add)} edges"
        )

        # Log what memories were actually retrieved
        if context_elements:
            logger.info("Retrieved memories for context:")
            for i, elem in enumerate(context_elements, 1):
                preview = (
                    elem.memory.content[:80] + "..."
                    if len(elem.memory.content) > 80
                    else elem.memory.content
                )
                logger.info(f"  {i}. [{elem.tokens} tokens] {preview}")
        else:
            logger.warning("No memories were retrieved for context!")

        return actions

    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}", exc_info=True)
        return []
