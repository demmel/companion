"""
Strategic token allocation for retrieved memories in DAG memory system.

Ensures retrieved memories get appropriate token values to survive pruning
while boosting existing context memories that connect to retrieved content.
"""

import logging
from typing import List, Dict, Set
from dataclasses import dataclass

from .models import MemoryElement, ContextElement, ContextGraph
from .contextual_expansion import MemoryCluster
from .similarity_scoring import MemoryRetrievalCandidate

logger = logging.getLogger(__name__)


@dataclass
class TokenAllocation:
    """Token allocation result for a memory."""

    memory_id: str
    base_tokens: int
    relevance_bonus: int
    context_bonus: int
    bridge_bonus: int
    final_tokens: int
    reasoning: str


class TokenAllocator:
    """Handles strategic token allocation for retrieved memories."""

    def __init__(self):
        """Initialize token allocator with default parameters."""
        # Base token ranges
        self.min_retrieved_tokens = 50
        self.max_retrieved_tokens = 100

        # Bonus amounts
        self.context_connection_bonus = 25
        self.bridge_bonus = 35
        self.supporting_memory_bonus = 15

        # Existing context boost
        self.existing_context_boost = 20

    def allocate_tokens_for_clusters(
        self,
        clusters: List[MemoryCluster],
        bridge_data: List[tuple],  # (cluster, bridge_memories, bridge_edges)
        current_context: ContextGraph
    ) -> List[TokenAllocation]:
        """
        Allocate strategic tokens for memory clusters and bridges.

        Args:
            clusters: Memory clusters to allocate tokens for
            bridge_data: Bridge information from contextual expansion
            current_context: Current context graph for connection analysis

        Returns:
            List of token allocations for all memories
        """
        allocations = []
        context_memory_ids = {elem.memory.id for elem in current_context.elements}

        # Process each cluster
        for i, cluster in enumerate(clusters):
            # Find corresponding bridge data
            bridge_memories = []
            bridge_edges = []
            for cluster_data, bridges, edges in bridge_data:
                if cluster_data == cluster:
                    bridge_memories = bridges
                    bridge_edges = edges
                    break

            # Allocate tokens for primary memory
            primary_allocation = self._allocate_primary_memory_tokens(
                cluster, i, bridge_memories, context_memory_ids
            )
            allocations.append(primary_allocation)

            # Allocate tokens for supporting memories
            for j, supporting_memory in enumerate(cluster.supporting_memories):
                supporting_allocation = self._allocate_supporting_memory_tokens(
                    supporting_memory, cluster, j, context_memory_ids
                )
                allocations.append(supporting_allocation)

            # Allocate tokens for bridge memories
            for j, bridge_memory in enumerate(bridge_memories):
                bridge_allocation = self._allocate_bridge_memory_tokens(
                    bridge_memory, cluster, j
                )
                allocations.append(bridge_allocation)

        logger.info(
            f"Allocated tokens for {len(allocations)} memories across {len(clusters)} clusters"
        )

        return allocations

    def _allocate_primary_memory_tokens(
        self,
        cluster: MemoryCluster,
        cluster_rank: int,
        bridge_memories: List[MemoryElement],
        context_memory_ids: Set[str]
    ) -> TokenAllocation:
        """Allocate tokens for the primary memory in a cluster."""

        # Base tokens based on relevance score and ranking
        relevance_ratio = min(cluster.relevance_score, 1.0)
        rank_penalty = min(cluster_rank * 5, 20)  # Reduce tokens for lower-ranked clusters

        base_tokens = int(
            self.min_retrieved_tokens +
            (self.max_retrieved_tokens - self.min_retrieved_tokens) * relevance_ratio
        ) - rank_penalty

        # Relevance bonus
        relevance_bonus = int(relevance_ratio * 20)

        # Context connection bonus
        context_bonus = 0
        if cluster.primary_memory.id in context_memory_ids:
            context_bonus = self.context_connection_bonus

        # Bridge bonus
        bridge_bonus = 0
        if bridge_memories:
            bridge_bonus = self.bridge_bonus

        final_tokens = max(base_tokens + relevance_bonus + context_bonus + bridge_bonus, 30)

        reasoning = f"Primary memory (rank {cluster_rank + 1}, relevance {cluster.relevance_score:.3f})"
        if context_bonus > 0:
            reasoning += ", connects to context"
        if bridge_bonus > 0:
            reasoning += f", bridges via {len(bridge_memories)} memories"

        return TokenAllocation(
            memory_id=cluster.primary_memory.id,
            base_tokens=base_tokens,
            relevance_bonus=relevance_bonus,
            context_bonus=context_bonus,
            bridge_bonus=bridge_bonus,
            final_tokens=final_tokens,
            reasoning=reasoning
        )

    def _allocate_supporting_memory_tokens(
        self,
        memory: MemoryElement,
        cluster: MemoryCluster,
        support_rank: int,
        context_memory_ids: Set[str]
    ) -> TokenAllocation:
        """Allocate tokens for supporting memories in a cluster."""

        # Base tokens for supporting memories (lower than primary)
        base_tokens = self.min_retrieved_tokens + self.supporting_memory_bonus
        rank_penalty = min(support_rank * 3, 15)
        base_tokens -= rank_penalty

        # Small relevance bonus based on cluster relevance
        relevance_bonus = int(cluster.relevance_score * 10)

        # Context connection bonus
        context_bonus = 0
        if memory.id in context_memory_ids:
            context_bonus = self.context_connection_bonus

        final_tokens = max(base_tokens + relevance_bonus + context_bonus, 25)

        reasoning = f"Supporting memory {support_rank + 1} in cluster"
        if context_bonus > 0:
            reasoning += ", connects to context"

        return TokenAllocation(
            memory_id=memory.id,
            base_tokens=base_tokens,
            relevance_bonus=relevance_bonus,
            context_bonus=context_bonus,
            bridge_bonus=0,
            final_tokens=final_tokens,
            reasoning=reasoning
        )

    def _allocate_bridge_memory_tokens(
        self,
        memory: MemoryElement,
        cluster: MemoryCluster,
        bridge_rank: int
    ) -> TokenAllocation:
        """Allocate tokens for bridge memories connecting clusters to context."""

        # Bridge memories get substantial tokens for their connecting value
        base_tokens = self.min_retrieved_tokens
        bridge_bonus = self.bridge_bonus
        rank_penalty = min(bridge_rank * 2, 10)

        # Small relevance bonus from connected cluster
        relevance_bonus = int(cluster.relevance_score * 5)

        final_tokens = max(base_tokens + bridge_bonus + relevance_bonus - rank_penalty, 30)

        reasoning = f"Bridge memory {bridge_rank + 1} connecting to context"

        return TokenAllocation(
            memory_id=memory.id,
            base_tokens=base_tokens,
            relevance_bonus=relevance_bonus,
            context_bonus=0,
            bridge_bonus=bridge_bonus,
            final_tokens=final_tokens,
            reasoning=reasoning
        )

    def calculate_context_boosts(
        self,
        context_graph: ContextGraph,
        retrieved_memory_ids: Set[str],
        connecting_edge_ids: Set[str]
    ) -> Dict[str, int]:
        """
        Calculate token boosts for existing context memories that connect to retrieved content.

        Args:
            context_graph: Current context graph
            retrieved_memory_ids: Set of memory IDs being retrieved
            connecting_edge_ids: Set of edge IDs that connect to retrieved memories

        Returns:
            Dictionary mapping memory_id -> boost_amount
        """
        boosts = {}

        # Boost memories that have edges connecting to retrieved memories
        for edge in context_graph.edges:
            if edge.id in connecting_edge_ids:
                # Boost both source and target if they're in context but not retrieved
                for memory_id in [edge.source_id, edge.target_id]:
                    if memory_id not in retrieved_memory_ids:
                        # Check if this memory is in current context
                        for elem in context_graph.elements:
                            if elem.memory.id == memory_id:
                                current_boost = boosts.get(memory_id, 0)
                                boosts[memory_id] = max(current_boost, self.existing_context_boost)
                                break

        logger.debug(f"Calculated context boosts for {len(boosts)} existing memories")

        return boosts


def create_context_elements_with_tokens(
    memories: List[MemoryElement],
    allocations: List[TokenAllocation]
) -> List[ContextElement]:
    """
    Create ContextElement objects with strategic token allocation.

    Args:
        memories: List of memory elements to create context elements for
        allocations: List of token allocations

    Returns:
        List of ContextElement objects with allocated tokens
    """
    # Create allocation lookup
    allocation_map = {alloc.memory_id: alloc for alloc in allocations}

    context_elements = []
    for memory in memories:
        allocation = allocation_map.get(memory.id)
        if allocation:
            context_element = ContextElement(
                memory=memory,
                tokens=allocation.final_tokens
            )
            context_elements.append(context_element)
            logger.debug(
                f"Created context element for {memory.id} with {allocation.final_tokens} tokens: {allocation.reasoning}"
            )

    return context_elements