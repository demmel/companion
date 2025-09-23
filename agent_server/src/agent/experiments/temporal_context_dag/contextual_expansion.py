"""
Contextual memory expansion for DAG memory retrieval.

Ensures retrieved memories come with their essential context by following
edges and identifying dependency clusters, preventing isolated memory retrieval.
"""

import logging
from typing import List, Set, Dict, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

from .models import MemoryGraph, ContextGraph, MemoryElement, MemoryEdge
from .edge_types import EdgeType, GraphEdgeType, REVERSE_MAPPING
from .similarity_scoring import MemoryRetrievalCandidate

logger = logging.getLogger(__name__)


@dataclass
class MemoryCluster:
    """A cluster of related memories that should be retrieved together."""

    primary_memory: MemoryElement
    supporting_memories: List[MemoryElement]
    connecting_edges: List[MemoryEdge]
    cluster_size: int
    relevance_score: float


class ContextualExpander:
    """Handles contextual expansion of retrieved memories."""

    def __init__(self, memory_graph: MemoryGraph, context_graph: ContextGraph):
        """
        Initialize contextual expander.

        Args:
            memory_graph: Complete memory graph for expansion
            context_graph: Current context graph for connection finding
        """
        self.memory_graph = memory_graph
        self.context_graph = context_graph

        # Build edge lookup maps for efficient traversal
        self.outgoing_edges = defaultdict(list)  # memory_id -> [edges from this memory]
        self.incoming_edges = defaultdict(list)  # memory_id -> [edges to this memory]

        for edge in memory_graph.edges.values():
            self.outgoing_edges[edge.source_id].append(edge)
            self.incoming_edges[edge.target_id].append(edge)

    def expand_retrieved_memories(
        self,
        candidates: List[MemoryRetrievalCandidate],
        max_expansion_depth: int = 2,
        max_cluster_size: int = 8
    ) -> List[MemoryCluster]:
        """
        Expand retrieved memory candidates with their essential context.

        Args:
            candidates: List of memory candidates from similarity scoring
            max_expansion_depth: Maximum depth to traverse for context
            max_cluster_size: Maximum total memories per cluster

        Returns:
            List of memory clusters with context dependencies
        """
        clusters = []

        for candidate in candidates:
            cluster = self._build_memory_cluster(
                candidate,
                max_expansion_depth,
                max_cluster_size
            )
            if cluster:
                clusters.append(cluster)

        # Remove overlapping clusters, keeping higher scoring ones
        clusters = self._deduplicate_clusters(clusters)

        logger.info(
            f"Expanded {len(candidates)} candidates into {len(clusters)} memory clusters"
        )

        return clusters

    def _build_memory_cluster(
        self,
        candidate: MemoryRetrievalCandidate,
        max_depth: int,
        max_size: int
    ) -> MemoryCluster:
        """Build a memory cluster starting from a candidate memory."""

        primary_memory = candidate.memory
        visited_memories = {primary_memory.id}
        supporting_memories = []
        connecting_edges = []

        # Use BFS to explore context dependencies
        queue = deque([(primary_memory.id, 0)])  # (memory_id, depth)

        while queue and len(visited_memories) < max_size:
            memory_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Get contextual dependencies
            dependencies = self._get_contextual_dependencies(memory_id)

            for dep_memory_id, edge in dependencies:
                if dep_memory_id not in visited_memories:
                    dep_memory = self.memory_graph.elements.get(dep_memory_id)
                    if dep_memory:
                        visited_memories.add(dep_memory_id)
                        supporting_memories.append(dep_memory)
                        connecting_edges.append(edge)
                        queue.append((dep_memory_id, depth + 1))

                        if len(visited_memories) >= max_size:
                            break

        return MemoryCluster(
            primary_memory=primary_memory,
            supporting_memories=supporting_memories,
            connecting_edges=connecting_edges,
            cluster_size=len(visited_memories),
            relevance_score=candidate.combined_score
        )

    def _get_contextual_dependencies(self, memory_id: str) -> List[Tuple[str, MemoryEdge]]:
        """
        Get memories that provide essential context for the given memory.

        Returns memories that:
        1. Explain or provide background (explained_by, caused_by edges)
        2. Contradict, clarify, or retract to show conflicts
        3. Are part of temporal sequences (follows/followed_by)
        """
        dependencies = []

        # Context-providing edges (incoming)
        context_providing_types = {
            GraphEdgeType.EXPLAINED_BY,
            GraphEdgeType.CAUSED,  # Something caused this memory
            GraphEdgeType.CLARIFIED_BY,
            GraphEdgeType.RETRACTED_BY,
            GraphEdgeType.CONTRADICTED_BY,
            GraphEdgeType.FOLLOWED_BY  # Chronological predecessor
        }

        for edge in self.incoming_edges[memory_id]:
            if edge.edge_type in context_providing_types:
                dependencies.append((edge.source_id, edge))

        # Consequences and clarifications (outgoing)
        context_requiring_types = {
            GraphEdgeType.EXPLAINS,
            GraphEdgeType.CAUSED  # This memory caused something else
        }

        for edge in self.outgoing_edges[memory_id]:
            if edge.edge_type in context_requiring_types:
                dependencies.append((edge.target_id, edge))

        return dependencies

    def _deduplicate_clusters(self, clusters: List[MemoryCluster]) -> List[MemoryCluster]:
        """
        Remove overlapping clusters, keeping higher scoring ones.

        If two clusters share significant overlap, keep the higher scoring one.
        """
        if len(clusters) <= 1:
            return clusters

        # Sort by relevance score (highest first)
        clusters.sort(key=lambda c: c.relevance_score, reverse=True)

        deduplicated = []
        used_memories = set()

        for cluster in clusters:
            # Get all memory IDs in this cluster
            cluster_memory_ids = {cluster.primary_memory.id}
            cluster_memory_ids.update(m.id for m in cluster.supporting_memories)

            # Check overlap with already used memories
            overlap = cluster_memory_ids & used_memories
            overlap_ratio = len(overlap) / len(cluster_memory_ids)

            # Keep cluster if overlap is small
            if overlap_ratio < 0.5:  # Less than 50% overlap
                deduplicated.append(cluster)
                used_memories.update(cluster_memory_ids)

        logger.debug(
            f"Deduplicated {len(clusters)} clusters to {len(deduplicated)} "
            f"(removed {len(clusters) - len(deduplicated)} overlapping clusters)"
        )

        return deduplicated

    def find_bridges_to_context(
        self,
        clusters: List[MemoryCluster],
        max_bridge_distance: int = 3
    ) -> List[Tuple[MemoryCluster, List[MemoryElement], List[MemoryEdge]]]:
        """
        Find bridge memories that connect retrieved clusters to current context.

        Args:
            clusters: Memory clusters to connect
            max_bridge_distance: Maximum distance to search for bridges

        Returns:
            List of (cluster, bridge_memories, bridge_edges) tuples
        """
        if not self.context_graph.elements:
            return [(cluster, [], []) for cluster in clusters]

        context_memory_ids = {elem.memory.id for elem in self.context_graph.elements}
        bridged_clusters = []

        for cluster in clusters:
            bridge_memories, bridge_edges = self._find_shortest_bridge_path(
                cluster, context_memory_ids, max_bridge_distance
            )
            bridged_clusters.append((cluster, bridge_memories, bridge_edges))

        return bridged_clusters

    def _find_shortest_bridge_path(
        self,
        cluster: MemoryCluster,
        context_memory_ids: Set[str],
        max_distance: int
    ) -> Tuple[List[MemoryElement], List[MemoryEdge]]:
        """Find shortest path from cluster to any memory in current context."""

        # Get all memory IDs in the cluster
        cluster_memory_ids = {cluster.primary_memory.id}
        cluster_memory_ids.update(m.id for m in cluster.supporting_memories)

        # BFS to find shortest path to context
        queue = deque([(memory_id, 0, []) for memory_id in cluster_memory_ids])
        visited = set(cluster_memory_ids)

        while queue:
            memory_id, distance, path_edges = queue.popleft()

            if distance >= max_distance:
                continue

            # Check all connected memories
            for edge in self.outgoing_edges[memory_id] + self.incoming_edges[memory_id]:
                next_memory_id = edge.target_id if edge.source_id == memory_id else edge.source_id

                if next_memory_id in context_memory_ids:
                    # Found a bridge to context!
                    bridge_memories = []
                    bridge_edges = path_edges + [edge]

                    # Collect bridge memories (excluding cluster and context memories)
                    for bridge_edge in bridge_edges:
                        for mem_id in [bridge_edge.source_id, bridge_edge.target_id]:
                            if (mem_id not in cluster_memory_ids and
                                mem_id not in context_memory_ids and
                                mem_id in self.memory_graph.elements):
                                memory = self.memory_graph.elements[mem_id]
                                if memory not in bridge_memories:
                                    bridge_memories.append(memory)

                    return bridge_memories, bridge_edges

                elif next_memory_id not in visited:
                    visited.add(next_memory_id)
                    queue.append((next_memory_id, distance + 1, path_edges + [edge]))

        # No bridge found within distance limit
        return [], []