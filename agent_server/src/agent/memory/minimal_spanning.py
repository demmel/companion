"""
Minimal spanning graph algorithms for connecting retrieved memories to context.

This module implements graph algorithms to efficiently connect retrieved memories
to the current context with minimal node/edge overhead, solving the context
explosion problem.
"""

from typing import Dict, List, Set, Optional
import heapq
from dataclasses import dataclass

from .models import MemoryElement, MemoryEdge


@dataclass
class ConnectionPath:
    """Represents a path connecting a memory to context."""

    target_memory_id: str  # Memory we want to connect
    path_nodes: List[str]  # Node IDs in the path (including target)
    path_edges: List[str]  # Edge IDs in the path
    total_cost: float  # Total cost of this path


@dataclass
class SpanningResult:
    """Result of minimal spanning graph computation."""

    connected_memories: Set[str]  # Memory IDs that got connected
    required_nodes: Set[str]  # All node IDs needed in context
    required_edges: Set[str]  # All edge IDs needed in context
    total_cost: float  # Total cost of the spanning solution
    paths: List[ConnectionPath]  # Individual paths for each connected memory


class MinimalSpanningConnector:
    """
    Finds minimal spanning subgraph to connect retrieved memories to context.

    This solves the context explosion problem by sharing paths between memories
    rather than finding individual shortest paths.
    """

    def __init__(
        self, memory_graph: Dict[str, MemoryElement], edges: Dict[str, MemoryEdge]
    ):
        """
        Initialize with the full memory graph.

        Args:
            memory_graph: Dictionary of memory_id -> MemoryElement
            edges: Dictionary of edge_id -> MemoryEdge
        """
        self.memory_graph = memory_graph
        self.edges = edges

        # Build adjacency lists for efficient graph traversal
        self.outgoing_edges: Dict[str, List[str]] = {}  # node -> [edge_ids]
        self.incoming_edges: Dict[str, List[str]] = {}  # node -> [edge_ids]

        for edge_id, edge in edges.items():
            # Outgoing edges
            if edge.source_id not in self.outgoing_edges:
                self.outgoing_edges[edge.source_id] = []
            self.outgoing_edges[edge.source_id].append(edge_id)

            # Incoming edges
            if edge.target_id not in self.incoming_edges:
                self.incoming_edges[edge.target_id] = []
            self.incoming_edges[edge.target_id].append(edge_id)

    def find_minimal_spanning_connection(
        self,
        target_memories: List[str],  # Memory IDs we want to connect
        context_nodes: Set[str],  # Memory IDs already in context
        token_budget: int,  # Maximum tokens we can spend
        memory_weights: Dict[str, float],  # Priority weights per memory
    ) -> SpanningResult:
        """
        Find minimal Steiner tree to connect target memories to context supernode.

        This solves the Steiner tree problem: find minimum cost subgraph that connects
        all target memories to the context supernode, sharing intermediate nodes
        between paths to minimize context explosion.

        Args:
            target_memories: Memory IDs we want to connect to context
            context_nodes: Memory IDs already in context (treated as supernode)
            token_budget: Maximum token budget for new nodes
            memory_weights: Priority weights per memory (higher = more important)

        Returns:
            SpanningResult with optimal Steiner tree solution
        """
        if not target_memories or not context_nodes:
            return SpanningResult(
                connected_memories=set(),
                required_nodes=set(),
                required_edges=set(),
                total_cost=0.0,
                paths=[],
            )

        # Solve Steiner tree problem
        steiner_result = self._solve_steiner_tree(
            targets=set(target_memories),
            terminals=context_nodes,
            memory_weights=memory_weights,
            token_budget=token_budget,
        )

        return steiner_result

    def _solve_steiner_tree(
        self,
        targets: Set[str],
        terminals: Set[str],
        memory_weights: Dict[str, float],
        token_budget: int,
    ) -> SpanningResult:
        """
        Solve Steiner tree problem to connect targets to terminals with minimal intermediate nodes.
        """
        steiner_nodes = set(terminals)  # Start with terminals in the tree
        steiner_edges = set()
        connected_targets = set()
        paths = []
        current_tokens = 0

        # Sort targets by priority
        sorted_targets = sorted(
            targets, key=lambda t: memory_weights.get(t, 1.0), reverse=True
        )

        for target in sorted_targets:
            if target in steiner_nodes:
                # Already connected
                connected_targets.add(target)
                paths.append(ConnectionPath(target, [target], [], 0.0))
                continue

            # Find shortest path to any node already in Steiner tree
            path = self._find_shortest_path_to_context(target, steiner_nodes)

            if path:
                # Calculate new nodes (excluding ones already in tree)
                new_nodes = [n for n in path.path_nodes if n not in steiner_nodes]
                estimated_tokens = len(new_nodes) * 100

                if current_tokens + estimated_tokens <= token_budget:
                    # Add this path to the tree
                    steiner_nodes.update(path.path_nodes)
                    steiner_edges.update(path.path_edges)
                    connected_targets.add(target)
                    current_tokens += estimated_tokens

                    paths.append(path)

        # Calculate total cost properly (only new nodes, not path costs)
        total_cost = (
            len(steiner_nodes - terminals) * 1.0
        )  # Cost = number of intermediate nodes

        return SpanningResult(
            connected_memories=connected_targets,
            required_nodes=steiner_nodes - terminals,
            required_edges=steiner_edges,
            total_cost=total_cost,
            paths=paths,
        )

    def _find_shortest_path_to_context(
        self, start_memory_id: str, context_nodes: Set[str]
    ) -> Optional[ConnectionPath]:
        """
        Find shortest path from start_memory to any node in context_nodes.

        Uses Dijkstra's algorithm with edge and node costs.
        """
        if start_memory_id in context_nodes:
            # Already in context
            return ConnectionPath(
                target_memory_id=start_memory_id,
                path_nodes=[start_memory_id],
                path_edges=[],
                total_cost=0.0,
            )

        # Dijkstra's algorithm
        # State: (cost, current_node, path_nodes, path_edges)
        heap = [(0.0, start_memory_id, [start_memory_id], [])]
        visited = set()

        while heap:
            current_cost, current_node, path_nodes, path_edges = heapq.heappop(heap)

            if current_node in visited:
                continue
            visited.add(current_node)

            # Check if we reached context
            if current_node in context_nodes:
                return ConnectionPath(
                    target_memory_id=start_memory_id,
                    path_nodes=path_nodes,
                    path_edges=path_edges,
                    total_cost=current_cost,
                )

            # Explore neighbors - treat graph as undirected by following both outgoing and incoming edges

            # Follow outgoing edges
            for edge_id in self.outgoing_edges.get(current_node, []):
                edge = self.edges[edge_id]
                next_node = edge.target_id

                if next_node in visited:
                    continue

                # Calculate costs
                edge_cost = self._get_edge_cost(edge)
                node_cost = self._get_node_cost(next_node)
                new_cost = current_cost + edge_cost + node_cost

                new_path_nodes = path_nodes + [next_node]
                new_path_edges = path_edges + [edge_id]

                heapq.heappush(
                    heap, (new_cost, next_node, new_path_nodes, new_path_edges)
                )

            # Follow incoming edges (reverse direction)
            for edge_id in self.incoming_edges.get(current_node, []):
                edge = self.edges[edge_id]
                next_node = edge.source_id  # Go backwards along the edge

                if next_node in visited:
                    continue

                # Calculate costs
                edge_cost = self._get_edge_cost(edge)
                node_cost = self._get_node_cost(next_node)
                new_cost = current_cost + edge_cost + node_cost

                new_path_nodes = path_nodes + [next_node]
                new_path_edges = path_edges + [edge_id]

                heapq.heappush(
                    heap, (new_cost, next_node, new_path_nodes, new_path_edges)
                )

        return None  # No path found

    def _select_optimal_paths(
        self,
        memory_paths: Dict[str, ConnectionPath],
        token_budget: int,
        memory_weights: Dict[str, float],
    ) -> Dict[str, ConnectionPath]:
        """
        Select optimal subset of paths that maximizes value while minimizing total cost.

        This is a variant of the knapsack problem where we want to:
        1. Maximize total memory value (based on weights)
        2. Minimize overlapping nodes (shared paths are better)
        3. Stay within token budget
        """
        if not memory_paths:
            return {}

        # Simple greedy approach: prioritize by value/cost ratio
        # More sophisticated algorithms (dynamic programming) could be used here

        # Calculate value/cost ratio for each memory
        memory_scores = []
        for memory_id, path in memory_paths.items():
            weight = memory_weights.get(memory_id, 1.0)
            # Higher weight, lower cost = better score
            score = weight / max(path.total_cost, 0.1)  # Avoid division by zero
            memory_scores.append((score, memory_id, path))

        # Sort by score (highest first)
        memory_scores.sort(reverse=True, key=lambda x: x[0])

        # Greedily select paths while tracking shared nodes
        selected_paths = {}
        used_nodes = set()
        current_tokens = 0

        for score, memory_id, path in memory_scores:
            # Calculate how many new nodes this path would add
            new_nodes = set(path.path_nodes) - used_nodes

            # Estimate tokens for new nodes (rough estimate)
            estimated_tokens = (
                len(new_nodes) * 100
            )  # Assume 100 tokens per node average

            if current_tokens + estimated_tokens <= token_budget:
                selected_paths[memory_id] = path
                used_nodes.update(path.path_nodes)
                current_tokens += estimated_tokens

        return selected_paths

    def _get_edge_cost(self, edge: MemoryEdge) -> float:
        """Calculate cost of including an edge in the spanning tree."""
        from .edge_types import GraphEdgeType

        # CONSTRAINT edges should be CHEAPEST to prioritize constraint retrieval
        match edge.edge_type:
            case GraphEdgeType.CONTRADICTED_BY:
                return 0.1  # Constraint edges are cheapest - we want these!
            case GraphEdgeType.RETRACTED_BY:
                return 0.1  # Constraint edges are cheapest - we want these!
            case GraphEdgeType.SUPERSEDED_BY:
                return 0.1  # Superseding edges are cheapest - we want these!
            case GraphEdgeType.CLARIFIED_BY:
                return 0.2  # Clarification edges are very cheap
            case GraphEdgeType.CAUSED:
                return 1.0  # Causal edges are medium cost
            case GraphEdgeType.EXPLAINS:
                return 1.0  # Explanatory edges are medium cost
            case GraphEdgeType.EXPLAINED_BY:
                return 1.0  # Explanatory edges are medium cost
            case _:
                return 1.0  # Default cost for unknown edge types

    def _get_node_cost(self, node_id: str) -> float:
        """Calculate cost of including a node in the spanning tree."""
        return 1.0
