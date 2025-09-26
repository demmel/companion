"""
Update chain connector for following correction/update edges instead of full connectivity.

This module implements retrieval that follows only update chains (supersedes, clarifies,
contradicts, retracts, corrects) to avoid pulling in excessive context while ensuring
we don't miss important corrections or updates to retrieved memories.
"""

from typing import Dict, List, Set, Optional
from collections import deque
from dataclasses import dataclass

from .models import MemoryElement, MemoryEdge
from .edge_types import GraphEdgeType


@dataclass
class UpdateChainResult:
    """Result of update chain expansion."""

    connected_memories: Set[str]  # Memory IDs that got connected
    required_nodes: Set[str]  # All node IDs needed in context (includes chains)
    required_edges: Set[str]  # All edge IDs needed in context
    total_memories: int  # Total number of memories retrieved


class UpdateChainConnector:
    """
    Expands retrieved memories by following update chains only.

    This avoids the context explosion problem by following only correction/update
    edges rather than trying to connect memories through arbitrary relationships.
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

        # Update edge types that should be followed in chains
        self.update_edge_types = {
            GraphEdgeType.SUPERSEDED_BY,
            GraphEdgeType.CLARIFIED_BY,
            GraphEdgeType.CONTRADICTED_BY,
            GraphEdgeType.RETRACTED_BY,
            GraphEdgeType.CORRECTED_BY
        }

        # Build adjacency lists for efficient traversal
        self._build_update_adjacency()

    def _build_update_adjacency(self):
        """Build adjacency lists for update edges (both directions for full chains)."""
        self.outgoing_updates = {}  # memory_id -> list of (target_id, edge_id)
        self.incoming_updates = {}  # memory_id -> list of (source_id, edge_id)

        for edge_id, edge in self.edges.items():
            if edge.edge_type in self.update_edge_types:
                # Outgoing: source -> target
                if edge.source_id not in self.outgoing_updates:
                    self.outgoing_updates[edge.source_id] = []
                self.outgoing_updates[edge.source_id].append((edge.target_id, edge_id))

                # Incoming: target <- source
                if edge.target_id not in self.incoming_updates:
                    self.incoming_updates[edge.target_id] = []
                self.incoming_updates[edge.target_id].append((edge.source_id, edge_id))

    def expand_with_update_chains(
        self,
        target_memories: List[str],
        token_budget: int = 2000,
        memory_weights: Optional[Dict[str, float]] = None
    ) -> UpdateChainResult:
        """
        Expand target memories with their update chains.

        For each target memory, follows both forward and backward update chains
        to ensure we capture the complete correction/update history.

        Args:
            target_memories: List of memory IDs to expand with chains
            token_budget: Maximum token budget (unused in this simple approach)
            memory_weights: Optional weights for memories (unused in this approach)

        Returns:
            UpdateChainResult with all memories and edges in the chains
        """
        all_required_nodes = set()
        all_required_edges = set()
        connected_memories = set()

        for memory_id in target_memories:
            if memory_id in self.memory_graph:
                # Get the full update chain for this memory
                chain_nodes, chain_edges = self._get_update_chain(memory_id)

                all_required_nodes.update(chain_nodes)
                all_required_edges.update(chain_edges)
                connected_memories.add(memory_id)

        return UpdateChainResult(
            connected_memories=connected_memories,
            required_nodes=all_required_nodes,
            required_edges=all_required_edges,
            total_memories=len(all_required_nodes)
        )

    def _get_update_chain(self, memory_id: str) -> tuple[Set[str], Set[str]]:
        """
        Get the complete update chain for a memory (both directions).

        Follows both incoming and outgoing update edges to get the full
        correction/update chain for proper context.

        Args:
            memory_id: Starting memory ID

        Returns:
            Tuple of (node_ids, edge_ids) in the complete update chain
        """
        visited_nodes = set()
        visited_edges = set()
        queue = deque([memory_id])

        while queue:
            current_id = queue.popleft()
            if current_id in visited_nodes:
                continue

            visited_nodes.add(current_id)

            # Follow outgoing update edges (what this memory updated)
            for target_id, edge_id in self.outgoing_updates.get(current_id, []):
                if target_id not in visited_nodes:
                    queue.append(target_id)
                visited_edges.add(edge_id)

            # Follow incoming update edges (what updated this memory)
            for source_id, edge_id in self.incoming_updates.get(current_id, []):
                if source_id not in visited_nodes:
                    queue.append(source_id)
                visited_edges.add(edge_id)

        return visited_nodes, visited_edges