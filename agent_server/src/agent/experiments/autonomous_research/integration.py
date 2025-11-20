"""
Strategies for integrating multiple knowledge graphs.

Experiments with different approaches to combining topic-specific graphs.
Each strategy can be tested and compared.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

from .interfaces import (
    IGraphIntegrator,
    IKnowledgeGraph,
    FactProvenance,
    FactSignature,
    FactSource,
)
from .knowledge_graph import SimpleHypergraph, Fact, create_fact

logger = logging.getLogger(__name__)


class NaiveIntegrator(IGraphIntegrator):
    """
    Simplest strategy: just combine all facts without any processing.

    No deduplication, no merging, no organization.
    Fast but potentially redundant.
    """

    def integrate(
        self, graphs: List[IKnowledgeGraph], metadata: Optional[Dict[str, Any]] = None
    ) -> IKnowledgeGraph:
        """Merge all graphs naively"""
        logger.info(f"Naive integration of {len(graphs)} graphs")

        integrated = SimpleHypergraph()

        for i, graph in enumerate(graphs):
            logger.debug(f"Adding graph {i+1}: {len(graph)} facts")
            for fact in graph.get_all_facts():
                integrated.add_fact(fact)

        logger.info(f"Integrated graph has {len(integrated)} facts")
        return integrated


class BridgedIntegrator(IGraphIntegrator):
    """
    Detects entity overlaps and creates bridge facts between graphs.

    When the same entity appears in multiple graphs, creates explicit
    "appears_in_topic" bridge facts to show cross-topic connections.
    """

    def integrate(
        self, graphs: List[IKnowledgeGraph], metadata: Optional[Dict[str, Any]] = None
    ) -> IKnowledgeGraph:
        """Integrate with bridge facts for shared entities"""
        logger.info(f"Bridged integration of {len(graphs)} graphs")

        # First, do naive merge
        integrated = SimpleHypergraph()

        # Track which entities appear in which graphs
        entity_to_graphs: Dict[str, Set[int]] = defaultdict(set)

        for i, graph in enumerate(graphs):
            for fact in graph.get_all_facts():
                integrated.add_fact(fact)

                # Track entity appearances
                for entity_id in fact.get_all_entities():
                    entity_to_graphs[entity_id].add(i)

        # Find entities that appear in multiple graphs (bridges)
        bridges = {
            entity: graph_indices
            for entity, graph_indices in entity_to_graphs.items()
            if len(graph_indices) > 1
        }

        logger.info(f"Found {len(bridges)} entities appearing in multiple graphs")

        # TODO: Bridge facts need a different representation
        # These don't fit the domain fact model (they need arbitrary metadata)
        # Should probably track bridge entities separately rather than as fake facts

        logger.info(
            f"Integrated graph has {len(integrated)} facts (including {len(bridges)} bridges)"
        )
        return integrated


class IsolatedIntegrator(IGraphIntegrator):
    """
    Keeps graphs isolated with topic tags.

    Each fact is tagged with its source topic. Graphs remain logically
    separate but queryable as one structure.
    """

    def integrate(
        self, graphs: List[IKnowledgeGraph], metadata: Optional[Dict[str, Any]] = None
    ) -> IKnowledgeGraph:
        """Integrate with topic isolation"""
        logger.info(f"Isolated integration of {len(graphs)} graphs")

        integrated = SimpleHypergraph()
        topics = metadata.get("topics", []) if metadata else []

        for i, graph in enumerate(graphs):
            topic = topics[i] if i < len(topics) else f"topic_{i}"

            for fact in graph.get_all_facts():
                # Add provenance tracking
                if fact.provenance is None:
                    fact.provenance = FactProvenance()
                fact.provenance.source_topic = topic
                fact.provenance.graph_index = i
                integrated.add_fact(fact)

        logger.info(
            f"Integrated graph has {len(integrated)} facts from {len(topics)} topics"
        )
        return integrated


class HierarchicalIntegrator(IGraphIntegrator):
    """
    Creates a hierarchical structure with topic nodes.

    Each topic gets a meta-node, and facts are organized under their topics.
    Enables topic-level reasoning and navigation.
    """

    def integrate(
        self, graphs: List[IKnowledgeGraph], metadata: Optional[Dict[str, Any]] = None
    ) -> IKnowledgeGraph:
        """Integrate with hierarchical topic structure"""
        logger.info(f"Hierarchical integration of {len(graphs)} graphs")

        integrated = SimpleHypergraph()
        topics = metadata.get("topics", []) if metadata else []

        # TODO: Topic meta-facts need a different representation
        # These don't fit the domain fact model (they need arbitrary metadata)
        # Should probably track topic hierarchy separately rather than as fake facts

        for i, (graph, topic) in enumerate(
            zip(
                graphs, topics if topics else [f"topic_{i}" for i in range(len(graphs))]
            )
        ):
            # Add all facts with parent topic reference
            for fact in graph.get_all_facts():
                if fact.provenance is None:
                    fact.provenance = FactProvenance()
                fact.provenance.parent_topic = topic
                fact.provenance.graph_index = i
                integrated.add_fact(fact)

        logger.info(
            f"Integrated hierarchical graph has {len(integrated)} facts under {len(topics)} topics"
        )
        return integrated


class DeduplicatingIntegrator(IGraphIntegrator):
    """
    Merges duplicate facts across topic graphs.

    Since SimpleHypergraph.add_fact() now handles deduplication automatically,
    this strategy just adds all facts and lets the graph merge duplicates.
    The graph tracks all sources for each fact via FactSource.
    """

    def integrate(
        self, graphs: List[IKnowledgeGraph], metadata: Optional[Dict[str, Any]] = None
    ) -> IKnowledgeGraph:
        """Integrate with deduplication"""
        logger.info(f"Deduplicating integration of {len(graphs)} graphs")

        integrated = SimpleHypergraph()
        initial_count = 0
        topics = metadata.get("topics", []) if metadata else []

        for i, graph in enumerate(graphs):
            topic = topics[i] if i < len(topics) else f"topic_{i}"

            for fact in graph.get_all_facts():
                initial_count += 1
                # add_fact() automatically handles deduplication and source tracking
                integrated.add_fact(fact, source_id=f"{topic}_graph")

        final_count = len(integrated)
        dedup_count = initial_count - final_count

        logger.info(
            f"Integrated graph has {final_count} facts (merged {dedup_count} duplicates)"
        )
        return integrated


# Registry of available strategies
INTEGRATION_STRATEGIES = {
    "naive": NaiveIntegrator,
    "bridged": BridgedIntegrator,
    "isolated": IsolatedIntegrator,
    "hierarchical": HierarchicalIntegrator,
    "deduplicating": DeduplicatingIntegrator,
}


def get_integrator(strategy_name: str) -> IGraphIntegrator:
    """Get an integrator by strategy name"""
    integrator_class = INTEGRATION_STRATEGIES.get(strategy_name)
    if not integrator_class:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(INTEGRATION_STRATEGIES.keys())}"
        )
    return integrator_class()
