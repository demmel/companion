"""
Evaluation metrics for autonomous research experiments.

Provides comprehensive KPIs for measuring quality, relevance, cost, and structure.
"""

import time
import logging
from typing import List, Dict, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from functools import wraps

from .interfaces import IKnowledgeGraph, Fact

logger = logging.getLogger(__name__)


# ============================================================================
# Cost Tracking
# ============================================================================


@dataclass
class CostMetrics:
    """Track LLM usage and time costs"""

    total_llm_calls: int = 0
    llm_calls_by_operation: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    total_time_seconds: float = 0.0
    time_by_operation: Dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    def record_llm_call(self, operation: str):
        """Record an LLM call"""
        self.total_llm_calls += 1
        self.llm_calls_by_operation[operation] += 1

    def record_time(self, operation: str, duration: float):
        """Record operation time"""
        self.total_time_seconds += duration
        self.time_by_operation[operation] += duration

    def facts_per_llm_call(self, fact_count: int) -> float:
        """Extraction efficiency"""
        return fact_count / self.total_llm_calls if self.total_llm_calls > 0 else 0.0

    def cost_per_fact(self, fact_count: int) -> float:
        """LLM calls needed per fact"""
        return self.total_llm_calls / fact_count if fact_count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total_llm_calls": self.total_llm_calls,
            "llm_calls_by_operation": dict(self.llm_calls_by_operation),
            "total_time_seconds": self.total_time_seconds,
            "time_by_operation": dict(self.time_by_operation),
        }


class CostTracker:
    """Global cost tracker for experiments"""

    def __init__(self):
        self.metrics = CostMetrics()

    def track_llm_call(self, operation: str):
        """Decorator to track LLM calls"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.metrics.record_llm_call(operation)
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start
                self.metrics.record_time(operation, duration)
                return result

            return wrapper

        return decorator

    def reset(self):
        """Reset metrics"""
        self.metrics = CostMetrics()


# Global tracker instance
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker"""
    return _cost_tracker


# ============================================================================
# Fact Quality Metrics
# ============================================================================


@dataclass
class FactQualityMetrics:
    """Measures quality of extracted facts"""

    total_facts: int = 0
    well_formed_facts: int = 0
    unique_entities: int = 0
    total_entity_mentions: int = 0
    unique_predicates: int = 0
    total_entities_in_facts: int = 0
    entity_reuse_count: int = 0  # entities appearing in multiple facts

    @property
    def well_formed_ratio(self) -> float:
        """% of facts that are well-formed"""
        return (
            self.well_formed_facts / self.total_facts if self.total_facts > 0 else 0.0
        )

    @property
    def entity_diversity(self) -> float:
        """Unique entities / total mentions"""
        return (
            self.unique_entities / self.total_entity_mentions
            if self.total_entity_mentions > 0
            else 0.0
        )

    @property
    def predicate_diversity(self) -> float:
        """Unique predicates / total facts"""
        return (
            self.unique_predicates / self.total_facts if self.total_facts > 0 else 0.0
        )

    @property
    def avg_entities_per_fact(self) -> float:
        """Average n-ary richness"""
        return (
            self.total_entities_in_facts / self.total_facts
            if self.total_facts > 0
            else 0.0
        )

    @property
    def entity_reuse_ratio(self) -> float:
        """% entities used in multiple facts (connectivity indicator)"""
        return (
            self.entity_reuse_count / self.unique_entities
            if self.unique_entities > 0
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "total_facts": self.total_facts,
            "well_formed_ratio": self.well_formed_ratio,
            "entity_diversity": self.entity_diversity,
            "predicate_diversity": self.predicate_diversity,
            "avg_entities_per_fact": self.avg_entities_per_fact,
            "entity_reuse_ratio": self.entity_reuse_ratio,
            "unique_entities": self.unique_entities,
            "unique_predicates": self.unique_predicates,
        }


def compute_fact_quality(graph: IKnowledgeGraph) -> FactQualityMetrics:
    """Compute quality metrics for a knowledge graph"""
    metrics = FactQualityMetrics()

    facts = graph.get_all_facts()
    metrics.total_facts = len(facts)

    if metrics.total_facts == 0:
        return metrics

    # Collect entities and predicates
    all_entities = []
    entity_to_facts = defaultdict(set)
    predicates = []

    for fact in facts:
        # Check well-formedness
        if fact.predicate and fact.entities and len(fact.entities) > 0:
            metrics.well_formed_facts += 1

        # Count entities
        fact_entities = fact.get_all_entities()
        metrics.total_entities_in_facts += len(fact_entities)
        all_entities.extend(fact_entities)

        # Track which facts each entity appears in
        for entity in fact_entities:
            entity_to_facts[entity].add(fact.id)

        # Count predicates
        predicates.append(fact.predicate)

    # Compute diversity
    metrics.unique_entities = len(set(all_entities))
    metrics.total_entity_mentions = len(all_entities)
    metrics.unique_predicates = len(set(predicates))

    # Entity reuse (entities in multiple facts)
    metrics.entity_reuse_count = sum(
        1 for facts_set in entity_to_facts.values() if len(facts_set) > 1
    )

    return metrics


# ============================================================================
# Retrieval Quality Metrics
# ============================================================================


@dataclass
class RetrievalQualityMetrics:
    """Measures quality of retrieval results"""

    total_queries: int = 0
    null_results: int = 0
    total_retrieved: int = 0
    query_entity_overlaps: List[float] = field(default_factory=list)
    retrieval_similarities: List[float] = field(default_factory=list)
    predicate_diversities: List[float] = field(default_factory=list)

    @property
    def null_result_ratio(self) -> float:
        """% queries with no results"""
        return self.null_results / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def avg_retrieved_per_query(self) -> float:
        """Average facts retrieved per query"""
        return (
            self.total_retrieved / self.total_queries if self.total_queries > 0 else 0.0
        )

    @property
    def avg_query_entity_overlap(self) -> float:
        """Average entity overlap between query and results"""
        return (
            float(np.mean(self.query_entity_overlaps))
            if self.query_entity_overlaps
            else 0.0
        )

    @property
    def avg_retrieval_similarity(self) -> float:
        """Average similarity score"""
        return (
            float(np.mean(self.retrieval_similarities))
            if self.retrieval_similarities
            else 0.0
        )

    @property
    def avg_predicate_diversity(self) -> float:
        """Average diversity of predicates in results"""
        return (
            float(np.mean(self.predicate_diversities))
            if self.predicate_diversities
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "null_result_ratio": self.null_result_ratio,
            "avg_retrieved_per_query": self.avg_retrieved_per_query,
            "avg_query_entity_overlap": self.avg_query_entity_overlap,
            "avg_predicate_diversity": self.avg_predicate_diversity,
        }


def compute_retrieval_quality(
    queries: List[str],
    retrieved_facts: List[List[Fact]],
    embedder=None,  # Optional sentence transformer for similarity
) -> RetrievalQualityMetrics:
    """Compute retrieval quality metrics"""
    metrics = RetrievalQualityMetrics()
    metrics.total_queries = len(queries)

    for query, facts in zip(queries, retrieved_facts):
        # Count results
        if not facts:
            metrics.null_results += 1
            continue

        metrics.total_retrieved += len(facts)

        # Query entity overlap
        query_tokens = set(query.lower().split())
        total_overlap = 0
        for fact in facts:
            fact_text = " ".join(fact.entities.values()).lower()
            overlap = len(query_tokens & set(fact_text.split()))
            total_overlap += overlap

        avg_overlap = total_overlap / len(facts) if facts else 0.0
        metrics.query_entity_overlaps.append(avg_overlap)

        # Predicate diversity in results
        predicates = [f.predicate for f in facts]
        diversity = len(set(predicates)) / len(predicates) if predicates else 0.0
        metrics.predicate_diversities.append(diversity)

        # Embedding similarity (if embedder provided)
        if embedder:
            try:
                query_emb = embedder.encode(query, convert_to_numpy=True)
                fact_texts = [" ".join(f.entities.values()) for f in facts]
                fact_embs = embedder.encode(fact_texts, convert_to_numpy=True)

                similarities = np.dot(fact_embs, query_emb)
                avg_sim = np.mean(similarities)
                metrics.retrieval_similarities.append(float(avg_sim))
            except:
                pass  # Skip if embedding fails

    return metrics


# ============================================================================
# Graph Structure Metrics
# ============================================================================


@dataclass
class GraphStructureMetrics:
    """Measures graph structural properties"""

    total_facts: int = 0
    total_entities: int = 0
    bridge_entities: int = 0  # entities in multiple topics/contexts
    isolated_facts: int = 0  # facts with no shared entities
    avg_fact_degree: float = 0.0  # avg entities per fact
    avg_entity_degree: float = 0.0  # avg facts per entity
    redundancy_detected: int = 0

    @property
    def bridge_ratio(self) -> float:
        """% entities that bridge multiple contexts"""
        return (
            self.bridge_entities / self.total_entities
            if self.total_entities > 0
            else 0.0
        )

    @property
    def isolation_ratio(self) -> float:
        """% facts that are isolated"""
        return self.isolated_facts / self.total_facts if self.total_facts > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "bridge_ratio": self.bridge_ratio,
            "isolation_ratio": self.isolation_ratio,
            "avg_fact_degree": self.avg_fact_degree,
            "avg_entity_degree": self.avg_entity_degree,
            "redundancy_detected": self.redundancy_detected,
        }


def compute_graph_structure(
    graph: IKnowledgeGraph, topic_metadata: Dict[str, Any]
) -> GraphStructureMetrics:
    """Compute graph structure metrics"""
    metrics = GraphStructureMetrics()

    facts = graph.get_all_facts()
    metrics.total_facts = len(facts)

    if metrics.total_facts == 0:
        return metrics

    # Build entity-fact bipartite graph
    entity_to_facts = defaultdict(set)
    entity_to_topics = defaultdict(set)
    fact_entities_count = []

    for fact in facts:
        entities = fact.get_all_entities()
        fact_entities_count.append(len(entities))

        for entity in entities:
            entity_to_facts[entity].add(fact.id)

            # Track topic if in fact provenance
            if fact.provenance and fact.provenance.source_topic:
                entity_to_topics[entity].add(fact.provenance.source_topic)

    metrics.total_entities = len(entity_to_facts)

    # Avg fact degree (entities per fact)
    metrics.avg_fact_degree = (
        float(np.mean(fact_entities_count)) if fact_entities_count else 0.0
    )

    # Avg entity degree (facts per entity)
    entity_degrees = [len(facts_set) for facts_set in entity_to_facts.values()]
    metrics.avg_entity_degree = (
        float(np.mean(entity_degrees)) if entity_degrees else 0.0
    )

    # Bridge entities (appear in multiple topics)
    metrics.bridge_entities = sum(
        1 for topics in entity_to_topics.values() if len(topics) > 1
    )

    # Isolated facts (no shared entities with other facts)
    for fact in facts:
        entities = fact.get_all_entities()
        # Check if any entity appears in other facts
        is_isolated = all(len(entity_to_facts[e]) == 1 for e in entities)
        if is_isolated:
            metrics.isolated_facts += 1

    # Redundancy detection (same predicate + similar entities)
    seen_signatures = set()
    for fact in facts:
        signature = (fact.predicate, tuple(sorted(fact.entities.values())))
        if signature in seen_signatures:
            metrics.redundancy_detected += 1
        else:
            seen_signatures.add(signature)

    return metrics


# ============================================================================
# Combined Evaluation
# ============================================================================


@dataclass
class ExperimentEvaluation:
    """Complete evaluation of an experiment"""

    fact_quality: FactQualityMetrics
    retrieval_quality: RetrievalQualityMetrics
    graph_structure: GraphStructureMetrics
    cost: CostMetrics

    def to_dict(self) -> dict:
        return {
            "fact_quality": self.fact_quality.to_dict(),
            "retrieval_quality": self.retrieval_quality.to_dict(),
            "graph_structure": self.graph_structure.to_dict(),
            "cost": self.cost.to_dict(),
        }

    def summary_string(self) -> str:
        """Human-readable summary"""
        lines = []
        lines.append("EVALUATION SUMMARY")
        lines.append("=" * 60)

        lines.append("\nFact Quality:")
        lines.append(f"  Well-formed: {self.fact_quality.well_formed_ratio:.1%}")
        lines.append(f"  Entity diversity: {self.fact_quality.entity_diversity:.2f}")
        lines.append(
            f"  Avg entities/fact: {self.fact_quality.avg_entities_per_fact:.1f}"
        )
        lines.append(f"  Entity reuse: {self.fact_quality.entity_reuse_ratio:.1%}")

        lines.append("\nRetrieval Quality:")
        lines.append(f"  Null results: {self.retrieval_quality.null_result_ratio:.1%}")
        lines.append(
            f"  Avg retrieved/query: {self.retrieval_quality.avg_retrieved_per_query:.1f}"
        )
        lines.append(
            f"  Entity overlap: {self.retrieval_quality.avg_query_entity_overlap:.2f}"
        )
        lines.append(
            f"  Predicate diversity: {self.retrieval_quality.avg_predicate_diversity:.2f}"
        )

        lines.append("\nGraph Structure:")
        lines.append(f"  Total facts: {self.graph_structure.total_facts}")
        lines.append(f"  Total entities: {self.graph_structure.total_entities}")
        lines.append(f"  Bridge entities: {self.graph_structure.bridge_ratio:.1%}")
        lines.append(f"  Isolated facts: {self.graph_structure.isolation_ratio:.1%}")
        lines.append(
            f"  Redundancy: {self.graph_structure.redundancy_detected} duplicates"
        )

        lines.append("\nCost:")
        lines.append(f"  Total LLM calls: {self.cost.total_llm_calls}")
        lines.append(f"  Total time: {self.cost.total_time_seconds:.1f}s")
        if self.fact_quality.total_facts > 0:
            lines.append(
                f"  Facts/call: {self.cost.facts_per_llm_call(self.fact_quality.total_facts):.2f}"
            )
            lines.append(
                f"  Calls/fact: {self.cost.cost_per_fact(self.fact_quality.total_facts):.2f}"
            )

        return "\n".join(lines)
