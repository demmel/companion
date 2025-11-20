"""
Simple hypergraph implementation for knowledge storage.

This is a minimal dict-based implementation. If we hit performance bottlenecks,
we can rewrite with better data structures. For now, keep it simple.
"""

from typing import List, Dict, Set, Any, Optional
from collections import defaultdict
from datetime import datetime
import uuid

from .interfaces import FactProvenance, FactSource, FactSignature, IKnowledgeGraph, Fact


class SimpleHypergraph(IKnowledgeGraph):
    """
    Minimal hypergraph using dicts and sets for indexing.

    Structure:
    - facts: Dict[fact_id, Fact] - all facts
    - entity_index: Dict[entity_id, Set[fact_id]] - fast entity lookup
    - predicate_index: Dict[predicate, Set[fact_id]] - fast predicate lookup
    """

    def __init__(self):
        self.facts: Dict[str, Fact] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.predicate_index: Dict[str, Set[str]] = defaultdict(set)
        self.signature_index: Dict[FactSignature, str] = {}  # signature -> fact_id

    def add_fact(self, fact: Fact, source_id: Optional[str] = None) -> None:
        """
        Add a fact and update indices.

        If a duplicate fact already exists (same predicate + entities), merge them:
        - Keep the existing fact
        - Add source information from the new fact
        - Merge domain properties (prefer non-null values)

        Args:
            fact: The fact to add
            source_id: Optional source identifier (URL, article title, etc.)
        """
        signature = FactSignature.from_fact(fact)

        # Check if this fact already exists
        if signature in self.signature_index:
            # Merge with existing fact
            existing_id = self.signature_index[signature]
            existing = self.facts[existing_id]

            # Ensure provenance exists
            if existing.provenance is None:
                existing.provenance = FactProvenance()

            # Add source information
            fact_source = FactSource(
                source_id=source_id or fact.id,
                time_period=fact.time_period,
                region=fact.region,
                confidence=fact.confidence,
            )
            existing.provenance.sources.append(fact_source)

            # Merge domain properties (prefer non-null values from new fact)
            if fact.time_period and not existing.time_period:
                existing.time_period = fact.time_period
            if fact.region and not existing.region:
                existing.region = fact.region
            if fact.confidence and not existing.confidence:
                existing.confidence = fact.confidence

            return  # Don't add duplicate

        # New fact - add it
        # Ensure provenance exists and add initial source
        if fact.provenance is None:
            fact.provenance = FactProvenance()

        fact_source = FactSource(
            source_id=source_id or fact.id,
            time_period=fact.time_period,
            region=fact.region,
            confidence=fact.confidence,
        )
        fact.provenance.sources.append(fact_source)

        # Store fact
        self.facts[fact.id] = fact
        self.signature_index[signature] = fact.id

        # Update entity index
        for entity_id in fact.get_all_entities():
            self.entity_index[entity_id].add(fact.id)

        # Update predicate index
        self.predicate_index[fact.predicate].add(fact.id)

    def find_facts_by_entity(self, entity_id: str) -> List[Fact]:
        """Find all facts involving this entity - O(k) where k is number of matching facts"""
        fact_ids = self.entity_index.get(entity_id, set())
        return [self.facts[fid] for fid in fact_ids]

    def find_facts_by_predicate(self, predicate: str) -> List[Fact]:
        """Find all facts with this predicate - O(k) where k is number of matching facts"""
        fact_ids = self.predicate_index.get(predicate, set())
        return [self.facts[fid] for fid in fact_ids]

    def get_all_facts(self) -> List[Fact]:
        """Get all facts - O(n)"""
        return list(self.facts.values())

    def get_all_entities(self) -> List[str]:
        """Get all unique entity IDs - O(m) where m is number of entities"""
        return list(self.entity_index.keys())

    def merge(self, other: IKnowledgeGraph) -> "SimpleHypergraph":
        """
        Merge another graph into a new graph.

        Simple strategy for now: just combine all facts.
        If we see duplicate detection issues, we can add deduplication logic.
        """
        merged = SimpleHypergraph()

        # Add all facts from self
        for fact in self.get_all_facts():
            merged.add_fact(fact)

        # Add all facts from other
        for fact in other.get_all_facts():
            merged.add_fact(fact)

        return merged

    def to_dict(self) -> dict:
        """Serialize to dict"""
        facts_data = []
        for f in self.facts.values():
            fact_dict = {
                "id": f.id,
                "predicate": f.predicate,
                "entities": f.entities,
                "time_period": f.time_period,
                "region": f.region,
                "confidence": f.confidence,
                "timestamp": f.timestamp.isoformat(),
            }
            if f.provenance:
                sources_data = [
                    {
                        "source_id": src.source_id,
                        "time_period": src.time_period,
                        "region": src.region,
                        "confidence": src.confidence,
                        "extracted_at": src.extracted_at.isoformat(),
                    }
                    for src in f.provenance.sources
                ]
                fact_dict["provenance"] = {
                    "source_topic": f.provenance.source_topic,
                    "parent_topic": f.provenance.parent_topic,
                    "graph_index": f.provenance.graph_index,
                    "sources": sources_data,
                }
            facts_data.append(fact_dict)
        return {"facts": facts_data}

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleHypergraph":
        """Deserialize from dict"""
        graph = cls()
        for fact_data in data.get("facts", []):
            provenance = None
            if "provenance" in fact_data:
                prov_data = fact_data["provenance"]
                sources = []
                for src_data in prov_data.get("sources", []):
                    source = FactSource(
                        source_id=src_data["source_id"],
                        time_period=src_data.get("time_period"),
                        region=src_data.get("region"),
                        confidence=src_data.get("confidence"),
                        extracted_at=datetime.fromisoformat(src_data["extracted_at"]),
                    )
                    sources.append(source)

                provenance = FactProvenance(
                    source_topic=prov_data.get("source_topic"),
                    parent_topic=prov_data.get("parent_topic"),
                    graph_index=prov_data.get("graph_index"),
                    sources=sources,
                )

            fact = Fact(
                id=fact_data["id"],
                predicate=fact_data["predicate"],
                entities=fact_data["entities"],
                time_period=fact_data.get("time_period"),
                region=fact_data.get("region"),
                confidence=fact_data.get("confidence"),
                provenance=provenance,
                timestamp=datetime.fromisoformat(fact_data["timestamp"]),
            )
            graph.add_fact(fact)
        return graph

    def __len__(self) -> int:
        """Number of facts in graph"""
        return len(self.facts)

    def __repr__(self) -> str:
        """Readable representation"""
        return f"SimpleHypergraph(facts={len(self.facts)}, entities={len(self.entity_index)})"


def create_fact(
    predicate: str,
    entities: Dict[str, str],
    time_period: Optional[str] = None,
    region: Optional[str] = None,
    confidence: Optional[str] = None,
    provenance: Optional[FactProvenance] = None,
    fact_id: Optional[str] = None,
) -> Fact:
    """
    Convenience function to create facts.

    Args:
        predicate: The relationship type
        entities: Dict of role -> entity_id
        time_period: When this fact occurred
        region: Where this fact occurred
        confidence: Confidence level (high/medium/low)
        provenance: Where this fact came from
        fact_id: Optional custom ID (auto-generated if not provided)

    Returns:
        Fact instance
    """
    if fact_id is None:
        fact_id = f"fact_{uuid.uuid4().hex[:12]}"

    return Fact(
        id=fact_id,
        predicate=predicate,
        entities=entities,
        time_period=time_period,
        region=region,
        confidence=confidence,
        provenance=provenance,
        timestamp=datetime.now(),
    )
