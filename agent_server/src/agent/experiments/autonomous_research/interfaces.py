"""
Component interfaces for autonomous research experiment.

These interfaces define contracts for components that can be swapped out
or rewritten as we learn what works. Implementations are meant to be simple
and expendable - we'll rewrite them when we hit bottlenecks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass(frozen=True)
class FactSignature:
    """Unique signature for fact deduplication based on predicate and entities"""

    predicate: str
    entities: tuple[str, ...]  # sorted entity IDs

    @classmethod
    def from_fact(cls, fact: "Fact") -> "FactSignature":
        return cls(
            predicate=fact.predicate, entities=tuple(sorted(fact.entities.values()))
        )


@dataclass
class IntegrationMetadata:
    """Metadata for graph integration"""

    topics: List[str]


@dataclass
class FactSource:
    """Information about this fact from a single source"""

    source_id: str  # URL, article title, or original fact ID
    time_period: Optional[str] = None
    region: Optional[str] = None
    confidence: Optional[str] = None
    extracted_at: datetime = field(default_factory=datetime.now)


@dataclass
class FactProvenance:
    """Provenance tracking: where did this fact come from"""

    source_topic: Optional[str] = None
    parent_topic: Optional[str] = None
    graph_index: Optional[int] = None
    sources: List[FactSource] = field(
        default_factory=list
    )  # All sources that mentioned this fact

    @property
    def source_count(self) -> int:
        """Number of independent sources for this fact"""
        return len(self.sources)


@dataclass
class Fact:
    """
    An n-ary fact extracted from research.

    A fact is a hyperedge connecting multiple entities through a predicate.
    Example: Trade(Byzantine Empire, Venice, silk, 10th century)
    """

    id: str
    predicate: str  # The relationship: "traded_with", "ruled_by", "invented", etc.
    entities: Dict[
        str, str
    ]  # role -> entity_id: {"trader": "Byzantine Empire", "partner": "Venice", "good": "silk"}
    # Domain properties (part of the fact itself)
    time_period: Optional[str] = None
    region: Optional[str] = None
    confidence: Optional[str] = None
    # Actual metadata
    provenance: Optional[FactProvenance] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def involves_entity(self, entity_id: str) -> bool:
        """Check if this fact involves a specific entity"""
        return entity_id in self.entities.values()

    def get_all_entities(self) -> List[str]:
        """Get all entity IDs in this fact"""
        return list(self.entities.values())


class IKnowledgeGraph(ABC):
    """
    Interface for knowledge graph implementations.

    The graph stores facts (hyperedges) and provides efficient lookup.
    Implementations can be rewritten if we hit performance issues.
    """

    @abstractmethod
    def add_fact(self, fact: Fact) -> None:
        """Add a fact to the graph"""
        pass

    @abstractmethod
    def find_facts_by_entity(self, entity_id: str) -> List[Fact]:
        """Find all facts that involve a specific entity"""
        pass

    @abstractmethod
    def find_facts_by_predicate(self, predicate: str) -> List[Fact]:
        """Find all facts with a specific predicate/relationship type"""
        pass

    @abstractmethod
    def get_all_facts(self) -> List[Fact]:
        """Get all facts in the graph"""
        pass

    @abstractmethod
    def get_all_entities(self) -> List[str]:
        """Get all unique entity IDs in the graph"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of facts in the graph"""
        pass

    @abstractmethod
    def merge(self, other: "IKnowledgeGraph") -> "IKnowledgeGraph":
        """
        Merge another graph into a new graph.
        Returns a new graph containing facts from both.
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to dict for persistence"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "IKnowledgeGraph":
        """Deserialize from dict"""
        pass


class IFactExtractor(ABC):
    """
    Interface for extracting structured facts from unstructured text.

    Takes research text and extracts n-ary facts. This is where LLM calls happen.
    Can be rewritten with different prompts/strategies if extraction quality is poor.
    """

    @abstractmethod
    def extract_facts(self, text: str, context: Optional[str] = None) -> List[Fact]:
        """
        Extract structured facts from text.

        Args:
            text: The text to extract facts from (article, web page, etc.)
            context: Optional research context (topic, previous findings)

        Returns:
            List of extracted facts
        """
        pass


class IResearchOrchestrator(ABC):
    """
    Interface for orchestrating multi-cycle autonomous research.

    Coordinates search → read → extract → think cycles to build knowledge graphs.
    Can be rewritten with different strategies (depth-first, breadth-first, etc.)
    """

    @abstractmethod
    def research_topic(
        self, topic: str, depth: int = 3, initial_questions: Optional[List[str]] = None
    ) -> IKnowledgeGraph:
        """
        Conduct autonomous research on a topic.

        Args:
            topic: The topic to research
            depth: Number of research cycles (search → read → extract iterations)
            initial_questions: Optional starting questions to guide research

        Returns:
            Knowledge graph built from research
        """
        pass


class IGraphIntegrator(ABC):
    """
    Interface for integrating multiple knowledge graphs.

    Experiments with different strategies: isolated, bridged, merged, hierarchical.
    Can be rewritten based on what we learn about cross-topic connections.
    """

    @abstractmethod
    def integrate(
        self, graphs: List[IKnowledgeGraph], metadata: Optional[Dict[str, Any]] = None
    ) -> IKnowledgeGraph:
        """
        Integrate multiple graphs using this strategy.

        Args:
            graphs: List of knowledge graphs to integrate
            metadata: Optional metadata about graphs (topics, timestamps, etc.)

        Returns:
            Integrated knowledge graph
        """
        pass


class IRetriever(ABC):
    """
    Interface for retrieving relevant facts from a knowledge graph.

    Given a query, returns the most relevant facts. Can experiment with:
    - Embedding similarity
    - Graph traversal
    - Hybrid approaches
    """

    @abstractmethod
    def retrieve(
        self, query: str, graph: IKnowledgeGraph, top_k: int = 10
    ) -> List[Fact]:
        """
        Retrieve relevant facts for a query.

        Args:
            query: The question or query
            graph: Knowledge graph to search
            top_k: Maximum number of facts to return

        Returns:
            List of relevant facts, ordered by relevance
        """
        pass
