"""Data models for the dreams experiment."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TraversalStrategy(str, Enum):
    """How to move through the memory graph."""

    RANDOM_JUMP = "random_jump"  # Teleports to random unvisited memory (ignores edges)
    RECENCY_WEIGHTED = (
        "recency_weighted"  # Biased toward recent memories (ignores edges)
    )
    SEMANTIC_DRIFT = "semantic_drift"  # Moves to most similar memory (ignores edges)
    CONTRAST_SEEKING = (
        "contrast_seeking"  # Moves to most different memory (ignores edges)
    )
    EDGE_FOLLOWING = "edge_following"  # Actually follows graph edges (true walk)


class NarrativeStyle(str, Enum):
    """How to generate dream narrative text."""

    FRAGMENT = "fragment"
    STREAM = "stream"
    POETIC = "poetic"
    SENSORY = "sensory"


class SeedSelection(str, Enum):
    """How to pick the starting memory for a dream."""

    RANDOM = "random"
    RECENT = "recent"
    EMOTIONAL = "emotional"
    UNPROCESSED = "unprocessed"


class DreamMode(str, Enum):
    """Purpose-driven dream types."""

    TODAY = "today"  # Consolidate memories since last sleep
    BIZARRE = "bizarre"  # Surreal contrast-seeking dreams
    CONNECT = "connect"  # Find connections between memories, create edges


@dataclass
class DiscoveredConnection:
    """A connection discovered during a Connect dream."""

    source_id: str
    target_id: str
    edge_type: str
    reasoning: str


@dataclass
class DreamConfig:
    """Configuration for dream generation."""

    seed_selection: SeedSelection
    traversal_strategy: TraversalStrategy
    depth: int
    narrative_style: NarrativeStyle

    def __str__(self) -> str:
        return (
            f"DreamConfig(seed={self.seed_selection.value}, "
            f"traversal={self.traversal_strategy.value}, "
            f"depth={self.depth}, style={self.narrative_style.value})"
        )


@dataclass
class Dream:
    """A generated dream experience."""

    seed_memory_id: str
    traversal_path: list[str]
    edges_used: list[str]
    narrative: str
    duration_memories: int
    themes_emerged: list[str]
    config: DreamConfig
    created_at: datetime = field(default_factory=datetime.now)
    mode: DreamMode | None = None  # Optional: which dream mode was used
    discovered_connections: list[DiscoveredConnection] = field(default_factory=list)

    def __str__(self) -> str:
        mode_str = f", mode={self.mode.value}" if self.mode else ""
        connections_str = (
            f", connections={len(self.discovered_connections)}"
            if self.discovered_connections
            else ""
        )
        return (
            f"Dream(seed={self.seed_memory_id[:8]}..., "
            f"depth={self.duration_memories}, "
            f"themes={self.themes_emerged}{mode_str}{connections_str})"
        )


@dataclass
class DreamEvaluation:
    """Evaluation scores for a dream."""

    dream: Dream
    dream_like_quality: int  # 1-5
    interestingness: int  # 1-5
    notes: str

    def __str__(self) -> str:
        return (
            f"Evaluation(dream_like={self.dream_like_quality}/5, "
            f"interesting={self.interestingness}/5)"
        )


@dataclass
class ExperimentResult:
    """Results from running an experiment."""

    experiment_name: str
    dreams: list[Dream]
    evaluations: list[DreamEvaluation]
    summary: str
    recommendations: list[str] = field(default_factory=list)
