"""Data models for memory extraction experiment."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class FactType(str, Enum):
    """Types of facts that can be extracted."""

    PREFERENCE = "preference"
    EVENT = "event"
    RELATIONSHIP = "relationship"
    STATE = "state"
    FACT = "fact"
    QUESTION = "question"
    APPEARANCE = "appearance"
    ENVIRONMENT = "environment"
    EMOTION = "emotion"
    VALUE = "value"


class AnnotationLabel(str, Enum):
    """Labels for fact annotation."""

    CORRECT = "correct"
    HALLUCINATED = "hallucinated"
    INFERRED = "inferred"


@dataclass
class ExtractedFact:
    """A single extracted fact from a memory."""

    content: str
    fact_type: FactType
    confidence: float
    entities: list[str]
    source_memory_id: str


@dataclass
class ExtractionResult:
    """All extractions from a single memory."""

    memory_id: str
    original_content: str
    facts: list[ExtractedFact]
    summary: str
    compression_ratio: float


@dataclass
class FactAnnotation:
    """Annotation of a single fact."""

    fact: ExtractedFact
    label: AnnotationLabel
    notes: str


@dataclass
class AnnotationResult:
    """Collection of annotations for an extraction."""

    extraction: ExtractionResult
    annotations: list[FactAnnotation]
    omissions: list[str]

    @property
    def correct_count(self) -> int:
        return sum(1 for a in self.annotations if a.label == AnnotationLabel.CORRECT)

    @property
    def hallucinated_count(self) -> int:
        return sum(
            1 for a in self.annotations if a.label == AnnotationLabel.HALLUCINATED
        )

    @property
    def inferred_count(self) -> int:
        return sum(1 for a in self.annotations if a.label == AnnotationLabel.INFERRED)

    @property
    def hallucination_rate(self) -> float:
        if not self.annotations:
            return 0.0
        return self.hallucinated_count / len(self.annotations)

    @property
    def accuracy_rate(self) -> float:
        if not self.annotations:
            return 0.0
        return self.correct_count / len(self.annotations)


@dataclass
class SearchResult:
    """Result from a memory search."""

    content: str
    score: float
    source_id: str
    source_type: str  # "raw" or "extracted"


@dataclass
class MemorySample:
    """A sample memory for extraction experiments."""

    memory_id: str
    content: str
    source_type: str  # "compressed_summary", "trigger_content", "response"
    timestamp: str


# Pydantic models for LLM responses


class ExtractedFactResponse(BaseModel):
    """LLM response for a single extracted fact."""

    content: str = Field(
        description="The extracted fact as a clear, searchable statement"
    )
    fact_type: str = Field(
        description="Type of fact: preference, event, relationship, state, fact, question, appearance, environment, emotion, value"
    )
    confidence: float = Field(
        description="Confidence in this extraction from 0.0 to 1.0", ge=0.0, le=1.0
    )
    entities: List[str] = Field(description="People or things mentioned in this fact")


class FactListExtractionResponse(BaseModel):
    """LLM response for Approach A: fact list extraction."""

    facts: List[ExtractedFactResponse] = Field(
        description="List of all facts extracted from the memory"
    )
    summary: str = Field(description="One-sentence summary of the memory")


class StructuredExtractionResponse(BaseModel):
    """LLM response for Approach B: structured extraction by category."""

    people: List[str] = Field(
        description="Facts about people mentioned and what we learn about them"
    )
    events: List[str] = Field(description="Events or plans discussed")
    preferences: List[str] = Field(description="Preferences or opinions expressed")
    emotions: List[str] = Field(description="Emotional states or moods")
    questions: List[str] = Field(description="Questions asked or answered")
    summary: str = Field(description="One-sentence summary of the memory")


class QueryAnswer(BaseModel):
    """A question-answer pair for Approach C."""

    question: str = Field(description="A question this memory could answer")
    answer: str = Field(description="The concise answer based on the memory")


class QueryFocusedExtractionResponse(BaseModel):
    """LLM response for Approach C: query-focused extraction."""

    qa_pairs: List[QueryAnswer] = Field(
        description="Questions this memory could answer, with their answers"
    )
    summary: str = Field(description="One-sentence summary of the memory")


class EntityInfo(BaseModel):
    """Information about a single entity for Approach D."""

    entity_name: str = Field(description="Name of the person or entity")
    learned_facts: List[str] = Field(description="What we learn about them")
    actions: List[str] = Field(description="What they said or did")
    relationships: List[str] = Field(description="How they relate to others")


class EntityCentricExtractionResponse(BaseModel):
    """LLM response for Approach D: entity-centric extraction."""

    entities: List[EntityInfo] = Field(
        description="Information extracted for each person/entity mentioned"
    )
    summary: str = Field(description="One-sentence summary of the memory")


class MinimalExtractionResponse(BaseModel):
    """LLM response for Approach E: minimal extraction."""

    most_important_fact: str = Field(
        description="The single most important fact in 10 words or less"
    )
    fact_type: str = Field(description="Type of this fact")


class AnnotationResponse(BaseModel):
    """LLM response for fact annotation."""

    label: str = Field(
        description="Label: 'correct' if fact is accurate, 'hallucinated' if not supported, 'inferred' if reasonable inference"
    )
    reasoning: str = Field(description="Brief explanation for the label")


class OmissionsResponse(BaseModel):
    """LLM response for finding omissions."""

    omitted_facts: List[str] = Field(
        description="Important facts from the original that were not extracted"
    )


class TestQueryGenerationResponse(BaseModel):
    """LLM response for generating test queries from a memory."""

    queries: List[str] = Field(
        description="2-3 questions that someone might ask that this memory would answer"
    )
