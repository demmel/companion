"""Data models for query classification experiment."""

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of queries that require different retrieval strategies."""

    CURRENT_STATE = "current_state"  # Current value of an attribute
    HISTORY = "history"  # Past events or changes over time
    ENTITY_OVERVIEW = "entity_overview"  # Everything known about an entity
    TEMPORAL = "temporal"  # Time-bounded queries
    CONTINUITY = "continuity"  # Following up on ongoing situations
    PROACTIVE_CONTEXT = "proactive_context"  # Agent needs context for user's message
    NO_RETRIEVAL = "no_retrieval"  # No retrieval needed


# Query type descriptions for prompts
QUERY_TYPE_DESCRIPTIONS = {
    QueryType.CURRENT_STATE: (
        "Asking for the current/most recent value of an attribute. "
        "Examples: 'What is David wearing?', 'Where does Sarah work?', "
        "'What is my dog's name?'"
    ),
    QueryType.HISTORY: (
        "Asking about past events, changes over time, or episodic memories. "
        "Examples: 'What has David worn?', 'Remember when we talked about X?', "
        "'What did I do yesterday?'"
    ),
    QueryType.ENTITY_OVERVIEW: (
        "Asking for everything known about an entity. "
        "Examples: 'What do I know about Sarah?', 'Tell me about my dog', "
        "'Who is my friend Mark?'"
    ),
    QueryType.TEMPORAL: (
        "Time-bounded queries with specific time references. "
        "Examples: 'What happened yesterday?', 'This morning...', "
        "'When I was stressed', 'Last week's meetings'"
    ),
    QueryType.CONTINUITY: (
        "Following up on ongoing situations or recent topics. "
        "Examples: 'How did the interview go?', 'Any update on that?', "
        "'Did they respond?'"
    ),
    QueryType.PROACTIVE_CONTEXT: (
        "Not a direct question but mentions entities/topics that need context. "
        "Examples: User says 'I saw Sarah at the coffee shop' - need Sarah's context. "
        "User mentions 'the project' - need project background."
    ),
    QueryType.NO_RETRIEVAL: (
        "Queries that don't need memory retrieval at all. "
        "Examples: 'Hello!', 'What time is it?', 'Thanks', "
        "'Can you help me write code?'"
    ),
}


class LabeledQuery(BaseModel):
    """A query with its ground truth classification."""

    query: str = Field(description="The query text")
    query_type: QueryType = Field(description="The correct classification")
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities mentioned in the query",
    )
    attributes: list[str] = Field(
        default_factory=list,
        description="Attributes being asked about (e.g., 'clothing', 'mood')",
    )
    time_reference: str = Field(
        default="",
        description="Time reference if present (e.g., 'yesterday', 'this morning')",
    )
    is_proactive: bool = Field(
        default=False,
        description="Whether this is a proactive context case (user statement vs question)",
    )
    reasoning: str = Field(
        default="",
        description="Why this query has this classification",
    )
    source: str = Field(
        default="seed",
        description="Source of the query (e.g., 'seed', 'mistral', 'claude', 'independent')",
    )


class ClassificationResult(BaseModel):
    """Result of classifying a query."""

    query: str = Field(description="The input query")
    predicted_type: QueryType = Field(description="The predicted query type")
    confidence: float = Field(
        description="Confidence in the prediction (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        default="",
        description="Explanation for the classification",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Time taken to classify in milliseconds",
    )


class LLMClassificationResponse(BaseModel):
    """Response model for LLM-based classification."""

    query_type: str = Field(
        description=(
            "Type of query: current_state, history, entity_overview, "
            "temporal, continuity, proactive_context, or no_retrieval"
        )
    )
    confidence: float = Field(
        description="Confidence in classification from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(description="Brief explanation for the classification")


# =============================================================================
# Extraction Models (for message + context → queries with types)
# =============================================================================


class ExtractedQuery(BaseModel):
    """A query extracted from user input with its type for retrieval routing.

    This is the key output of the extraction process - a query ready to be
    routed to the appropriate retrieval strategy.
    """

    query_text: str = Field(description="The query to run against memory")
    query_type: QueryType = Field(description="The type of query for routing")
    reference: str = Field(
        description="What triggered this query (entity name, topic, pronoun, etc.)"
    )
    reasoning: str = Field(
        description="Why this query would help provide context for the response"
    )


class ExtractionInput(BaseModel):
    """Input for query extraction.

    The extraction process takes a user message along with recent conversation
    context to understand implicit references and determine what queries would
    help provide useful context.
    """

    message: str = Field(description="The current user message")
    context: list[str] = Field(
        description="Recent conversation history (context window)"
    )


class ExtractionResult(BaseModel):
    """Result of query extraction.

    Contains the extracted queries along with a summary of why these queries
    matter for understanding the user's message.
    """

    queries: list[ExtractedQuery] = Field(
        default_factory=list,
        description="Queries extracted from the user message",
    )
    context_summary: str = Field(
        default="",
        description="Brief summary of why these queries matter for the response",
    )


class LLMExtractionResponse(BaseModel):
    """Response model for LLM-based query extraction.

    This is the structured output format that the LLM will produce.
    """

    queries: list[dict[str, str]] = Field(
        description=(
            "List of extracted queries. Each dict has keys: "
            "'query_text', 'query_type', 'reference', 'reasoning'"
        )
    )
    context_summary: str = Field(
        description="Brief summary of why these queries matter"
    )
    no_retrieval_needed: bool = Field(
        default=False,
        description="True if no memory retrieval is needed for this message",
    )


# =============================================================================
# Extraction Dataset Models
# =============================================================================


class LabeledExtractionExample(BaseModel):
    """A labeled example for training/evaluating query extraction.

    Each example contains a user message with context and the expected
    queries that should be extracted.
    """

    message: str = Field(description="The user message to extract queries from")
    context: list[str] = Field(
        default_factory=list,
        description="Recent conversation context",
    )
    expected_queries: list[ExtractedQuery] = Field(
        default_factory=list,
        description="The queries that should be extracted",
    )
    notes: str = Field(
        default="",
        description="Notes about this example (edge case, difficulty, etc.)",
    )


@dataclass
class ExtractionDataset:
    """A dataset of labeled extraction examples."""

    examples: list[LabeledExtractionExample]
    name: str = ""
    description: str = ""

    def get_query_type_distribution(self) -> dict[QueryType, int]:
        """Get the distribution of query types across all examples."""
        distribution: dict[QueryType, int] = {}
        for example in self.examples:
            for query in example.expected_queries:
                distribution[query.query_type] = (
                    distribution.get(query.query_type, 0) + 1
                )
        return distribution


# =============================================================================
# Evaluation Models
# =============================================================================


@dataclass
class EvaluationMetrics:
    """Metrics from evaluating a classifier."""

    accuracy: float
    per_class_precision: dict[QueryType, float] = field(default_factory=dict)
    per_class_recall: dict[QueryType, float] = field(default_factory=dict)
    per_class_f1: dict[QueryType, float] = field(default_factory=dict)
    confusion_matrix: dict[tuple[QueryType, QueryType], int] = field(
        default_factory=dict
    )
    total_samples: int = 0
    correct_predictions: int = 0
    avg_latency_ms: float = 0.0
    total_tokens_used: int = 0


@dataclass
class Dataset:
    """A dataset of labeled queries."""

    queries: list[LabeledQuery]
    name: str = ""
    description: str = ""

    def get_distribution(self) -> dict[QueryType, int]:
        """Get the distribution of query types."""
        distribution: dict[QueryType, int] = {}
        for query in self.queries:
            distribution[query.query_type] = distribution.get(query.query_type, 0) + 1
        return distribution

    def split(
        self, train_ratio: float = 0.8
    ) -> tuple["Dataset", "Dataset"]:
        """Split dataset into train and test sets."""
        import random

        # Group by type to ensure stratified split
        by_type: dict[QueryType, list[LabeledQuery]] = {}
        for query in self.queries:
            if query.query_type not in by_type:
                by_type[query.query_type] = []
            by_type[query.query_type].append(query)

        train_queries: list[LabeledQuery] = []
        test_queries: list[LabeledQuery] = []

        for query_type, queries in by_type.items():
            random.shuffle(queries)
            split_idx = int(len(queries) * train_ratio)
            train_queries.extend(queries[:split_idx])
            test_queries.extend(queries[split_idx:])

        return (
            Dataset(queries=train_queries, name=f"{self.name}_train"),
            Dataset(queries=test_queries, name=f"{self.name}_test"),
        )
