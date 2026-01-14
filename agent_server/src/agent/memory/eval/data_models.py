"""Data models for memory evaluation."""

from typing import Literal

from pydantic import BaseModel, Field

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.memory.memory import MemoryQueries


class EvalScenario(BaseModel):
    """A single evaluation scenario for memory retrieval."""

    scenario_id: str = Field(description="Unique identifier for this scenario")
    name: str = Field(description="Human-readable name for the scenario")
    description: str = Field(description="What this scenario tests")

    # Conversation history to replay through store()
    trigger_history: list[TriggerHistoryEntry] = Field(
        description="Trigger entries to replay through memory.store()"
    )

    # Query to test
    test_query: MemoryQueries = Field(
        description="The query to run after replaying the conversation"
    )

    # Ground truth - natural language descriptions of expected information
    expected_information: list[str] = Field(
        description="Natural language descriptions of what should be retrieved"
    )


class Judgment(BaseModel):
    """LLM judgment on whether expected information is present."""

    expected_item: str = Field(description="The expected information being judged")
    is_present: bool = Field(description="Whether the information was found")
    reasoning: str = Field(description="Explanation for the judgment")
    confidence: float = Field(
        description="Confidence in the judgment (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


class EvalResult(BaseModel):
    """Result of evaluating a single scenario."""

    scenario_id: str = Field(description="ID of the evaluated scenario")
    memory_implementation: str = Field(description="Name of the memory implementation tested")

    # Raw output from memory.query()
    output: str = Field(description="The string output from memory.query()")

    # LLM judgments for each expected item
    judgments: list[Judgment] = Field(description="Judgments for each expected item")

    # Computed metrics
    recall: float = Field(
        description="Fraction of expected items that were present",
        ge=0.0,
        le=1.0,
    )

    # Performance
    retrieval_time_ms: float = Field(description="Time to execute query() in milliseconds")


class EvalRun(BaseModel):
    """Results from a complete evaluation run."""

    run_id: str = Field(description="Unique identifier for this run")
    timestamp: str = Field(description="ISO timestamp of when the run started")

    # Which implementations were tested
    memory_implementations: list[str] = Field(description="Names of implementations tested")

    # Results per scenario per implementation
    results: list[EvalResult] = Field(description="All evaluation results")

    # Aggregate metrics per implementation
    aggregate_recall: dict[str, float] = Field(
        description="Mean recall per implementation"
    )
    aggregate_time_ms: dict[str, float] = Field(
        description="Mean retrieval time per implementation"
    )
