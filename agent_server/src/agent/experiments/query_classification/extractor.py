"""Query extraction from user messages with context.

This module implements the core extraction logic that takes a user message
along with conversation context and extracts queries with their types for
retrieval routing.

The key insight is that query type classification should happen AS PART OF
extraction, not as a separate step. When we see "I saw Sarah at the coffee
shop", we need to:
1. Detect that "Sarah" is a reference needing context
2. Determine that we need an entity_overview query for Sarah
3. Possibly extract additional queries based on context

This is fundamentally different from standalone query classification.
"""

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from .models import (
    ExtractedQuery,
    ExtractionInput,
    ExtractionResult,
    QueryType,
    QUERY_TYPE_DESCRIPTIONS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Extraction Prompt
# =============================================================================

EXTRACTION_PROMPT = """You are analyzing a user message to extract queries that will help provide relevant context from memory.

USER MESSAGE:
{message}

RECENT CONVERSATION:
{context}

Your task is to identify what information from memory would help respond to this message effectively.

QUERY TYPES (choose the most appropriate for each extracted query):
- current_state: Need the CURRENT value of something (e.g., "What is Sarah's job?" when user mentions Sarah)
- history: Need past events or how something changed over time (e.g., past conversations about a topic)
- entity_overview: Need comprehensive info about a person/thing (e.g., when a new person is mentioned)
- temporal: Need info from a specific time period (e.g., "yesterday", "last week")
- continuity: Need to follow up on an ongoing situation (e.g., when user says "the interview")
- proactive_context: Need general context for mentioned entities/topics
- no_retrieval: No memory retrieval needed (greetings, math, general help)

EXTRACTION GUIDELINES:
1. Look for explicit references (names, specific topics, events)
2. Look for implicit references ("my mom", "the project", "that thing")
3. Look for pronouns that refer to known entities
4. Consider what context would make the response more personalized
5. Don't extract queries for things that don't need memory (greetings, general questions)
6. For each reference, choose the query type that best matches what we need to retrieve

EXAMPLES:

Message: "I saw Sarah at the coffee shop, she looked tired"
Context: ["Earlier we discussed Sarah's new job", "She mentioned being stressed"]
→ Extract:
  - query_text: "Sarah", type: entity_overview, reference: "Sarah", reasoning: "Need Sarah's background for context"
  - query_text: "Sarah current mood and energy", type: current_state, reference: "looked tired", reasoning: "User noting tiredness suggests tracking her state"
  - query_text: "Sarah's job stress", type: continuity, reference: "looked tired + job context", reasoning: "Follow up on ongoing job stress situation"

Message: "How did the interview go?"
Context: ["User had a job interview yesterday"]
→ Extract:
  - query_text: "job interview outcome", type: continuity, reference: "the interview", reasoning: "Following up on the interview mentioned in context"

Message: "Hello! How are you?"
Context: []
→ No queries needed (greeting, no memory retrieval required)

Message: "My mom called me today"
Context: ["User's mom lives in Florida", "They had some tension last month"]
→ Extract:
  - query_text: "mom", type: entity_overview, reference: "my mom", reasoning: "Need mom's background to respond appropriately"
  - query_text: "relationship with mom recently", type: continuity, reference: "tension context", reasoning: "Recent context about their relationship"

Now analyze the user message and extract the appropriate queries."""


# =============================================================================
# LLM Response Models
# =============================================================================


class ExtractedQueryResponse(BaseModel):
    """A single extracted query in the LLM response."""

    query_text: str = Field(description="The query to run against memory")
    query_type: str = Field(
        description=(
            "Type: current_state, history, entity_overview, temporal, "
            "continuity, proactive_context, or no_retrieval"
        )
    )
    reference: str = Field(
        description="What triggered this query (entity, topic, pronoun)"
    )
    reasoning: str = Field(description="Why this query would help")


class ExtractionResponse(BaseModel):
    """LLM response for query extraction."""

    queries: list[ExtractedQueryResponse] = Field(
        default_factory=list,
        description="List of extracted queries",
    )
    context_summary: str = Field(
        default="",
        description="Brief summary of why these queries matter",
    )
    no_retrieval_needed: bool = Field(
        default=False,
        description="True if no memory retrieval is needed (greeting, general help, etc.)",
    )


# =============================================================================
# Query Extractor
# =============================================================================


@dataclass
class ExtractorConfig:
    """Configuration for the query extractor."""

    model: SupportedModel = SupportedModel.MISTRAL_SMALL_3_2_Q4
    temperature: float = 0.1
    max_context_messages: int = 10


class QueryExtractor:
    """Extracts queries from user messages with context.

    This is the main interface for query extraction. It takes a user message
    along with recent conversation context and returns a list of queries
    with their types, ready for retrieval routing.
    """

    def __init__(
        self,
        llm: LLM,
        config: ExtractorConfig | None = None,
    ):
        self.llm = llm
        self.config = config or ExtractorConfig()

    def extract(
        self,
        message: str,
        context: list[str],
    ) -> ExtractionResult:
        """Extract queries from a user message with context.

        Args:
            message: The user's current message
            context: Recent conversation history (list of message strings)

        Returns:
            ExtractionResult with extracted queries and context summary
        """
        # Limit context to configured max
        recent_context = context[-self.config.max_context_messages :]

        # Format context for prompt
        if recent_context:
            context_str = "\n".join(f"- {msg}" for msg in recent_context)
        else:
            context_str = "(no recent context)"

        prompt = EXTRACTION_PROMPT.format(
            message=message,
            context=context_str,
        )

        try:
            response = direct_structured_llm_call(
                prompt=prompt,
                response_model=ExtractionResponse,
                model=self.config.model,
                llm=self.llm,
                caller="query_extractor",
                temperature=self.config.temperature,
            )

            # Handle no-retrieval case
            if response.no_retrieval_needed:
                return ExtractionResult(
                    queries=[],
                    context_summary="No memory retrieval needed for this message.",
                )

            # Convert response to ExtractedQuery objects
            queries: list[ExtractedQuery] = []
            for q in response.queries:
                try:
                    query_type = QueryType(q.query_type.lower().strip())
                except ValueError:
                    logger.warning(
                        f"Unknown query type '{q.query_type}', "
                        f"defaulting to proactive_context"
                    )
                    query_type = QueryType.PROACTIVE_CONTEXT

                queries.append(
                    ExtractedQuery(
                        query_text=q.query_text,
                        query_type=query_type,
                        reference=q.reference,
                        reasoning=q.reasoning,
                    )
                )

            return ExtractionResult(
                queries=queries,
                context_summary=response.context_summary,
            )

        except Exception as e:
            logger.error(f"Query extraction failed: {e}")
            # Return empty result on error
            return ExtractionResult(
                queries=[],
                context_summary=f"Extraction failed: {e}",
            )

    def extract_from_input(self, extraction_input: ExtractionInput) -> ExtractionResult:
        """Extract queries from an ExtractionInput object.

        Convenience method that wraps extract() for use with ExtractionInput.
        """
        return self.extract(
            message=extraction_input.message,
            context=extraction_input.context,
        )


# =============================================================================
# Batch Extraction
# =============================================================================


def extract_queries_batch(
    extractor: QueryExtractor,
    inputs: list[ExtractionInput],
) -> list[ExtractionResult]:
    """Extract queries from multiple inputs.

    Args:
        extractor: The QueryExtractor instance to use
        inputs: List of extraction inputs

    Returns:
        List of extraction results, one per input
    """
    results: list[ExtractionResult] = []
    for i, input_data in enumerate(inputs):
        logger.info(f"Extracting queries from input {i + 1}/{len(inputs)}")
        result = extractor.extract_from_input(input_data)
        results.append(result)
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Test the query extractor with sample inputs."""
    import json
    logging.basicConfig(level=logging.INFO)

    from agent.llm import create_llm

    # Create extractor
    llm = create_llm()
    extractor = QueryExtractor(llm)

    # Sample test cases
    test_cases = [
        ExtractionInput(
            message="I saw Sarah at the coffee shop, she looked tired",
            context=[
                "Earlier we discussed Sarah's new job",
                "She mentioned being stressed",
            ],
        ),
        ExtractionInput(
            message="How did the interview go?",
            context=["User had a job interview yesterday"],
        ),
        ExtractionInput(
            message="Hello! How are you today?",
            context=[],
        ),
        ExtractionInput(
            message="My mom called me today",
            context=[
                "User's mom lives in Florida",
                "They had some tension last month",
            ],
        ),
        ExtractionInput(
            message="What happened yesterday?",
            context=["User went to a meeting", "User met with their boss"],
        ),
        ExtractionInput(
            message="I'm thinking about that job offer",
            context=[
                "User received a job offer from TechCorp",
                "The offer was for a senior engineer role",
            ],
        ),
    ]

    print("\n" + "=" * 60)
    print("Query Extraction Test")
    print("=" * 60)

    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i + 1} ---")
        print(f"Message: {test_case.message}")
        print(f"Context: {test_case.context}")

        result = extractor.extract_from_input(test_case)

        print(f"\nResult:")
        print(f"  Context Summary: {result.context_summary}")
        print(f"  Queries ({len(result.queries)}):")
        for q in result.queries:
            print(f"    - {q.query_text}")
            print(f"      Type: {q.query_type.value}")
            print(f"      Reference: {q.reference}")
            print(f"      Reasoning: {q.reasoning}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
