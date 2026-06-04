"""Query classifier implementations for routing queries to appropriate strategies.

This module provides:
- LLMQueryClassifier: Zero-shot LLM baseline classifier
- RuleBasedQueryClassifier: Fast rule-based classifier for common patterns
"""

import logging

from pydantic import BaseModel, Field

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from .models import DetectedReference, QueryType, QueryClassificationResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Query Classification Prompts
# =============================================================================

QUERY_CLASSIFICATION_PROMPT = """Classify this query to determine the best retrieval strategy.

QUERY: {query}

RECENT CONTEXT:
{context}

DETECTED REFERENCES:
{references}

CLASSIFICATION TYPES:
- current_state: Asking for the CURRENT value of something (what is X wearing NOW?, where is X?)
- history: Asking about PAST events or changes over time (what has X worn?, how has X changed?)
- entity_overview: Asking for a summary of what's known about someone/something (who is X?, tell me about X)
- temporal: Asking about a specific TIME PERIOD (what happened yesterday?, last week's events)
- continuity: Following up on an ongoing situation or event (how did the interview go?, any updates on X?)
- proactive_context: No explicit question, but references suggest context would help
- no_retrieval: Greeting, thanks, or no memory context needed (hello, thanks, bye)

Choose the classification that best matches the query intent."""


REFERENCE_DETECTION_PROMPT = """Analyze this user input to find references that would benefit from memory context.

USER INPUT: {user_input}

RECENT CONVERSATION:
{context}

Look for:
- People mentioned by name (who are they?)
- Pronouns referring to known people ("my mom", "my boss")
- Events being followed up on ("the interview", "the date")
- Ongoing situations ("work", "the situation", "things")
- Places with shared history ("the usual spot", "that restaurant")
- Past conversations ("remember when", "what you suggested")
- Recurring issues ("again", "is back", "still")
- Implicit references ("the project", "my resolution")

For each reference found, identify:
- text: The reference text
- type: entity, topic, time, or event

Be thorough - it's better to retrieve context you don't use than to miss something."""


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================


class ReferenceItem(BaseModel):
    """A single detected reference."""

    text: str = Field(description="The reference text found in input")
    type: str = Field(description="Type: entity, topic, time, or event")


class ReferenceDetectionResponse(BaseModel):
    """LLM response for reference detection."""

    has_references: bool = Field(
        description="Whether any references were found that need context"
    )
    references: list[ReferenceItem] = Field(
        description="List of detected references",
        default_factory=list,
    )


# =============================================================================
# LLM Query Classifier
# =============================================================================


class LLMQueryClassifier:
    """Zero-shot LLM classifier for query type routing.

    This is the baseline classifier that uses an LLM to classify queries.
    It can be swapped for more efficient implementations later.
    """

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel = SupportedModel.MISTRAL_SMALL_3_2_Q4,
    ):
        self.llm = llm
        self.model = model

    def classify(
        self,
        query: str,
        context: list[str],
        detected_references: list[DetectedReference],
    ) -> QueryType:
        """Classify query type using LLM.

        Args:
            query: The user's query
            context: Recent conversation context
            detected_references: References detected in the query

        Returns:
            QueryType indicating the best retrieval strategy
        """
        # Format context
        context_str = "\n".join(context[-5:]) if context else "(no recent context)"

        # Format references
        if detected_references:
            refs_str = "\n".join(
                f"- {r.text} ({r.reference_type})" for r in detected_references
            )
        else:
            refs_str = "(no references detected)"

        prompt = QUERY_CLASSIFICATION_PROMPT.format(
            query=query,
            context=context_str,
            references=refs_str,
        )

        try:
            response = direct_structured_llm_call(
                prompt=prompt,
                response_model=QueryClassificationResponse,
                model=self.model,
                llm=self.llm,
                caller="query_classifier",
                temperature=0.1,
            )

            # Map response to QueryType
            query_type_str = response.query_type.lower().strip()
            try:
                return QueryType(query_type_str)
            except ValueError:
                logger.warning(
                    f"Unknown query type '{query_type_str}', defaulting to proactive_context"
                )
                return QueryType.PROACTIVE_CONTEXT

        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            # Default to proactive context on error - better to over-retrieve
            return QueryType.PROACTIVE_CONTEXT

    def detect_references(
        self,
        user_input: str,
        context: list[str],
    ) -> list[DetectedReference]:
        """Detect references in user input that may need context.

        Args:
            user_input: The user's input
            context: Recent conversation context

        Returns:
            List of detected references
        """
        context_str = "\n".join(context[-5:]) if context else "(no recent context)"

        prompt = REFERENCE_DETECTION_PROMPT.format(
            user_input=user_input,
            context=context_str,
        )

        try:
            response = direct_structured_llm_call(
                prompt=prompt,
                response_model=ReferenceDetectionResponse,
                model=self.model,
                llm=self.llm,
                caller="reference_detection",
                temperature=0.1,
            )

            references: list[DetectedReference] = []
            if response.has_references:
                for ref in response.references:
                    references.append(
                        DetectedReference(
                            text=ref.text,
                            reference_type=ref.type,
                        )
                    )

            return references

        except Exception as e:
            logger.error(f"Reference detection failed: {e}")
            return []


# =============================================================================
# Rule-Based Classifier (Fast Fallback)
# =============================================================================


class RuleBasedQueryClassifier:
    """Fast rule-based classifier for common query patterns.

    This classifier uses simple pattern matching and keywords for quick
    classification without LLM calls. Use as a fast fallback or first pass.
    """

    # Patterns for each query type
    CURRENT_STATE_PATTERNS = [
        "what is",
        "what's",
        "where is",
        "where's",
        "how is",
        "how's",
        "current",
        "right now",
        "at the moment",
        "today",
    ]

    HISTORY_PATTERNS = [
        "what has",
        "what have",
        "how has",
        "how have",
        "history",
        "over time",
        "changes",
        "evolution",
        "used to",
        "before",
    ]

    ENTITY_OVERVIEW_PATTERNS = [
        "who is",
        "who's",
        "tell me about",
        "what do i know about",
        "what do you know about",
        "describe",
        "summary of",
    ]

    TEMPORAL_PATTERNS = [
        "yesterday",
        "last week",
        "last month",
        "last year",
        "ago",
        "on monday",
        "on tuesday",
        "on wednesday",
        "on thursday",
        "on friday",
        "on saturday",
        "on sunday",
        "this morning",
        "this afternoon",
        "this evening",
        "last night",
        "earlier today",
    ]

    CONTINUITY_PATTERNS = [
        "how did",
        "how was",
        "any update",
        "what happened with",
        "followup",
        "follow up",
        "any news",
        "result of",
        "outcome of",
    ]

    NO_RETRIEVAL_PATTERNS = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "see you",
    ]

    def classify(
        self,
        query: str,
        context: list[str],
        detected_references: list[DetectedReference],
    ) -> QueryType:
        """Classify query using pattern matching.

        Args:
            query: The user's query
            context: Recent conversation context (unused in rule-based)
            detected_references: References detected in the query

        Returns:
            QueryType based on pattern matching
        """
        query_lower = query.lower().strip()

        # Check patterns in order of specificity
        if self._matches_any(query_lower, self.NO_RETRIEVAL_PATTERNS):
            return QueryType.NO_RETRIEVAL

        if self._matches_any(query_lower, self.TEMPORAL_PATTERNS):
            return QueryType.TEMPORAL

        if self._matches_any(query_lower, self.ENTITY_OVERVIEW_PATTERNS):
            return QueryType.ENTITY_OVERVIEW

        if self._matches_any(query_lower, self.CURRENT_STATE_PATTERNS):
            return QueryType.CURRENT_STATE

        if self._matches_any(query_lower, self.HISTORY_PATTERNS):
            return QueryType.HISTORY

        if self._matches_any(query_lower, self.CONTINUITY_PATTERNS):
            return QueryType.CONTINUITY

        # If references were detected, use proactive context
        if detected_references:
            return QueryType.PROACTIVE_CONTEXT

        # Default to proactive context - better to over-retrieve
        return QueryType.PROACTIVE_CONTEXT

    def _matches_any(self, text: str, patterns: list[str]) -> bool:
        """Check if text matches any of the patterns."""
        return any(pattern in text for pattern in patterns)


# =============================================================================
# Hybrid Classifier
# =============================================================================


class HybridQueryClassifier:
    """Hybrid classifier that uses rules first, LLM for uncertain cases.

    This provides fast classification for common patterns while falling
    back to LLM for ambiguous queries.
    """

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel = SupportedModel.MISTRAL_SMALL_3_2_Q4,
        use_llm_fallback: bool = True,
    ):
        self.rule_classifier = RuleBasedQueryClassifier()
        self.llm_classifier = LLMQueryClassifier(llm, model) if use_llm_fallback else None
        self.use_llm_fallback = use_llm_fallback

    def classify(
        self,
        query: str,
        context: list[str],
        detected_references: list[DetectedReference],
    ) -> QueryType:
        """Classify using rules first, LLM for fallback.

        Args:
            query: The user's query
            context: Recent conversation context
            detected_references: References detected in the query

        Returns:
            QueryType for routing
        """
        # Try rule-based first
        rule_result = self.rule_classifier.classify(
            query, context, detected_references
        )

        # If rule-based gave a confident result (not proactive_context), use it
        if rule_result != QueryType.PROACTIVE_CONTEXT:
            return rule_result

        # Fall back to LLM for uncertain cases
        if self.use_llm_fallback and self.llm_classifier:
            return self.llm_classifier.classify(query, context, detected_references)

        return rule_result

    def detect_references(
        self,
        user_input: str,
        context: list[str],
    ) -> list[DetectedReference]:
        """Detect references using LLM classifier."""
        if self.llm_classifier:
            return self.llm_classifier.detect_references(user_input, context)
        return []
