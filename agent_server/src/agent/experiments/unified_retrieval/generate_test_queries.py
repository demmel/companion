"""Generate test queries with ground truth for proper IR evaluation.

This script creates a ground truth dataset by:
1. Extracting candidate queries from conversation history
2. Auto-suggesting expected retrieval results using heuristics
3. Outputting JSON for human review

Usage:
    uv run python -m agent.experiments.unified_retrieval.generate_test_queries \
        --conversation <id> --output test_queries_groundtruth.json
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from agent.embedding_service import EmbeddingService, get_embedding_service

from .build_indices import (
    CACHE_DIR,
    CONVERSATIONS_DIR,
    load_conversation_memories,
    load_indices,
    build_all_indices,
)
from .models import QueryType, Memory, Fact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Ground Truth Models
# =============================================================================


@dataclass
class ExpectedFact:
    """A fact expected to be retrieved for a query."""

    entity: str
    attribute: str
    value: str


@dataclass
class GroundTruthQuery:
    """A test query with ground truth for evaluation.

    This is the format needed for proper IR evaluation:
    - We know exactly which memories/facts should be retrieved
    - We can compute precision/recall objectively
    """

    id: str
    query: str
    query_type: str

    # Ground truth - what SHOULD be retrieved
    expected_memory_ids: list[str] = field(default_factory=list)
    expected_facts: list[ExpectedFact] = field(default_factory=list)
    expected_entity: str | None = None
    expected_attribute: str | None = None

    # Metadata
    source_turn_index: int | None = None
    confidence: float = 0.0
    needs_review: bool = True
    notes: str = ""


@dataclass
class CandidateQuery:
    """A query extracted from conversation that may need retrieval."""

    query: str
    query_type: str
    source_index: int
    timestamp: datetime

    # Detected elements
    detected_entities: list[str] = field(default_factory=list)
    detected_time_refs: list[str] = field(default_factory=list)
    detected_attributes: list[str] = field(default_factory=list)


@dataclass
class SuggestedResults:
    """Auto-suggested expected results for a query."""

    memory_ids: list[str] = field(default_factory=list)
    facts: list[ExpectedFact] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


# =============================================================================
# Query Detection Patterns
# =============================================================================


# Patterns for detecting query types
ENTITY_PATTERNS = [
    r"what (?:is|are|about) (\w+)",
    r"tell me about (\w+)",
    r"who is (\w+)",
    r"(\w+)'s",
    r"describe (\w+)",
]

TIME_PATTERNS = [
    r"\byesterday\b",
    r"\blast week\b",
    r"\blast month\b",
    r"\bthis morning\b",
    r"\btoday\b",
    r"\bon (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\blast (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]

STATE_PATTERNS = [
    r"what (?:is|are) .+ (?:wearing|doing|feeling|thinking)",
    r"where (?:is|are)",
    r"how (?:is|are) .+ (?:feeling|doing)",
    r"current (?:status|state|location|mood)",
]

HISTORY_PATTERNS = [
    r"what (?:has|have) .+ (?:worn|done|said|told)",
    r"how (?:has|have) .+ (?:changed|evolved|progressed)",
    r"what (?:projects|things|topics) have we",
]

ENTITY_OVERVIEW_PATTERNS = [
    r"who is (\w+)",
    r"tell me about (\w+)",
    r"what do (?:i|we|you) know about",
    r"describe .+ relationship",
    r"everything about",
]


def detect_entities(text: str) -> list[str]:
    """Detect entity references in text."""
    entities: list[str] = []

    # Use patterns to find entity mentions
    for pattern in ENTITY_PATTERNS:
        matches = re.findall(pattern, text.lower())
        entities.extend(matches)

    # Also look for capitalized words (names)
    words = text.split()
    for word in words:
        clean = word.strip(".,!?\"'")
        if clean and clean[0].isupper() and len(clean) > 1:
            entities.append(clean.lower())

    return list(set(entities))


def detect_time_references(text: str) -> list[str]:
    """Detect temporal references in text."""
    refs: list[str] = []

    for pattern in TIME_PATTERNS:
        matches = re.findall(pattern, text.lower())
        refs.extend(matches)

    return list(set(refs))


def classify_query_type(text: str) -> str:
    """Classify the query type based on patterns."""
    lower = text.lower()

    # Check for no retrieval (greetings/acknowledgments)
    if lower.strip() in ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", "got it"]:
        return "no_retrieval"

    # Check for current state queries
    for pattern in STATE_PATTERNS:
        if re.search(pattern, lower):
            return "current_state"

    # Check for entity overview
    for pattern in ENTITY_OVERVIEW_PATTERNS:
        if re.search(pattern, lower):
            return "entity_overview"

    # Check for history queries
    for pattern in HISTORY_PATTERNS:
        if re.search(pattern, lower):
            return "history"

    # Check for temporal queries
    if detect_time_references(text):
        return "temporal"

    # Check for continuity (follow-up questions)
    continuity_keywords = ["how did", "any updates", "what happened with", "turn out", "did anything change"]
    for keyword in continuity_keywords:
        if keyword in lower:
            return "continuity"

    # Default to proactive context if entities detected
    if detect_entities(text):
        return "proactive_context"

    return "no_retrieval"


# =============================================================================
# Query Extraction
# =============================================================================


def extract_candidate_queries(
    conversation_id: str,
    min_length: int = 15,
    max_queries: int = 150,
) -> list[CandidateQuery]:
    """Extract candidate queries from conversation history.

    Looks for user messages that likely need retrieval context.
    """
    # Load conversation
    possible_paths = [
        CONVERSATIONS_DIR / f"{conversation_id}_triggers.json",
        CONVERSATIONS_DIR / "archive" / conversation_id / f"{conversation_id}_triggers.json",
    ]

    triggers_file = None
    for path in possible_paths:
        if path.exists():
            triggers_file = path
            break

    if not triggers_file:
        raise FileNotFoundError(f"Conversation not found: {conversation_id}")

    with open(triggers_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    candidates: list[CandidateQuery] = []

    for i, entry in enumerate(entries):
        trigger = entry.get("trigger", {})
        trigger_type = trigger.get("type", "")

        if trigger_type != "user_input":
            continue

        user_input = trigger.get("content", "")

        if not user_input or len(user_input) < min_length:
            continue

        # Get timestamp
        timestamp_str = entry.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            timestamp = datetime.now()

        # Classify and detect references
        query_type = classify_query_type(user_input)
        entities = detect_entities(user_input)
        time_refs = detect_time_references(user_input)

        # Skip pure greetings for candidate extraction
        if query_type == "no_retrieval" and not entities and not time_refs:
            continue

        candidates.append(CandidateQuery(
            query=user_input,
            query_type=query_type,
            source_index=i,
            timestamp=timestamp,
            detected_entities=entities,
            detected_time_refs=time_refs,
        ))

        if len(candidates) >= max_queries:
            break

    logger.info(f"Extracted {len(candidates)} candidate queries")
    return candidates


# =============================================================================
# Ground Truth Suggestion
# =============================================================================


def suggest_expected_results(
    candidate: CandidateQuery,
    memories: list[Memory],
    embedding_service: EmbeddingService,
    kg_facts: list[tuple[str, str, str, str]] | None = None,
) -> SuggestedResults:
    """Suggest expected retrieval results using heuristics.

    Args:
        candidate: The query to find results for
        memories: All available memories (filtered to before query's timestamp)
        embedding_service: For similarity search
        kg_facts: Optional list of (entity, attribute, value, memory_id) tuples

    Returns:
        SuggestedResults with suggested memory IDs and facts
    """
    # Only consider memories from BEFORE this query
    past_memories = [m for m in memories if m.timestamp < candidate.timestamp]

    if not past_memories:
        return SuggestedResults(
            confidence=0.0,
            reasoning="No past memories available",
        )

    # Compute query embedding
    query_embedding = embedding_service.encode(candidate.query)

    # Score memories by similarity
    scored_memories: list[tuple[Memory, float]] = []
    for memory in past_memories:
        if memory.embedding_vector:
            score = embedding_service.cosine_similarity(query_embedding, memory.embedding_vector)
            scored_memories.append((memory, score))

    # Sort by score
    scored_memories.sort(key=lambda x: x[1], reverse=True)

    # Select based on query type
    suggested_ids: list[str] = []
    suggested_facts: list[ExpectedFact] = []
    confidence = 0.0
    reasoning = ""

    if candidate.query_type == "current_state":
        # For current state: need most recent memory about the entity/attribute
        # Find memories mentioning detected entities
        entity_memories = []
        for memory, score in scored_memories:
            for entity in candidate.detected_entities:
                if entity.lower() in memory.content.lower():
                    entity_memories.append((memory, score))
                    break

        if entity_memories:
            # Take most recent among entity matches
            entity_memories.sort(key=lambda x: x[0].timestamp, reverse=True)
            suggested_ids = [entity_memories[0][0].memory_id]
            confidence = 0.9
            reasoning = f"Most recent memory mentioning {candidate.detected_entities}"
        else:
            # Fall back to top similarity
            if scored_memories:
                suggested_ids = [scored_memories[0][0].memory_id]
                confidence = 0.5
                reasoning = "Fallback to top similarity match"

    elif candidate.query_type == "history":
        # For history: need multiple memories about the topic
        # Take top-k by similarity
        suggested_ids = [m.memory_id for m, s in scored_memories[:5] if s > 0.3]
        confidence = 0.7 if suggested_ids else 0.3
        reasoning = f"Top {len(suggested_ids)} similar memories for history query"

    elif candidate.query_type == "entity_overview":
        # For entity overview: need memories about the entity
        entity_memories = []
        for memory, score in scored_memories:
            for entity in candidate.detected_entities:
                if entity.lower() in memory.content.lower():
                    entity_memories.append((memory, score))
                    break

        suggested_ids = [m.memory_id for m, s in entity_memories[:5]]
        confidence = 0.8 if suggested_ids else 0.4
        reasoning = f"Memories mentioning {candidate.detected_entities}"

    elif candidate.query_type == "temporal":
        # For temporal: look for memories near time reference
        # Without proper time parsing, use recency + similarity
        recent_memories = sorted(past_memories, key=lambda m: m.timestamp, reverse=True)[:20]
        for memory in recent_memories:
            for memory_scored, score in scored_memories:
                if memory.memory_id == memory_scored.memory_id and score > 0.3:
                    suggested_ids.append(memory.memory_id)
                    break
            if len(suggested_ids) >= 5:
                break

        confidence = 0.6
        reasoning = "Recent memories with similarity > 0.3"

    elif candidate.query_type == "continuity":
        # For continuity: recent memories on same topic
        recent_high_sim = [
            (m, s) for m, s in scored_memories[:10]
            if s > 0.4
        ]
        recent_high_sim.sort(key=lambda x: x[0].timestamp, reverse=True)
        suggested_ids = [m.memory_id for m, s in recent_high_sim[:5]]
        confidence = 0.7 if suggested_ids else 0.4
        reasoning = "Recent memories with high topic similarity"

    else:  # proactive_context
        # For proactive: top similarity matches
        suggested_ids = [m.memory_id for m, s in scored_memories[:5] if s > 0.3]
        confidence = 0.6 if suggested_ids else 0.3
        reasoning = "Top similarity matches for context"

    return SuggestedResults(
        memory_ids=suggested_ids,
        facts=suggested_facts,
        confidence=confidence,
        reasoning=reasoning,
    )


# =============================================================================
# Main Generation Pipeline
# =============================================================================


def generate_ground_truth_dataset(
    conversation_id: str,
    output_path: Path,
    max_queries: int = 100,
    use_cached_indices: bool = True,
) -> list[GroundTruthQuery]:
    """Generate ground truth test dataset.

    Args:
        conversation_id: Conversation to extract queries from
        output_path: Where to save the JSON output
        max_queries: Maximum queries to generate
        use_cached_indices: Whether to use cached embeddings

    Returns:
        List of ground truth queries
    """
    logger.info(f"Generating ground truth for conversation: {conversation_id}")

    # Load conversation and build indices
    embedding_service = get_embedding_service()
    memory_elements, memories = load_conversation_memories(conversation_id)

    # Compute embeddings for memories if needed
    for memory in memories:
        if not memory.embedding_vector:
            memory.embedding_vector = embedding_service.encode(memory.content)

    # Extract candidates
    candidates = extract_candidate_queries(
        conversation_id,
        max_queries=max_queries * 2,  # Extract more, filter later
    )

    # Generate ground truth for each candidate
    ground_truth_queries: list[GroundTruthQuery] = []

    for i, candidate in enumerate(candidates):
        # Suggest expected results
        suggestions = suggest_expected_results(
            candidate,
            memories,
            embedding_service,
        )

        # Create ground truth query
        gt_query = GroundTruthQuery(
            id=f"{candidate.query_type[:4]}_{i+1:03d}",
            query=candidate.query,
            query_type=candidate.query_type,
            expected_memory_ids=suggestions.memory_ids,
            expected_facts=suggestions.facts,
            expected_entity=candidate.detected_entities[0] if candidate.detected_entities else None,
            source_turn_index=candidate.source_index,
            confidence=suggestions.confidence,
            needs_review=True,
            notes=suggestions.reasoning,
        )
        ground_truth_queries.append(gt_query)

        if len(ground_truth_queries) >= max_queries:
            break

    # Balance query types
    by_type: dict[str, list[GroundTruthQuery]] = {}
    for q in ground_truth_queries:
        by_type.setdefault(q.query_type, []).append(q)

    logger.info("Query type distribution:")
    for qtype, queries in sorted(by_type.items()):
        logger.info(f"  {qtype}: {len(queries)}")

    # Save to JSON
    output_data = {
        "description": "Ground truth test queries for unified retrieval evaluation",
        "version": "2.0",
        "conversation_id": conversation_id,
        "generation_timestamp": datetime.now().isoformat(),
        "query_types": list(by_type.keys()),
        "total_queries": len(ground_truth_queries),
        "queries": [
            {
                "id": q.id,
                "query": q.query,
                "query_type": q.query_type,
                "expected_memory_ids": q.expected_memory_ids,
                "expected_facts": [asdict(f) for f in q.expected_facts],
                "expected_entity": q.expected_entity,
                "expected_attribute": q.expected_attribute,
                "source_turn_index": q.source_turn_index,
                "confidence": q.confidence,
                "needs_review": q.needs_review,
                "notes": q.notes,
            }
            for q in ground_truth_queries
        ],
        "review_instructions": {
            "purpose": "Review and correct the suggested ground truth for accurate IR evaluation",
            "fields_to_verify": [
                "expected_memory_ids - which memories SHOULD be retrieved?",
                "expected_facts - what facts SHOULD be retrieved?",
                "query_type - is the classification correct?",
            ],
            "after_review": "Set needs_review to false for verified queries",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(ground_truth_queries)} queries to {output_path}")
    return ground_truth_queries


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate ground truth test queries for retrieval evaluation"
    )
    parser.add_argument(
        "--conversation",
        type=str,
        required=True,
        help="Conversation ID to extract queries from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Maximum number of queries to generate (default: 100)",
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        CACHE_DIR / f"{args.conversation}_groundtruth.json"
    )

    generate_ground_truth_dataset(
        conversation_id=args.conversation,
        output_path=output_path,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()
