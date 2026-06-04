"""Create labeled dataset for query extraction evaluation.

This creates hand-crafted examples of user messages with context and the
expected queries that should be extracted. Unlike the standalone query
classification dataset, these examples have full context windows and
focus on the extraction task.

Run:
    uv run python -m agent.experiments.query_classification.create_extraction_dataset
"""

import json
import logging
from pathlib import Path

from .models import (
    ExtractedQuery,
    ExtractionDataset,
    LabeledExtractionExample,
    QueryType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Hand-Crafted Extraction Examples
# =============================================================================

EXTRACTION_EXAMPLES: list[dict[str, str | list[str] | list[dict[str, str]]]] = [
    # -------------------------------------------------------------------------
    # Entity mentions with context
    # -------------------------------------------------------------------------
    {
        "message": "I saw Sarah at the coffee shop, she looked tired",
        "context": [
            "Earlier we discussed Sarah's new job",
            "She mentioned being stressed about deadlines",
        ],
        "expected_queries": [
            {
                "query_text": "Sarah",
                "query_type": "entity_overview",
                "reference": "Sarah",
                "reasoning": "Need Sarah's background for context",
            },
            {
                "query_text": "Sarah current mood and energy",
                "query_type": "current_state",
                "reference": "looked tired",
                "reasoning": "User noting tiredness, should track her current state",
            },
            {
                "query_text": "Sarah job stress",
                "query_type": "continuity",
                "reference": "job stress mentioned in context",
                "reasoning": "Follow up on ongoing job stress situation",
            },
        ],
        "notes": "Multiple query types for single entity mention",
    },
    {
        "message": "My mom called me today",
        "context": [
            "User's mom lives in Florida",
            "They had some tension last month over holiday plans",
        ],
        "expected_queries": [
            {
                "query_text": "mom",
                "query_type": "entity_overview",
                "reference": "my mom",
                "reasoning": "Need mom's background to respond appropriately",
            },
            {
                "query_text": "relationship with mom",
                "query_type": "continuity",
                "reference": "tension mentioned in context",
                "reasoning": "Recent context about their relationship tension",
            },
        ],
        "notes": "Family relationship with relevant context",
    },
    {
        "message": "Mike is in town this weekend",
        "context": [],
        "expected_queries": [
            {
                "query_text": "Mike",
                "query_type": "entity_overview",
                "reference": "Mike",
                "reasoning": "Need to know who Mike is",
            },
        ],
        "notes": "Entity mention with no context - just need overview",
    },
    # -------------------------------------------------------------------------
    # Continuity / follow-up scenarios
    # -------------------------------------------------------------------------
    {
        "message": "How did the interview go?",
        "context": ["User had a job interview yesterday at TechCorp"],
        "expected_queries": [
            {
                "query_text": "job interview outcome",
                "query_type": "continuity",
                "reference": "the interview",
                "reasoning": "Following up on the interview mentioned in context",
            },
        ],
        "notes": "Direct follow-up question on recent event",
    },
    {
        "message": "Any update on that?",
        "context": [
            "User was waiting to hear back about an apartment application",
            "The landlord said they would decide by Friday",
        ],
        "expected_queries": [
            {
                "query_text": "apartment application status",
                "query_type": "continuity",
                "reference": "that (apartment application)",
                "reasoning": "Following up on pending apartment decision",
            },
        ],
        "notes": "Vague follow-up that requires context resolution",
    },
    {
        "message": "Did they ever respond?",
        "context": ["User sent an important email to their client last week"],
        "expected_queries": [
            {
                "query_text": "client email response",
                "query_type": "continuity",
                "reference": "they (client)",
                "reasoning": "Following up on client communication",
            },
        ],
        "notes": "Pronoun resolution needed",
    },
    # -------------------------------------------------------------------------
    # Temporal queries
    # -------------------------------------------------------------------------
    {
        "message": "What happened yesterday?",
        "context": ["User went to a meeting", "User met with their boss"],
        "expected_queries": [
            {
                "query_text": "events from yesterday",
                "query_type": "temporal",
                "reference": "yesterday",
                "reasoning": "Time-bounded query for specific day",
            },
        ],
        "notes": "Explicit temporal reference",
    },
    {
        "message": "How was last week for you?",
        "context": ["User had several important meetings last week"],
        "expected_queries": [
            {
                "query_text": "last week events and state",
                "query_type": "temporal",
                "reference": "last week",
                "reasoning": "Week-bounded query for recent period",
            },
        ],
        "notes": "Temporal with implied state query",
    },
    {
        "message": "I was really productive this morning",
        "context": ["User has been struggling with focus lately"],
        "expected_queries": [
            {
                "query_text": "user productivity and focus",
                "query_type": "continuity",
                "reference": "focus struggles in context",
                "reasoning": "Connect to ongoing focus challenges",
            },
        ],
        "notes": "Time reference but continuity is more relevant",
    },
    # -------------------------------------------------------------------------
    # Current state queries
    # -------------------------------------------------------------------------
    {
        "message": "Where does Sarah work now?",
        "context": ["Sarah recently switched jobs"],
        "expected_queries": [
            {
                "query_text": "Sarah current job",
                "query_type": "current_state",
                "reference": "Sarah work",
                "reasoning": "Asking for current employment state",
            },
        ],
        "notes": "Explicit current state question",
    },
    {
        "message": "What's my brother up to these days?",
        "context": ["User's brother moved to Seattle last year"],
        "expected_queries": [
            {
                "query_text": "brother",
                "query_type": "entity_overview",
                "reference": "my brother",
                "reasoning": "Need comprehensive info about brother",
            },
            {
                "query_text": "brother current activities",
                "query_type": "current_state",
                "reference": "up to these days",
                "reasoning": "Asking for current state/activities",
            },
        ],
        "notes": "Overview combined with current state",
    },
    # -------------------------------------------------------------------------
    # History queries
    # -------------------------------------------------------------------------
    {
        "message": "What have we talked about regarding the project?",
        "context": ["User is working on Project Alpha"],
        "expected_queries": [
            {
                "query_text": "Project Alpha discussions",
                "query_type": "history",
                "reference": "the project",
                "reasoning": "Historical query about past conversations",
            },
        ],
        "notes": "Explicit history request",
    },
    {
        "message": "Remember when we discussed my career goals?",
        "context": ["User mentioned wanting to switch to management"],
        "expected_queries": [
            {
                "query_text": "career goals discussion",
                "query_type": "history",
                "reference": "career goals",
                "reasoning": "Recalling past conversation about career",
            },
        ],
        "notes": "Explicit memory recall request",
    },
    # -------------------------------------------------------------------------
    # Proactive context (statements, not questions)
    # -------------------------------------------------------------------------
    {
        "message": "I'm thinking about that job offer",
        "context": [
            "User received a job offer from TechCorp",
            "The offer was for a senior engineer role with 20% raise",
        ],
        "expected_queries": [
            {
                "query_text": "TechCorp job offer",
                "query_type": "proactive_context",
                "reference": "that job offer",
                "reasoning": "Need context about the offer for discussion",
            },
        ],
        "notes": "Statement needing context, not a question",
    },
    {
        "message": "The apartment issue got sorted out",
        "context": ["User had a plumbing problem in their apartment"],
        "expected_queries": [
            {
                "query_text": "apartment issue",
                "query_type": "continuity",
                "reference": "apartment issue",
                "reasoning": "Closing out an ongoing issue",
            },
        ],
        "notes": "Resolution statement for tracked issue",
    },
    {
        "message": "I talked to my therapist about it",
        "context": [
            "User has been dealing with anxiety",
            "User mentioned considering therapy",
        ],
        "expected_queries": [
            {
                "query_text": "therapist",
                "query_type": "entity_overview",
                "reference": "my therapist",
                "reasoning": "Need context about the therapist",
            },
            {
                "query_text": "user anxiety",
                "query_type": "continuity",
                "reference": "it (implied)",
                "reasoning": "What they likely discussed",
            },
        ],
        "notes": "Multiple references including pronoun",
    },
    # -------------------------------------------------------------------------
    # No retrieval needed
    # -------------------------------------------------------------------------
    {
        "message": "Hello! How are you?",
        "context": [],
        "expected_queries": [],
        "notes": "Greeting - no retrieval needed",
    },
    {
        "message": "Thanks for your help!",
        "context": ["We just finished discussing a technical issue"],
        "expected_queries": [],
        "notes": "Gratitude - no retrieval needed",
    },
    {
        "message": "Can you help me write an email?",
        "context": [],
        "expected_queries": [],
        "notes": "Task request - no memory retrieval needed",
    },
    {
        "message": "What's 15% of 230?",
        "context": [],
        "expected_queries": [],
        "notes": "Math - no memory retrieval needed",
    },
    {
        "message": "Good morning!",
        "context": ["Yesterday was a rough day for the user"],
        "expected_queries": [],
        "notes": "Greeting - context doesn't change that",
    },
    # -------------------------------------------------------------------------
    # Complex / edge cases
    # -------------------------------------------------------------------------
    {
        "message": "So David finally asked Jennifer out",
        "context": [
            "David is user's coworker",
            "Jennifer is David's crush",
            "User mentioned David was nervous about asking",
        ],
        "expected_queries": [
            {
                "query_text": "David",
                "query_type": "entity_overview",
                "reference": "David",
                "reasoning": "Need David's context",
            },
            {
                "query_text": "Jennifer",
                "query_type": "entity_overview",
                "reference": "Jennifer",
                "reasoning": "Need Jennifer's context",
            },
            {
                "query_text": "David asking Jennifer out",
                "query_type": "continuity",
                "reference": "finally (ongoing situation)",
                "reasoning": "This was an anticipated event",
            },
        ],
        "notes": "Multiple entities and relationship tracking",
    },
    {
        "message": "I'm not sure what to do about the car",
        "context": [
            "User's car has been having engine problems",
            "Mechanic quoted $2000 for repairs",
        ],
        "expected_queries": [
            {
                "query_text": "car problems",
                "query_type": "continuity",
                "reference": "the car",
                "reasoning": "Ongoing car issue context",
            },
        ],
        "notes": "Implicit reference to known issue",
    },
    {
        "message": "My sister and I had a long talk",
        "context": [
            "User's sister is going through a divorce",
            "They haven't talked in a while",
        ],
        "expected_queries": [
            {
                "query_text": "sister",
                "query_type": "entity_overview",
                "reference": "my sister",
                "reasoning": "Need sister's background",
            },
            {
                "query_text": "sister divorce situation",
                "query_type": "continuity",
                "reference": "divorce context",
                "reasoning": "Likely topic of their conversation",
            },
            {
                "query_text": "relationship with sister",
                "query_type": "continuity",
                "reference": "haven't talked",
                "reasoning": "Communication pattern context",
            },
        ],
        "notes": "Family situation with multiple relevant contexts",
    },
    {
        "message": "The meeting went well but I'm exhausted",
        "context": [
            "User had an important presentation today",
            "User was anxious about it",
        ],
        "expected_queries": [
            {
                "query_text": "presentation/meeting today",
                "query_type": "continuity",
                "reference": "the meeting",
                "reasoning": "Follow up on the anticipated meeting",
            },
        ],
        "notes": "Resolution of anticipated event with emotional state",
    },
    # -------------------------------------------------------------------------
    # Ambiguous cases
    # -------------------------------------------------------------------------
    {
        "message": "What do you know about machine learning?",
        "context": [],
        "expected_queries": [],
        "notes": "General knowledge question - no memory retrieval",
    },
    {
        "message": "What do you know about my project?",
        "context": ["User is working on a Python web app"],
        "expected_queries": [
            {
                "query_text": "user project",
                "query_type": "entity_overview",
                "reference": "my project",
                "reasoning": "Asking for all known info about their project",
            },
        ],
        "notes": "Personal vs general knowledge - 'my' makes it personal",
    },
    {
        "message": "How's everything going?",
        "context": [
            "User started a new job last month",
            "User has been house hunting",
        ],
        "expected_queries": [
            {
                "query_text": "user recent life updates",
                "query_type": "continuity",
                "reference": "everything",
                "reasoning": "General check-in likely refers to ongoing situations",
            },
        ],
        "notes": "Vague but context suggests retrieval would help",
    },
]


def build_extraction_dataset() -> ExtractionDataset:
    """Build the extraction dataset from hand-crafted examples."""
    examples: list[LabeledExtractionExample] = []

    for item in EXTRACTION_EXAMPLES:
        message = str(item["message"])
        context_raw = item.get("context", [])
        context = [str(c) for c in context_raw] if isinstance(context_raw, list) else []

        expected_queries: list[ExtractedQuery] = []
        queries_raw = item.get("expected_queries", [])
        if isinstance(queries_raw, list):
            for q in queries_raw:
                if isinstance(q, dict):
                    query_type_str = str(q.get("query_type", "proactive_context"))
                    try:
                        query_type = QueryType(query_type_str)
                    except ValueError:
                        query_type = QueryType.PROACTIVE_CONTEXT

                    expected_queries.append(
                        ExtractedQuery(
                            query_text=str(q.get("query_text", "")),
                            query_type=query_type,
                            reference=str(q.get("reference", "")),
                            reasoning=str(q.get("reasoning", "")),
                        )
                    )

        notes = str(item.get("notes", ""))

        examples.append(
            LabeledExtractionExample(
                message=message,
                context=context,
                expected_queries=expected_queries,
                notes=notes,
            )
        )

    return ExtractionDataset(
        examples=examples,
        name="v1",
        description="Hand-crafted extraction examples covering all query types",
    )


def save_extraction_dataset(dataset: ExtractionDataset, output_dir: Path) -> None:
    """Save extraction dataset to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"extraction_{dataset.name}.json"

    data = {
        "name": dataset.name,
        "description": dataset.description,
        "examples": [
            {
                "message": ex.message,
                "context": ex.context,
                "expected_queries": [
                    {
                        "query_text": q.query_text,
                        "query_type": q.query_type.value,
                        "reference": q.reference,
                        "reasoning": q.reasoning,
                    }
                    for q in ex.expected_queries
                ],
                "notes": ex.notes,
            }
            for ex in dataset.examples
        ],
        "distribution": {
            k.value: v for k, v in dataset.get_query_type_distribution().items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(dataset.examples)} examples to {output_path}")


def load_extraction_dataset(path: Path) -> ExtractionDataset:
    """Load extraction dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: list[LabeledExtractionExample] = []
    for item in data["examples"]:
        expected_queries: list[ExtractedQuery] = []
        for q in item.get("expected_queries", []):
            expected_queries.append(
                ExtractedQuery(
                    query_text=q["query_text"],
                    query_type=QueryType(q["query_type"]),
                    reference=q["reference"],
                    reasoning=q["reasoning"],
                )
            )
        examples.append(
            LabeledExtractionExample(
                message=item["message"],
                context=item.get("context", []),
                expected_queries=expected_queries,
                notes=item.get("notes", ""),
            )
        )

    return ExtractionDataset(
        examples=examples,
        name=data.get("name", ""),
        description=data.get("description", ""),
    )


def main() -> None:
    """Create and save the extraction dataset."""
    logging.basicConfig(level=logging.INFO)

    # Build dataset
    logger.info("Building extraction dataset...")
    dataset = build_extraction_dataset()

    # Print statistics
    print("\n" + "=" * 60)
    print("Extraction Dataset Statistics")
    print("=" * 60)
    print(f"Total examples: {len(dataset.examples)}")

    # Count examples by type
    no_retrieval_count = sum(
        1 for ex in dataset.examples if len(ex.expected_queries) == 0
    )
    print(f"No-retrieval examples: {no_retrieval_count}")
    print(f"Retrieval examples: {len(dataset.examples) - no_retrieval_count}")

    # Query type distribution
    print("\nQuery type distribution:")
    distribution = dataset.get_query_type_distribution()
    for qt, count in sorted(distribution.items(), key=lambda x: -x[1]):
        print(f"  {qt.value}: {count}")

    total_queries = sum(distribution.values())
    print(f"\nTotal queries across all examples: {total_queries}")

    # Save dataset
    experiment_dir = Path(__file__).parent
    dataset_dir = experiment_dir / "output" / "dataset"
    save_extraction_dataset(dataset, dataset_dir)

    print(f"\nDataset saved to {dataset_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
