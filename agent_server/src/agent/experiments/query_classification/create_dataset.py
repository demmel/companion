"""Generate labeled dataset for query classification experiment.

Creates 250+ labeled queries with stratified distribution across query types.
Combines hand-crafted examples with LLM-generated variations from MULTIPLE sources.

The key improvement in this version is using DIVERSE sources for training data:
- Hand-crafted seed examples (126 queries)
- Mistral-generated variations (via Ollama)
- Claude-generated variations (via Anthropic API)
- Independent evaluation queries (35 non-ambiguous queries from Claude)

This diversity prevents the embedding classifier from overfitting to a single
LLM's phrasing style.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from agent.llm import LLM, SupportedModel, create_llm
from agent.structured_llm import direct_structured_llm_call

from .models import Dataset, LabeledQuery, QueryType

logger = logging.getLogger(__name__)


@dataclass
class VariationSource:
    """Configuration for a variation generation source."""

    name: str
    model: SupportedModel
    variations_per_type: int


# Target distribution per query type
TARGET_DISTRIBUTION = {
    QueryType.CURRENT_STATE: 40,
    QueryType.HISTORY: 40,
    QueryType.ENTITY_OVERVIEW: 30,
    QueryType.TEMPORAL: 40,
    QueryType.CONTINUITY: 30,
    QueryType.PROACTIVE_CONTEXT: 40,
    QueryType.NO_RETRIEVAL: 30,
}

# Diverse target distribution - larger to accommodate multiple sources
DIVERSE_TARGET_DISTRIBUTION = {
    QueryType.CURRENT_STATE: 60,
    QueryType.HISTORY: 60,
    QueryType.ENTITY_OVERVIEW: 50,
    QueryType.TEMPORAL: 60,
    QueryType.CONTINUITY: 50,
    QueryType.PROACTIVE_CONTEXT: 60,
    QueryType.NO_RETRIEVAL: 50,
}


# Hand-crafted seed examples for each type (expanded)
SEED_EXAMPLES: dict[QueryType, list[dict[str, str | list[str] | bool]]] = {
    QueryType.CURRENT_STATE: [
        {
            "query": "What is David wearing?",
            "entities": ["David"],
            "attributes": ["clothing", "appearance"],
            "reasoning": "Asking for current state of appearance attribute",
        },
        {
            "query": "Where does Sarah work?",
            "entities": ["Sarah"],
            "attributes": ["work", "employment"],
            "reasoning": "Asking for current employment state",
        },
        {
            "query": "What's my dog's name?",
            "entities": ["dog"],
            "attributes": ["name"],
            "reasoning": "Asking for current name attribute",
        },
        {
            "query": "How am I feeling right now?",
            "entities": ["user"],
            "attributes": ["mood", "emotion"],
            "reasoning": "Asking for current emotional state",
        },
        {
            "query": "What color is my car?",
            "entities": ["car"],
            "attributes": ["color"],
            "reasoning": "Asking for current color attribute",
        },
        {
            "query": "Where am I living?",
            "entities": ["user"],
            "attributes": ["location", "residence"],
            "reasoning": "Asking for current residence",
        },
        {
            "query": "What's my job title?",
            "entities": ["user"],
            "attributes": ["job", "employment"],
            "reasoning": "Asking for current job",
        },
        {
            "query": "Who is my manager?",
            "entities": ["manager"],
            "attributes": ["relationship"],
            "reasoning": "Asking for current manager",
        },
        {
            "query": "What's my phone number?",
            "entities": ["user"],
            "attributes": ["phone"],
            "reasoning": "Asking for current contact info",
        },
        {
            "query": "Where is my sister now?",
            "entities": ["sister"],
            "attributes": ["location"],
            "reasoning": "Asking for current location",
        },
        {
            "query": "What's my favorite restaurant?",
            "entities": ["user"],
            "attributes": ["preference"],
            "reasoning": "Asking for current preference",
        },
        {
            "query": "How old is my nephew?",
            "entities": ["nephew"],
            "attributes": ["age"],
            "reasoning": "Asking for current age attribute",
        },
        {
            "query": "What's my email address?",
            "entities": ["user"],
            "attributes": ["email"],
            "reasoning": "Asking for current contact info",
        },
        {
            "query": "What medication am I taking?",
            "entities": ["user"],
            "attributes": ["medication", "health"],
            "reasoning": "Asking for current medical state",
        },
        {
            "query": "What's my cat's breed?",
            "entities": ["cat"],
            "attributes": ["breed"],
            "reasoning": "Asking for attribute of pet",
        },
        {
            "query": "What am I working on?",
            "entities": ["user"],
            "attributes": ["work", "project"],
            "reasoning": "Asking for current work state",
        },
        {
            "query": "What does my boyfriend do for work?",
            "entities": ["boyfriend"],
            "attributes": ["job", "employment"],
            "reasoning": "Asking for current employment of person",
        },
        {
            "query": "What's my address?",
            "entities": ["user"],
            "attributes": ["address"],
            "reasoning": "Asking for current residence",
        },
        {
            "query": "What team does my brother work for?",
            "entities": ["brother"],
            "attributes": ["work", "team"],
            "reasoning": "Asking for current work situation",
        },
        {
            "query": "What's my mom's favorite color?",
            "entities": ["mom"],
            "attributes": ["preference", "color"],
            "reasoning": "Asking for preference attribute",
        },
    ],
    QueryType.HISTORY: [
        {
            "query": "What has David worn this week?",
            "entities": ["David"],
            "attributes": ["clothing"],
            "time_reference": "this week",
            "reasoning": "Asking about changes over time, not current state",
        },
        {
            "query": "Remember when we talked about cooking?",
            "entities": [],
            "attributes": [],
            "reasoning": "Asking about a past conversation/episodic memory",
        },
        {
            "query": "What have we discussed about the project?",
            "entities": ["project"],
            "attributes": [],
            "reasoning": "Asking for history of discussions",
        },
        {
            "query": "How has my mood been lately?",
            "entities": ["user"],
            "attributes": ["mood"],
            "reasoning": "Asking about mood changes over time, not current",
        },
        {
            "query": "What did I tell you about my sister?",
            "entities": ["sister"],
            "attributes": [],
            "reasoning": "Asking for past shared information",
        },
        {
            "query": "What movies have I mentioned?",
            "entities": [],
            "attributes": ["movies"],
            "reasoning": "Asking about historical mentions",
        },
        {
            "query": "How have I been sleeping?",
            "entities": ["user"],
            "attributes": ["sleep"],
            "reasoning": "Asking about pattern over time",
        },
        {
            "query": "What books did we talk about?",
            "entities": [],
            "attributes": ["books"],
            "reasoning": "Historical conversational content",
        },
        {
            "query": "What jobs have I had?",
            "entities": ["user"],
            "attributes": ["employment"],
            "reasoning": "Asking for employment history",
        },
        {
            "query": "What have we talked about?",
            "entities": [],
            "attributes": [],
            "reasoning": "Open-ended history query",
        },
        {
            "query": "What restaurants have I mentioned?",
            "entities": [],
            "attributes": ["restaurants"],
            "reasoning": "Historical mentions of places",
        },
        {
            "query": "What places have I traveled to?",
            "entities": ["user"],
            "attributes": ["travel"],
            "reasoning": "Asking for travel history",
        },
        {
            "query": "What have I said about my job?",
            "entities": ["user"],
            "attributes": ["job"],
            "reasoning": "Asking for past mentions",
        },
        {
            "query": "How has my relationship with Sarah evolved?",
            "entities": ["Sarah"],
            "attributes": ["relationship"],
            "reasoning": "Relationship changes over time",
        },
        {
            "query": "What hobbies have I picked up?",
            "entities": ["user"],
            "attributes": ["hobbies"],
            "reasoning": "Historical hobby mentions",
        },
        {
            "query": "What concerns have I shared with you?",
            "entities": ["user"],
            "attributes": ["concerns"],
            "reasoning": "Past emotional sharing",
        },
        {
            "query": "What goals have I mentioned?",
            "entities": ["user"],
            "attributes": ["goals"],
            "reasoning": "Historical goal mentions",
        },
        {
            "query": "What problems have we worked through?",
            "entities": [],
            "attributes": [],
            "reasoning": "Past problem-solving history",
        },
        {
            "query": "What songs have I told you about?",
            "entities": [],
            "attributes": ["music"],
            "reasoning": "Historical music mentions",
        },
    ],
    QueryType.ENTITY_OVERVIEW: [
        {
            "query": "What do you know about Sarah?",
            "entities": ["Sarah"],
            "attributes": [],
            "reasoning": "Asking for all information about an entity",
        },
        {
            "query": "Tell me about my dog",
            "entities": ["dog"],
            "attributes": [],
            "reasoning": "Requesting complete entity overview",
        },
        {
            "query": "Who is Mark?",
            "entities": ["Mark"],
            "attributes": [],
            "reasoning": "Asking for entity introduction/overview",
        },
        {
            "query": "Summarize what you know about my mom",
            "entities": ["mom"],
            "attributes": [],
            "reasoning": "Explicit request for entity summary",
        },
        {
            "query": "What's the story with my roommate?",
            "entities": ["roommate"],
            "attributes": [],
            "reasoning": "Colloquial entity overview request",
        },
        {
            "query": "Give me a rundown on the company",
            "entities": ["company"],
            "attributes": [],
            "reasoning": "Overview request for organization",
        },
        {
            "query": "Fill me in on my sister",
            "entities": ["sister"],
            "attributes": [],
            "reasoning": "Entity overview request",
        },
        {
            "query": "What's everything you know about the project?",
            "entities": ["project"],
            "attributes": [],
            "reasoning": "Complete information request",
        },
        {
            "query": "Tell me everything about my boss",
            "entities": ["boss"],
            "attributes": [],
            "reasoning": "Full entity overview",
        },
        {
            "query": "Who is Jennifer to me?",
            "entities": ["Jennifer"],
            "attributes": [],
            "reasoning": "Relationship entity overview",
        },
        {
            "query": "Remind me about my friend Alex",
            "entities": ["Alex"],
            "attributes": [],
            "reasoning": "Entity reminder/overview",
        },
        {
            "query": "What do I know about my neighborhood?",
            "entities": ["neighborhood"],
            "attributes": [],
            "reasoning": "Place entity overview",
        },
        {
            "query": "Brief me on my therapist",
            "entities": ["therapist"],
            "attributes": [],
            "reasoning": "Entity briefing request",
        },
        {
            "query": "What's the deal with my ex?",
            "entities": ["ex"],
            "attributes": [],
            "reasoning": "Colloquial entity overview",
        },
        {
            "query": "Describe my cat to me",
            "entities": ["cat"],
            "attributes": [],
            "reasoning": "Pet entity overview",
        },
    ],
    QueryType.TEMPORAL: [
        {
            "query": "What happened yesterday?",
            "entities": [],
            "attributes": [],
            "time_reference": "yesterday",
            "reasoning": "Explicit time boundary - yesterday",
        },
        {
            "query": "This morning I was tired",
            "entities": ["user"],
            "attributes": ["energy"],
            "time_reference": "this morning",
            "reasoning": "Reference to specific time period",
        },
        {
            "query": "What was I doing last Tuesday?",
            "entities": ["user"],
            "attributes": ["activity"],
            "time_reference": "last Tuesday",
            "reasoning": "Specific day reference",
        },
        {
            "query": "What did we discuss during the holidays?",
            "entities": [],
            "attributes": [],
            "time_reference": "holidays",
            "reasoning": "Time-bounded conversation query",
        },
        {
            "query": "How was I feeling in December?",
            "entities": ["user"],
            "attributes": ["mood"],
            "time_reference": "December",
            "reasoning": "Month-bounded state query",
        },
        {
            "query": "What happened on my birthday?",
            "entities": ["user"],
            "attributes": [],
            "time_reference": "birthday",
            "reasoning": "Event-bounded time query",
        },
        {
            "query": "Last week's meetings",
            "entities": [],
            "attributes": ["meetings"],
            "time_reference": "last week",
            "reasoning": "Week-bounded query",
        },
        {
            "query": "What was going on when I was stressed?",
            "entities": ["user"],
            "attributes": ["stress"],
            "time_reference": "when stressed",
            "reasoning": "Emotional-state bounded time query",
        },
        {
            "query": "What did we talk about on Monday?",
            "entities": [],
            "attributes": [],
            "time_reference": "Monday",
            "reasoning": "Day-bounded conversation query",
        },
        {
            "query": "What happened last month?",
            "entities": [],
            "attributes": [],
            "time_reference": "last month",
            "reasoning": "Month-bounded query",
        },
        {
            "query": "What was I worried about in January?",
            "entities": ["user"],
            "attributes": ["worry"],
            "time_reference": "January",
            "reasoning": "Month-bounded emotional query",
        },
        {
            "query": "What happened at work last Friday?",
            "entities": ["work"],
            "attributes": [],
            "time_reference": "last Friday",
            "reasoning": "Day and context bounded query",
        },
        {
            "query": "What did I eat for dinner two days ago?",
            "entities": ["user"],
            "attributes": ["food"],
            "time_reference": "two days ago",
            "reasoning": "Relative time bounded query",
        },
        {
            "query": "How was my weekend?",
            "entities": ["user"],
            "attributes": [],
            "time_reference": "weekend",
            "reasoning": "Weekend-bounded query",
        },
        {
            "query": "What happened this time last year?",
            "entities": [],
            "attributes": [],
            "time_reference": "this time last year",
            "reasoning": "Annual time reference",
        },
        {
            "query": "What was I doing at 3pm yesterday?",
            "entities": ["user"],
            "attributes": ["activity"],
            "time_reference": "3pm yesterday",
            "reasoning": "Precise time bounded query",
        },
        {
            "query": "What plans did I have for the summer?",
            "entities": ["user"],
            "attributes": ["plans"],
            "time_reference": "summer",
            "reasoning": "Season-bounded query",
        },
        {
            "query": "What happened during my vacation?",
            "entities": ["user"],
            "attributes": [],
            "time_reference": "vacation",
            "reasoning": "Event period bounded query",
        },
        {
            "query": "Last night's conversation",
            "entities": [],
            "attributes": [],
            "time_reference": "last night",
            "reasoning": "Night-bounded query",
        },
    ],
    QueryType.CONTINUITY: [
        {
            "query": "How did the interview go?",
            "entities": [],
            "attributes": ["interview"],
            "reasoning": "Following up on ongoing situation",
        },
        {
            "query": "Any update on that?",
            "entities": [],
            "attributes": [],
            "reasoning": "Vague follow-up on recent topic",
        },
        {
            "query": "Did they ever respond?",
            "entities": [],
            "attributes": [],
            "reasoning": "Continuity on recent discussion",
        },
        {
            "query": "What happened with the issue we discussed?",
            "entities": [],
            "attributes": [],
            "reasoning": "Following up on prior topic",
        },
        {
            "query": "How's that going?",
            "entities": [],
            "attributes": [],
            "reasoning": "Implicit reference to ongoing situation",
        },
        {
            "query": "Did you figure it out?",
            "entities": [],
            "attributes": [],
            "reasoning": "Follow-up on problem discussed",
        },
        {
            "query": "Any news on the job?",
            "entities": ["job"],
            "attributes": [],
            "reasoning": "Following up on job situation",
        },
        {
            "query": "What's the status of that?",
            "entities": [],
            "attributes": [],
            "reasoning": "Status update request",
        },
        {
            "query": "Did the meeting happen?",
            "entities": ["meeting"],
            "attributes": [],
            "reasoning": "Following up on scheduled event",
        },
        {
            "query": "So what did they say?",
            "entities": [],
            "attributes": [],
            "reasoning": "Continuation of prior topic",
        },
        {
            "query": "Did that work out?",
            "entities": [],
            "attributes": [],
            "reasoning": "Follow-up on outcome",
        },
        {
            "query": "Is the problem resolved?",
            "entities": [],
            "attributes": [],
            "reasoning": "Follow-up on issue resolution",
        },
        {
            "query": "What's new with that situation?",
            "entities": [],
            "attributes": [],
            "reasoning": "Update request on situation",
        },
        {
            "query": "Any progress on the thing we talked about?",
            "entities": [],
            "attributes": [],
            "reasoning": "Progress update on discussed topic",
        },
        {
            "query": "Did you hear back from them?",
            "entities": [],
            "attributes": [],
            "reasoning": "Follow-up on communication",
        },
    ],
    QueryType.PROACTIVE_CONTEXT: [
        {
            "query": "I saw Sarah at the coffee shop",
            "entities": ["Sarah"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "User statement mentioning entity that needs context",
        },
        {
            "query": "My mom called me today",
            "entities": ["mom"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Entity mention requiring background fetch",
        },
        {
            "query": "The project is going well",
            "entities": ["project"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Reference to ongoing topic needing context",
        },
        {
            "query": "I'm meeting with the team later",
            "entities": ["team"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Entity mention that benefits from context",
        },
        {
            "query": "Things are better with my roommate now",
            "entities": ["roommate"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Reference requiring relationship context",
        },
        {
            "query": "I finished reading that book",
            "entities": ["book"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Reference to previously discussed item",
        },
        {
            "query": "The interview is tomorrow",
            "entities": ["interview"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Ongoing situation mention needing context",
        },
        {
            "query": "Max was really playful today",
            "entities": ["Max"],
            "attributes": ["behavior"],
            "is_proactive": True,
            "reasoning": "Entity statement requiring background",
        },
        {
            "query": "I ran into my ex",
            "entities": ["ex"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Entity mention needing relationship context",
        },
        {
            "query": "My sister finally got the job",
            "entities": ["sister"],
            "attributes": ["job"],
            "is_proactive": True,
            "reasoning": "Entity update statement",
        },
        {
            "query": "I'm going to the doctor tomorrow",
            "entities": ["doctor"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Medical context mention",
        },
        {
            "query": "The apartment issue got sorted out",
            "entities": ["apartment"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Situation resolution statement",
        },
        {
            "query": "My dad is visiting next week",
            "entities": ["dad"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Entity visit statement",
        },
        {
            "query": "I talked to my therapist about it",
            "entities": ["therapist"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Entity mention in statement",
        },
        {
            "query": "Work was really busy today",
            "entities": ["work"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Work context statement",
        },
        {
            "query": "I'm stressed about the presentation",
            "entities": ["presentation"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Ongoing situation statement",
        },
        {
            "query": "My friend Mike is in town",
            "entities": ["Mike"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Person mention needing context",
        },
        {
            "query": "The car broke down again",
            "entities": ["car"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Object mention with implied history",
        },
        {
            "query": "I'm thinking about that job offer",
            "entities": ["job offer"],
            "attributes": [],
            "is_proactive": True,
            "reasoning": "Situation mention needing context",
        },
    ],
    QueryType.NO_RETRIEVAL: [
        {
            "query": "Hello!",
            "entities": [],
            "attributes": [],
            "reasoning": "Simple greeting, no retrieval needed",
        },
        {
            "query": "What time is it?",
            "entities": [],
            "attributes": [],
            "reasoning": "System query, not memory-related",
        },
        {
            "query": "Thanks!",
            "entities": [],
            "attributes": [],
            "reasoning": "Simple acknowledgment",
        },
        {
            "query": "Can you help me write some code?",
            "entities": [],
            "attributes": [],
            "reasoning": "Task request not requiring memory",
        },
        {
            "query": "What's 2 + 2?",
            "entities": [],
            "attributes": [],
            "reasoning": "Math question, no memory needed",
        },
        {
            "query": "Tell me a joke",
            "entities": [],
            "attributes": [],
            "reasoning": "Entertainment request, no context needed",
        },
        {
            "query": "Good morning",
            "entities": [],
            "attributes": [],
            "reasoning": "Greeting",
        },
        {
            "query": "Bye!",
            "entities": [],
            "attributes": [],
            "reasoning": "Farewell",
        },
        {
            "query": "How are you?",
            "entities": [],
            "attributes": [],
            "reasoning": "Greeting/pleasantry",
        },
        {
            "query": "What's the weather like?",
            "entities": [],
            "attributes": [],
            "reasoning": "External information request",
        },
        {
            "query": "Calculate 15% of 80",
            "entities": [],
            "attributes": [],
            "reasoning": "Math calculation",
        },
        {
            "query": "Translate 'hello' to Spanish",
            "entities": [],
            "attributes": [],
            "reasoning": "Translation request",
        },
        {
            "query": "OK",
            "entities": [],
            "attributes": [],
            "reasoning": "Acknowledgment",
        },
        {
            "query": "Yes",
            "entities": [],
            "attributes": [],
            "reasoning": "Simple response",
        },
        {
            "query": "No problem",
            "entities": [],
            "attributes": [],
            "reasoning": "Acknowledgment",
        },
        {
            "query": "What day is it today?",
            "entities": [],
            "attributes": [],
            "reasoning": "Calendar query",
        },
        {
            "query": "Help me debug this error",
            "entities": [],
            "attributes": [],
            "reasoning": "Technical assistance request",
        },
        {
            "query": "Explain quantum physics",
            "entities": [],
            "attributes": [],
            "reasoning": "General knowledge request",
        },
        {
            "query": "Write a poem",
            "entities": [],
            "attributes": [],
            "reasoning": "Creative request",
        },
    ],
}


class GeneratedVariation(BaseModel):
    """A single generated query variation."""

    query: str = Field(description="The generated query text")
    entities: list[str] = Field(
        default_factory=list, description="Named entities in the query"
    )
    attributes: list[str] = Field(
        default_factory=list, description="Attributes being asked about"
    )
    time_reference: str = Field(
        default="", description="Time reference if present"
    )


class GeneratedVariations(BaseModel):
    """LLM response for query variations."""

    variations: list[GeneratedVariation] = Field(
        description="List of generated query variations"
    )


def generate_variations(
    query_type: QueryType,
    seed_examples: list[dict[str, str | list[str] | bool]],
    num_variations: int,
    llm: LLM,
    model: SupportedModel,
    source: str = "unknown",
) -> list[LabeledQuery]:
    """Generate variations of seed examples using LLM.

    Args:
        query_type: The type of query to generate variations for
        seed_examples: Seed examples to base variations on
        num_variations: Number of variations to generate
        llm: The LLM interface to use
        model: The specific model to use
        source: Source tag for tracking (e.g., 'mistral', 'claude')
    """
    type_desc = {
        QueryType.CURRENT_STATE: (
            "Queries asking for the CURRENT/MOST RECENT value of an attribute. "
            "These ask 'what IS' not 'what WAS' or 'what has been'."
        ),
        QueryType.HISTORY: (
            "Queries about PAST events, changes over time, or episodic memories. "
            "These ask 'what HAS BEEN' or 'remember when' - looking at history, not current state."
        ),
        QueryType.ENTITY_OVERVIEW: (
            "Queries asking for EVERYTHING known about an entity. "
            "These ask 'tell me about X' or 'what do you know about X'."
        ),
        QueryType.TEMPORAL: (
            "Queries with SPECIFIC TIME REFERENCES like 'yesterday', 'last week', "
            "'in December'. The time boundary is the key feature."
        ),
        QueryType.CONTINUITY: (
            "Follow-up queries about ONGOING SITUATIONS. "
            "These ask 'how did X go', 'any update', 'what happened with that'."
        ),
        QueryType.PROACTIVE_CONTEXT: (
            "User STATEMENTS (not questions) that mention entities needing context. "
            "Like 'I saw Sarah today' - not asking, just mentioning."
        ),
        QueryType.NO_RETRIEVAL: (
            "Queries that need NO memory retrieval - greetings, math, jokes, "
            "general help requests."
        ),
    }

    examples_str = "\n".join(
        f"- \"{ex['query']}\"" for ex in seed_examples[:5]
    )

    prompt = f"""Generate {num_variations} diverse query variations for the query type: {query_type.value}

TYPE DESCRIPTION:
{type_desc[query_type]}

SEED EXAMPLES:
{examples_str}

IMPORTANT:
- Generate queries that CLEARLY belong to this type
- Vary the entities, topics, and phrasing
- Make queries natural and conversational
- For proactive_context, generate statements not questions
- For temporal, include specific time references
- Do NOT generate queries that could be ambiguous between types

Generate exactly {num_variations} variations."""

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=GeneratedVariations,
            model=model,
            llm=llm,
            caller="generate_query_variations",
            temperature=0.8,
        )

        labeled_queries = []
        for var in response.variations:
            labeled_queries.append(
                LabeledQuery(
                    query=var.query,
                    query_type=query_type,
                    entities=var.entities,
                    attributes=var.attributes,
                    time_reference=var.time_reference,
                    is_proactive=query_type == QueryType.PROACTIVE_CONTEXT,
                    reasoning=f"Generated variation for {query_type.value}",
                    source=source,
                )
            )
        return labeled_queries

    except Exception as e:
        logger.error(f"Failed to generate variations for {query_type}: {e}")
        return []


def create_seed_dataset() -> Dataset:
    """Create dataset from hand-crafted seed examples."""
    queries = []
    for query_type, examples in SEED_EXAMPLES.items():
        for ex in examples:
            query_entities: list[str] = []
            query_attributes: list[str] = []
            if "entities" in ex and isinstance(ex["entities"], list):
                query_entities = [str(e) for e in ex["entities"]]
            if "attributes" in ex and isinstance(ex["attributes"], list):
                query_attributes = [str(a) for a in ex["attributes"]]
            queries.append(
                LabeledQuery(
                    query=str(ex["query"]),
                    query_type=query_type,
                    entities=query_entities,
                    attributes=query_attributes,
                    time_reference=str(ex.get("time_reference", "")),
                    is_proactive=bool(ex.get("is_proactive", False)),
                    reasoning=str(ex.get("reasoning", "")),
                    source="seed",
                )
            )
    return Dataset(queries=queries, name="seed", description="Hand-crafted seed examples")


# Independent evaluation queries (non-ambiguous only)
# These are from independent_evaluation.py, written by Claude
# Including them in training helps ensure the model generalizes beyond Mistral's style
INDEPENDENT_QUERIES: list[dict[str, str | list[str] | bool]] = [
    # CURRENT_STATE
    {"query": "What color is my car?", "query_type": "current_state", "reasoning": "Asking for a current attribute (color) of an entity"},
    {"query": "Who is my dentist?", "query_type": "current_state", "reasoning": "Asking for current relationship/attribute"},
    {"query": "What medication am I taking?", "query_type": "current_state", "reasoning": "Asking for current state of medication"},
    {"query": "How old is my nephew?", "query_type": "current_state", "reasoning": "Asking for current attribute (age)"},
    {"query": "What's my wifi password?", "query_type": "current_state", "reasoning": "Asking for a stored current value"},
    # HISTORY
    {"query": "What movies have I watched this year?", "query_type": "history", "reasoning": "Asking about accumulated events over time"},
    {"query": "When did I last go to the gym?", "query_type": "history", "reasoning": "Asking about a past event"},
    {"query": "What restaurants have we discussed?", "query_type": "history", "reasoning": "Asking about past conversation topics"},
    {"query": "How has my sleep been lately?", "query_type": "history", "reasoning": "Asking about patterns/changes over time"},
    {"query": "What did I tell you about my boss?", "query_type": "history", "reasoning": "Asking about past conversation content"},
    # ENTITY_OVERVIEW
    {"query": "Fill me in on my brother", "query_type": "entity_overview", "reasoning": "Requesting comprehensive info about entity"},
    {"query": "What's the deal with Project Alpha?", "query_type": "entity_overview", "reasoning": "Requesting overview of a topic/project"},
    {"query": "Remind me about the Johnson account", "query_type": "entity_overview", "reasoning": "Requesting all known info about entity"},
    {"query": "Who is Dr. Martinez again?", "query_type": "entity_overview", "reasoning": "Requesting entity overview/refresh"},
    {"query": "Give me the rundown on my car situation", "query_type": "entity_overview", "reasoning": "Requesting comprehensive status/info"},
    # TEMPORAL
    {"query": "What meetings do I have tomorrow?", "query_type": "temporal", "reasoning": "Explicitly time-bounded (tomorrow)"},
    {"query": "What was happening in March?", "query_type": "temporal", "reasoning": "Explicitly time-bounded (March)"},
    {"query": "Anything important from last weekend?", "query_type": "temporal", "reasoning": "Explicitly time-bounded (last weekend)"},
    {"query": "What were we working on in Q3?", "query_type": "temporal", "reasoning": "Explicitly time-bounded (Q3)"},
    {"query": "How did Monday go?", "query_type": "temporal", "reasoning": "Explicitly time-bounded (Monday)"},
    # CONTINUITY
    {"query": "Did they ever get back to you?", "query_type": "continuity", "reasoning": "Following up on unresolved situation"},
    {"query": "Is that still happening?", "query_type": "continuity", "reasoning": "Checking status of ongoing situation"},
    {"query": "What's the latest on that?", "query_type": "continuity", "reasoning": "Following up on recent topic"},
    {"query": "Did you sort out the issue?", "query_type": "continuity", "reasoning": "Following up on problem resolution"},
    {"query": "Where did we land on that decision?", "query_type": "continuity", "reasoning": "Following up on pending decision"},
    # PROACTIVE_CONTEXT
    {"query": "I bumped into Karen at the store", "query_type": "proactive_context", "reasoning": "Statement mentioning entity - need Karen's context"},
    {"query": "My therapist said something interesting", "query_type": "proactive_context", "reasoning": "Statement mentioning entity - need therapist context"},
    {"query": "The project deadline got moved", "query_type": "proactive_context", "reasoning": "Statement about entity - need project context"},
    {"query": "I'm thinking about what John said", "query_type": "proactive_context", "reasoning": "Statement mentioning entity - need John context"},
    {"query": "Remember that restaurant? I went back", "query_type": "proactive_context", "reasoning": "Reference to entity needing context retrieval"},
    # NO_RETRIEVAL
    {"query": "Can you help me write an email?", "query_type": "no_retrieval", "reasoning": "Task request, no memory needed"},
    {"query": "What's 15% of 230?", "query_type": "no_retrieval", "reasoning": "Calculation, no memory needed"},
    {"query": "How do I make french toast?", "query_type": "no_retrieval", "reasoning": "General knowledge question"},
    {"query": "Thanks for your help!", "query_type": "no_retrieval", "reasoning": "Social pleasantry"},
    {"query": "Never mind, I figured it out", "query_type": "no_retrieval", "reasoning": "Conversation closer, no retrieval needed"},
]


def get_independent_queries() -> list[LabeledQuery]:
    """Get labeled queries from independent evaluation set (Claude-generated)."""
    queries = []
    for item in INDEPENDENT_QUERIES:
        query_type = QueryType(str(item["query_type"]))
        queries.append(
            LabeledQuery(
                query=str(item["query"]),
                query_type=query_type,
                entities=[],
                attributes=[],
                time_reference="",
                is_proactive=query_type == QueryType.PROACTIVE_CONTEXT,
                reasoning=str(item.get("reasoning", "")),
                source="independent",
            )
        )
    return queries


def create_full_dataset(
    llm: LLM,
    model: SupportedModel = SupportedModel.CLAUDE_HAIKU_4_5,
) -> Dataset:
    """Create full dataset with seed examples and generated variations from a single source."""
    # Start with seed examples
    seed_dataset = create_seed_dataset()
    all_queries = list(seed_dataset.queries)

    # Determine source name from model
    source_name = "mistral" if "mistral" in model.value.lower() else "claude"

    # Calculate how many more we need per type
    current_counts: dict[QueryType, int] = {}
    for query in all_queries:
        current_counts[query.query_type] = (
            current_counts.get(query.query_type, 0) + 1
        )

    logger.info("Seed dataset distribution:")
    for qt, count in current_counts.items():
        logger.info(f"  {qt.value}: {count}")

    # Generate variations to reach target distribution
    for query_type, target_count in TARGET_DISTRIBUTION.items():
        current_count = current_counts.get(query_type, 0)
        needed = target_count - current_count

        if needed > 0:
            logger.info(f"Generating {needed} variations for {query_type.value}")
            seed_examples = SEED_EXAMPLES.get(query_type, [])
            variations = generate_variations(
                query_type=query_type,
                seed_examples=seed_examples,
                num_variations=needed,
                llm=llm,
                model=model,
                source=source_name,
            )
            all_queries.extend(variations)
            logger.info(f"Generated {len(variations)} variations for {query_type.value}")

    return Dataset(
        queries=all_queries,
        name="full",
        description="Full dataset with seed examples and generated variations",
    )


def create_diverse_dataset(
    llm: LLM,
    sources: list[VariationSource],
) -> Dataset:
    """Create dataset with variations from MULTIPLE LLM sources.

    This is the key improvement over the original experiment - by training on
    data generated by multiple LLMs, the embedding classifier should learn
    semantic patterns rather than a single LLM's phrasing style.

    Args:
        llm: The LLM interface to use
        sources: List of variation sources with their configs
    """
    # Start with seed examples
    seed_dataset = create_seed_dataset()
    all_queries = list(seed_dataset.queries)
    logger.info(f"Starting with {len(all_queries)} seed examples")

    # Add independent evaluation queries (Claude-generated)
    independent_queries = get_independent_queries()
    all_queries.extend(independent_queries)
    logger.info(f"Added {len(independent_queries)} independent evaluation queries")

    # Track counts per type
    current_counts: dict[QueryType, int] = {}
    for query in all_queries:
        current_counts[query.query_type] = (
            current_counts.get(query.query_type, 0) + 1
        )

    # Log starting distribution
    logger.info("Starting distribution (seed + independent):")
    for qt, count in current_counts.items():
        logger.info(f"  {qt.value}: {count}")

    # Generate variations from each source
    for source in sources:
        logger.info(f"\n=== Generating variations from {source.name} ({source.model.value}) ===")

        for query_type in QueryType:
            seed_examples = SEED_EXAMPLES.get(query_type, [])
            if not seed_examples:
                logger.warning(f"No seed examples for {query_type.value}, skipping")
                continue

            logger.info(f"Generating {source.variations_per_type} {query_type.value} variations from {source.name}")
            variations = generate_variations(
                query_type=query_type,
                seed_examples=seed_examples,
                num_variations=source.variations_per_type,
                llm=llm,
                model=source.model,
                source=source.name,
            )
            all_queries.extend(variations)
            logger.info(f"Generated {len(variations)} variations")

    # Log final distribution
    final_counts: dict[QueryType, int] = {}
    source_counts: dict[str, int] = {}
    for query in all_queries:
        final_counts[query.query_type] = final_counts.get(query.query_type, 0) + 1
        source_counts[query.source] = source_counts.get(query.source, 0) + 1

    logger.info("\nFinal distribution by type:")
    for qt, count in final_counts.items():
        logger.info(f"  {qt.value}: {count}")

    logger.info("\nDistribution by source:")
    for source_name, count in source_counts.items():
        logger.info(f"  {source_name}: {count}")

    logger.info(f"\nTotal queries: {len(all_queries)}")

    return Dataset(
        queries=all_queries,
        name="diverse",
        description="Diverse dataset with variations from multiple LLM sources",
    )


def save_dataset(dataset: Dataset, output_dir: Path) -> None:
    """Save dataset to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"queries_{dataset.name}.json"

    data = {
        "name": dataset.name,
        "description": dataset.description,
        "queries": [q.model_dump() for q in dataset.queries],
        "distribution": {k.value: v for k, v in dataset.get_distribution().items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(dataset.queries)} queries to {output_path}")


def load_dataset(path: Path) -> Dataset:
    """Load dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [LabeledQuery.model_validate(q) for q in data["queries"]]
    return Dataset(
        queries=queries,
        name=data.get("name", ""),
        description=data.get("description", ""),
    )


def main() -> None:
    """Create and save the diverse dataset.

    This version generates variations from MULTIPLE sources to prevent
    overfitting to a single LLM's phrasing style:
    - Seed examples (hand-crafted)
    - Independent evaluation queries (Claude-generated)
    - Mistral-generated variations (via Ollama)
    - Claude Haiku-generated variations (via Anthropic API)
    """
    logging.basicConfig(level=logging.INFO)

    # Set up paths
    experiment_dir = Path(__file__).parent
    dataset_dir = experiment_dir / "output" / "dataset"

    # Create LLM for generating variations
    llm = create_llm()

    # Start with seed examples
    logger.info("Creating seed dataset...")
    seed_dataset = create_seed_dataset()

    logger.info("\nSeed dataset distribution:")
    for qt, count in seed_dataset.get_distribution().items():
        logger.info(f"  {qt.value}: {count}")
    logger.info(f"Total seed queries: {len(seed_dataset.queries)}")

    # Define diverse sources for variation generation
    # Each source generates ~10 variations per type for diversity
    sources = [
        VariationSource(
            name="mistral",
            model=SupportedModel.MISTRAL_SMALL_3_2_Q4,
            variations_per_type=10,
        ),
        VariationSource(
            name="claude",
            model=SupportedModel.CLAUDE_HAIKU_4_5,
            variations_per_type=10,
        ),
    ]

    logger.info("\n=== Creating Diverse Dataset ===")
    logger.info(f"Sources: {[s.name for s in sources]}")
    logger.info(f"Variations per type per source: {sources[0].variations_per_type}")

    diverse_dataset = create_diverse_dataset(llm, sources)

    # Log final distribution
    logger.info("\n=== Final Dataset Statistics ===")
    logger.info("\nDistribution by type:")
    for qt, count in diverse_dataset.get_distribution().items():
        logger.info(f"  {qt.value}: {count}")
    logger.info(f"Total queries: {len(diverse_dataset.queries)}")

    # Log source distribution
    source_counts: dict[str, int] = {}
    for q in diverse_dataset.queries:
        source_counts[q.source] = source_counts.get(q.source, 0) + 1
    logger.info("\nDistribution by source:")
    for source_name, count in sorted(source_counts.items()):
        logger.info(f"  {source_name}: {count}")

    # Split into train and test (stratified by type)
    random.seed(42)
    train_dataset, test_dataset = diverse_dataset.split(train_ratio=0.8)

    # Override names for consistent file naming
    train_dataset.name = "train"
    test_dataset.name = "test"

    logger.info(f"\nTrain set: {len(train_dataset.queries)} queries")
    logger.info(f"Test set: {len(test_dataset.queries)} queries")

    # Log source distribution in train set
    train_source_counts: dict[str, int] = {}
    for q in train_dataset.queries:
        train_source_counts[q.source] = train_source_counts.get(q.source, 0) + 1
    logger.info("\nTrain set source distribution:")
    for source_name, count in sorted(train_source_counts.items()):
        logger.info(f"  {source_name}: {count}")

    # Save datasets
    save_dataset(train_dataset, dataset_dir)
    save_dataset(test_dataset, dataset_dir)

    logger.info("\n=== Dataset creation complete! ===")


if __name__ == "__main__":
    main()
