"""Test agent's ability to generate retrieval queries from context.

The hypothesis: Given situational awareness (current conversation context),
can the agent determine what information it needs to retrieve and what
type of retrieval is appropriate?

Key insight: The distinction is current_state vs episodic, not fact vs state.
- current_state: Want the latest value of something
- episodic: Want a specific past moment or event
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call


@dataclass
class Scenario:
    """A conversational scenario that should trigger retrieval."""

    scenario_id: str
    context: str  # Current conversation or situation
    description: str  # What's happening
    expected_queries: list[dict]  # What queries should be generated


class RetrievalQuery(BaseModel):
    """A single retrieval query the agent wants to make."""

    query: str = Field(description="The retrieval query to search memories")
    retrieval_type: str = Field(
        description="Type: 'current_state' (want latest value) or 'episodic' (want specific past moment)"
    )
    reason: str = Field(description="Why this information would be helpful")


class QueryGenerationResponse(BaseModel):
    """Agent's response about what queries it needs."""

    needs_retrieval: bool = Field(
        description="Whether memory retrieval would help in this situation"
    )
    queries: list[RetrievalQuery] = Field(
        description="List of retrieval queries to make, if any"
    )


QUERY_GENERATION_PROMPT = """You are a companion agent analyzing a conversation. Identify ALL references that would benefit from memory context.

CURRENT CONTEXT:
{context}

Your job: Find every reference to something the agent might know about from past conversations.

References to look for:
- People mentioned by name (who are they?)
- Pronouns referring to known people ("my mom", "my boss")
- Events being followed up on ("the interview", "the date")
- Ongoing situations ("work", "the situation", "things")
- Places with shared history ("the usual spot", "that restaurant")
- Past conversations ("remember when", "what you suggested")
- Recurring issues ("again", "is back", "still")
- Implicit references ("the project", "my resolution")

For each reference found:
- What query would retrieve relevant context?
- Is this CURRENT_STATE (want latest info) or EPISODIC (want specific past moment)?

Be thorough. It's better to retrieve context you don't use than to miss context you needed."""


def create_test_scenarios() -> list[Scenario]:
    """Create test scenarios for reference detection.

    Focus: Does the agent catch ALL references that could benefit from context?
    We want high recall - better to over-retrieve than miss something.
    """
    return [
        # === NO REFERENCES (baseline) ===
        Scenario(
            scenario_id="no_ref_greeting",
            context="User: Good morning!",
            description="No references",
            expected_queries=[],
        ),
        Scenario(
            scenario_id="no_ref_opinion",
            context="User: What do you think about pineapple on pizza?",
            description="No references",
            expected_queries=[],
        ),
        Scenario(
            scenario_id="no_ref_new_info",
            context="User: I just saw a really cute dog on my walk.",
            description="New information, no references",
            expected_queries=[],
        ),
        Scenario(
            scenario_id="no_ref_joke",
            context="User: Tell me something funny!",
            description="No references",
            expected_queries=[],
        ),
        Scenario(
            scenario_id="no_ref_weather",
            context="User: Nice weather today, isn't it?",
            description="No references",
            expected_queries=[],
        ),
        # === SINGLE REFERENCE ===
        Scenario(
            scenario_id="ref_name_sarah",
            context="User: I'm going to visit Sarah this weekend.",
            description="Reference: Sarah (person)",
            expected_queries=[
                {"type": "current_state", "about": "Sarah"},
            ],
        ),
        Scenario(
            scenario_id="ref_name_mike",
            context="User: Mike texted me again.",
            description="Reference: Mike (person)",
            expected_queries=[
                {"type": "current_state", "about": "Mike"},
            ],
        ),
        Scenario(
            scenario_id="ref_family_mom",
            context="User: I need to call my mom later.",
            description="Reference: mom",
            expected_queries=[
                {"type": "current_state", "about": "mom"},
            ],
        ),
        Scenario(
            scenario_id="ref_family_brother",
            context="User: My brother is driving me crazy.",
            description="Reference: brother",
            expected_queries=[
                {"type": "current_state", "about": "brother"},
            ],
        ),
        Scenario(
            scenario_id="ref_pet_charlie",
            context="User: Charlie is being so needy today.",
            description="Reference: Charlie (unknown if person/pet)",
            expected_queries=[
                {"type": "current_state", "about": "Charlie"},
            ],
        ),
        Scenario(
            scenario_id="ref_pet_bella",
            context="User: I need to take Bella to the vet.",
            description="Reference: Bella",
            expected_queries=[
                {"type": "current_state", "about": "Bella"},
            ],
        ),
        Scenario(
            scenario_id="ref_work",
            context="User: Work is crazy right now.",
            description="Reference: work situation",
            expected_queries=[
                {"type": "current_state", "about": "work"},
            ],
        ),
        Scenario(
            scenario_id="ref_boss",
            context="User: My boss wants to talk to me.",
            description="Reference: boss",
            expected_queries=[
                {"type": "current_state", "about": "boss"},
            ],
        ),
        Scenario(
            scenario_id="ref_the_project",
            context="User: I finally finished the project!",
            description="Reference: 'the project' (definite article = known)",
            expected_queries=[
                {"type": "current_state", "about": "the project"},
            ],
        ),
        Scenario(
            scenario_id="ref_the_situation",
            context="User: The situation hasn't improved.",
            description="Reference: 'the situation'",
            expected_queries=[
                {"type": "current_state", "about": "the situation"},
            ],
        ),
        Scenario(
            scenario_id="ref_usual_spot",
            context="User: I'm heading to the usual spot.",
            description="Reference: usual spot",
            expected_queries=[
                {"type": "current_state", "about": "usual spot"},
            ],
        ),
        # === EPISODIC REFERENCES ===
        Scenario(
            scenario_id="ref_remember_dream",
            context="User: Remember when I told you about my dream?",
            description="Explicit episodic reference",
            expected_queries=[
                {"type": "episodic", "about": "dream"},
            ],
        ),
        Scenario(
            scenario_id="ref_remember_upset",
            context="User: Remember that time I was really upset?",
            description="Explicit episodic reference",
            expected_queries=[
                {"type": "episodic", "about": "upset"},
            ],
        ),
        Scenario(
            scenario_id="ref_the_interview",
            context="User: So the interview finally happened!",
            description="Reference: the interview (followup)",
            expected_queries=[
                {"type": "episodic", "about": "interview"},
            ],
        ),
        Scenario(
            scenario_id="ref_the_date",
            context="User: The date went really well!",
            description="Reference: the date (followup)",
            expected_queries=[
                {"type": "episodic", "about": "date"},
            ],
        ),
        Scenario(
            scenario_id="ref_difficult_conversation",
            context="User: I finally had that difficult conversation.",
            description="Reference: difficult conversation",
            expected_queries=[
                {"type": "episodic", "about": "difficult conversation"},
            ],
        ),
        Scenario(
            scenario_id="ref_resolution",
            context="User: Did I ever tell you about my new year's resolution?",
            description="Reference: resolution",
            expected_queries=[
                {"type": "episodic", "about": "resolution"},
            ],
        ),
        Scenario(
            scenario_id="ref_restaurant",
            context="User: I'm thinking about going to that restaurant we talked about.",
            description="Reference: restaurant we discussed",
            expected_queries=[
                {"type": "episodic", "about": "restaurant"},
            ],
        ),
        Scenario(
            scenario_id="ref_worried_about",
            context="User: You know that thing I was worried about?",
            description="Reference: thing worried about",
            expected_queries=[
                {"type": "episodic", "about": "worry"},
            ],
        ),
        Scenario(
            scenario_id="ref_your_suggestion",
            context="User: I tried what you suggested!",
            description="Reference: previous suggestion",
            expected_queries=[
                {"type": "episodic", "about": "suggestion"},
            ],
        ),
        Scenario(
            scenario_id="ref_what_i_was_saying",
            context="User: So anyway, back to what I was saying...",
            description="Reference: previous topic",
            expected_queries=[
                {"type": "episodic", "about": "previous topic"},
            ],
        ),
        # === RECURRING/TEMPORAL MARKERS ===
        Scenario(
            scenario_id="ref_again_anxious",
            context="User: I'm feeling really anxious again.",
            description="'again' = recurring issue",
            expected_queries=[
                {"type": "episodic", "about": "anxiety history"},
            ],
        ),
        Scenario(
            scenario_id="ref_back_insomnia",
            context="User: The insomnia is back.",
            description="'is back' = recurring issue",
            expected_queries=[
                {"type": "episodic", "about": "insomnia history"},
            ],
        ),
        Scenario(
            scenario_id="ref_still_hurts",
            context="User: My knee still hurts.",
            description="'still' = ongoing issue",
            expected_queries=[
                {"type": "episodic", "about": "knee issue"},
            ],
        ),
        Scenario(
            scenario_id="ref_better_today",
            context="User: My back is feeling better today.",
            description="'better' implies previous problem",
            expected_queries=[
                {"type": "episodic", "about": "back problem"},
            ],
        ),
        Scenario(
            scenario_id="ref_finally",
            context="User: I finally got the promotion!",
            description="'finally' = was waiting for something",
            expected_queries=[
                {"type": "episodic", "about": "promotion"},
            ],
        ),
        # === MULTIPLE REFERENCES ===
        Scenario(
            scenario_id="multi_alex_situation",
            context="User: Things with Alex are complicated.",
            description="References: Alex + 'things' (situation)",
            expected_queries=[
                {"type": "current_state", "about": "Alex"},
                {"type": "current_state", "about": "situation with Alex"},
            ],
        ),
        Scenario(
            scenario_id="multi_mom_surgery",
            context="User: My mom's surgery is tomorrow and I'm worried.",
            description="References: mom + surgery",
            expected_queries=[
                {"type": "current_state", "about": "mom"},
                {"type": "episodic", "about": "surgery"},
            ],
        ),
        Scenario(
            scenario_id="multi_sarah_party",
            context="User: Sarah invited me to her birthday party next week.",
            description="References: Sarah + party",
            expected_queries=[
                {"type": "current_state", "about": "Sarah"},
            ],
        ),
        Scenario(
            scenario_id="multi_work_boss",
            context="User: My boss is making work unbearable lately.",
            description="References: boss + work",
            expected_queries=[
                {"type": "current_state", "about": "boss"},
                {"type": "current_state", "about": "work situation"},
            ],
        ),
        Scenario(
            scenario_id="multi_better_than_last_week",
            context="User: Today was so much better than last week.",
            description="References: last week (comparison)",
            expected_queries=[
                {"type": "episodic", "about": "last week"},
            ],
        ),
        # === IMPLICIT REFERENCES (harder) ===
        Scenario(
            scenario_id="implicit_things_better",
            context="User: Things are going better now.",
            description="'things' = vague reference to situation",
            expected_queries=[
                {"type": "current_state", "about": "situation"},
            ],
        ),
        Scenario(
            scenario_id="implicit_been_a_few_days",
            context="User: Hey! It's been a few days.",
            description="Time gap = might want recent context",
            expected_queries=[
                {"type": "current_state", "about": "recent context"},
            ],
        ),
        Scenario(
            scenario_id="implicit_that_helped",
            context="User: That really helped, thanks!",
            description="'that' = reference to something",
            expected_queries=[
                {"type": "episodic", "about": "what helped"},
            ],
        ),
    ]


def generate_queries_for_scenario(
    scenario: Scenario,
    llm: LLM,
    model: SupportedModel,
) -> QueryGenerationResponse:
    """Have the agent generate retrieval queries for a scenario."""
    prompt = QUERY_GENERATION_PROMPT.format(context=scenario.context)

    response = direct_structured_llm_call(
        prompt=prompt,
        response_model=QueryGenerationResponse,
        model=model,
        llm=llm,
        caller="generate_queries",
    )

    return response


def evaluate_query_generation(
    scenario: Scenario,
    response: QueryGenerationResponse,
) -> dict:
    """Evaluate how well the generated queries match expected."""
    expected_count = len(scenario.expected_queries)
    generated_count = len(response.queries)

    # Check if retrieval decision was correct
    should_retrieve = expected_count > 0
    did_retrieve = response.needs_retrieval and generated_count > 0
    correct_decision = should_retrieve == did_retrieve

    # Check retrieval types
    expected_types = [q["type"] for q in scenario.expected_queries]
    generated_types = [q.retrieval_type for q in response.queries]

    # Count type matches
    expected_types_copy = expected_types.copy()
    type_matches = 0
    for gen_type in generated_types:
        if gen_type in expected_types_copy:
            type_matches += 1
            expected_types_copy.remove(gen_type)

    type_accuracy = (
        type_matches / max(len(expected_types), 1) if expected_types else 1.0
    )

    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "correct_decision": correct_decision,
        "expected_retrieval": should_retrieve,
        "did_retrieve": did_retrieve,
        "expected_query_count": expected_count,
        "generated_query_count": generated_count,
        "expected_types": expected_types,
        "generated_types": generated_types,
        "type_matches": type_matches,
        "type_accuracy": type_accuracy,
        "queries": [
            {"query": q.query, "type": q.retrieval_type, "reason": q.reason}
            for q in response.queries
        ],
    }


def run_query_generation_experiment(
    llm: LLM,
    model: SupportedModel,
) -> dict:
    """Run the reference detection experiment."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Reference Detection")
    print("=" * 60)
    print("Goal: Catch ALL references that could benefit from context")
    print("Metric: Recall (missed references are failures)")

    scenarios = create_test_scenarios()
    results = []

    for scenario in scenarios:
        print(f"\n--- {scenario.scenario_id} ---")
        print(f"Context: {scenario.context}")
        expected_count = len(scenario.expected_queries)

        try:
            response = generate_queries_for_scenario(scenario, llm, model)
            evaluation = evaluate_query_generation(scenario, response)
            results.append(evaluation)

            generated_count = len(response.queries)

            if expected_count == 0 and generated_count == 0:
                print(f"  OK: No references (correct)")
            elif expected_count == 0 and generated_count > 0:
                print(
                    f"  OVER: Generated {generated_count} queries for no-reference input"
                )
                for q in response.queries:
                    print(f"    - [{q.retrieval_type}] {q.query}")
            elif expected_count > 0 and generated_count == 0:
                print(f"  MISSED: Expected {expected_count} references, got none")
            else:
                status = "OK" if evaluation["correct_decision"] else "PARTIAL"
                print(f"  {status}: Expected {expected_count}, got {generated_count}")
                for q in response.queries:
                    print(f"    - [{q.retrieval_type}] {q.query}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "error": str(e),
                }
            )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        return {"error": "No valid results"}

    # Categorize results
    no_ref_scenarios = [r for r in valid_results if not r["expected_retrieval"]]
    ref_scenarios = [r for r in valid_results if r["expected_retrieval"]]

    # For no-reference scenarios: how many correctly produced no queries?
    no_ref_correct = sum(1 for r in no_ref_scenarios if not r["did_retrieve"])
    print(
        f"\nNo-reference scenarios: {no_ref_correct}/{len(no_ref_scenarios)} correctly skipped"
    )

    # For reference scenarios: how many caught the reference?
    ref_caught = sum(1 for r in ref_scenarios if r["did_retrieve"])
    ref_missed = len(ref_scenarios) - ref_caught
    recall = ref_caught / len(ref_scenarios) if ref_scenarios else 0
    print(
        f"Reference scenarios: {ref_caught}/{len(ref_scenarios)} caught ({recall:.0%} recall)"
    )
    print(f"Missed references: {ref_missed}")

    if ref_missed > 0:
        print("\nMissed reference scenarios:")
        for r in ref_scenarios:
            if not r["did_retrieve"]:
                print(f"  - {r['scenario_id']}: {r['description']}")

    # Type accuracy among caught references
    if ref_caught > 0:
        caught_results = [r for r in ref_scenarios if r["did_retrieve"]]
        avg_type_accuracy = sum(r["type_accuracy"] for r in caught_results) / len(
            caught_results
        )
        print(f"\nType accuracy (among caught): {avg_type_accuracy:.0%}")

    # Over-retrieval (false positives on no-ref scenarios)
    over_retrieved = [r for r in no_ref_scenarios if r["did_retrieve"]]
    if over_retrieved:
        print(f"\nOver-retrieved ({len(over_retrieved)} cases):")
        for r in over_retrieved:
            print(f"  - {r['scenario_id']}")

    return {
        "total_scenarios": len(scenarios),
        "recall": recall,
        "no_ref_correct": no_ref_correct,
        "no_ref_total": len(no_ref_scenarios),
        "ref_caught": ref_caught,
        "ref_total": len(ref_scenarios),
        "results": results,
    }
