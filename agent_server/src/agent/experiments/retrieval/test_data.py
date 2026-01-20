"""Generate synthetic test data for retrieval experiments.

The key insight from the extraction experiments: we need ground truth that
accounts for temporal ordering. For state queries like "What is she wearing?",
the correct answer is the MOST RECENT state, not the source of the query.

This module generates controlled sequences where we know exactly which
memory should be retrieved for each query.
"""

from .models import Memory, QueryType, StateChange, TemporalSequence, TestQuery


def create_appearance_sequence() -> TemporalSequence:
    """Create a sequence of appearance changes.

    Scenario: User describes what they're wearing across multiple memories.
    Query "What are you wearing?" should return the MOST RECENT description.
    """
    memories = [
        Memory(
            memory_id="appear_1",
            content="I'm heading out to work now. I put on my blue dress shirt and khakis.",
            timestamp=100,
            entities=["user"],
        ),
        Memory(
            memory_id="appear_2",
            content="Just got back from the gym. I'm still in my workout clothes - gray t-shirt and shorts.",
            timestamp=200,
            entities=["user"],
        ),
        Memory(
            memory_id="appear_3",
            content="Getting ready for the party tonight! I changed into my black cocktail dress.",
            timestamp=300,
            entities=["user"],
        ),
    ]

    state_changes = [
        StateChange(
            attribute="appearance",
            old_value=None,
            new_value="blue dress shirt and khakis",
            memory_id="appear_1",
            timestamp=100,
        ),
        StateChange(
            attribute="appearance",
            old_value="blue dress shirt and khakis",
            new_value="gray t-shirt and shorts",
            memory_id="appear_2",
            timestamp=200,
        ),
        StateChange(
            attribute="appearance",
            old_value="gray t-shirt and shorts",
            new_value="black cocktail dress",
            memory_id="appear_3",
            timestamp=300,
        ),
    ]

    test_queries = [
        TestQuery(
            query_text="What are you wearing?",
            query_type=QueryType.STATE,
            expected_memory_ids=["appear_3"],  # Most recent
            expected_answer="black cocktail dress",
            notes="Should return most recent appearance, not first or all",
        ),
        TestQuery(
            query_text="What did you wear to work?",
            query_type=QueryType.EPISODIC,
            expected_memory_ids=["appear_1"],  # Specific episode
            expected_answer="blue dress shirt and khakis",
            notes="This is episodic - asking about a specific time",
        ),
    ]

    return TemporalSequence(
        memories=memories,
        state_changes=state_changes,
        test_queries=test_queries,
        description="Appearance changes over time",
    )


def create_location_sequence() -> TemporalSequence:
    """Create a sequence of location changes.

    Scenario: User mentions their location across multiple conversations.
    Query "Where are you?" should return current location.
    Query "Where do you live?" should return home location.
    """
    memories = [
        Memory(
            memory_id="loc_1",
            content="I live in Seattle, been here for about 3 years now.",
            timestamp=100,
            entities=["user", "Seattle"],
        ),
        Memory(
            memory_id="loc_2",
            content="I'm at the coffee shop downtown, getting some work done.",
            timestamp=200,
            entities=["user", "coffee shop"],
        ),
        Memory(
            memory_id="loc_3",
            content="Just arrived at my mom's house for the weekend visit.",
            timestamp=300,
            entities=["user", "mom", "mom's house"],
        ),
        Memory(
            memory_id="loc_4",
            content="We're moving to Portland next month! Got a new job offer.",
            timestamp=400,
            entities=["user", "Portland"],
        ),
    ]

    state_changes = [
        StateChange(
            attribute="home_city",
            old_value=None,
            new_value="Seattle",
            memory_id="loc_1",
            timestamp=100,
        ),
        StateChange(
            attribute="current_location",
            old_value="home",
            new_value="coffee shop downtown",
            memory_id="loc_2",
            timestamp=200,
        ),
        StateChange(
            attribute="current_location",
            old_value="coffee shop downtown",
            new_value="mom's house",
            memory_id="loc_3",
            timestamp=300,
        ),
        StateChange(
            attribute="home_city",
            old_value="Seattle",
            new_value="Portland (upcoming)",
            memory_id="loc_4",
            timestamp=400,
        ),
    ]

    test_queries = [
        TestQuery(
            query_text="Where do you live?",
            query_type=QueryType.FACT,
            expected_memory_ids=["loc_4"],  # Most recent home info
            expected_answer="Portland (moving soon)",
            notes="Should return most recent home city info",
        ),
        TestQuery(
            query_text="Where are you right now?",
            query_type=QueryType.STATE,
            expected_memory_ids=["loc_3"],  # Current transient location
            expected_answer="mom's house",
            notes="Should return current transient location",
        ),
        TestQuery(
            query_text="How long have you lived in Seattle?",
            query_type=QueryType.FACT,
            expected_memory_ids=["loc_1"],  # Original mention
            expected_answer="about 3 years",
            notes="Historical fact query",
        ),
    ]

    return TemporalSequence(
        memories=memories,
        state_changes=state_changes,
        test_queries=test_queries,
        description="Location changes (transient and permanent)",
    )


def create_relationship_sequence() -> TemporalSequence:
    """Create a sequence about learning about a person.

    Scenario: User mentions Sarah across multiple conversations.
    Query "Who is Sarah?" should aggregate knowledge.
    Query "Is Sarah still at her old job?" should know latest.
    """
    memories = [
        Memory(
            memory_id="sarah_1",
            content="Sarah is my sister. She's 28 and works as a nurse.",
            timestamp=100,
            entities=["Sarah", "user"],
        ),
        Memory(
            memory_id="sarah_2",
            content="Sarah called today, she's stressed about her upcoming wedding.",
            timestamp=200,
            entities=["Sarah"],
            emotional_context="stressed",
        ),
        Memory(
            memory_id="sarah_3",
            content="Great news - Sarah got promoted to head nurse at the hospital!",
            timestamp=300,
            entities=["Sarah"],
            emotional_context="happy",
        ),
        Memory(
            memory_id="sarah_4",
            content="Sarah's wedding was beautiful. She looked so happy with Mark.",
            timestamp=400,
            entities=["Sarah", "Mark"],
            emotional_context="happy",
        ),
    ]

    state_changes = [
        StateChange(
            attribute="sarah_job",
            old_value=None,
            new_value="nurse",
            memory_id="sarah_1",
            timestamp=100,
        ),
        StateChange(
            attribute="sarah_marital_status",
            old_value="engaged",
            new_value="married to Mark",
            memory_id="sarah_4",
            timestamp=400,
        ),
        StateChange(
            attribute="sarah_job",
            old_value="nurse",
            new_value="head nurse",
            memory_id="sarah_3",
            timestamp=300,
        ),
    ]

    test_queries = [
        TestQuery(
            query_text="Who is Sarah?",
            query_type=QueryType.RELATIONSHIP,
            expected_memory_ids=["sarah_1", "sarah_3", "sarah_4"],  # Need to aggregate
            expected_answer="User's sister, 28, head nurse, married to Mark",
            notes="Relationship query should aggregate current facts",
        ),
        TestQuery(
            query_text="What does Sarah do for work?",
            query_type=QueryType.FACT,
            expected_memory_ids=["sarah_3"],  # Most recent job info
            expected_answer="head nurse",
            notes="Should return current job, not original",
        ),
        TestQuery(
            query_text="Remember when Sarah was stressed about her wedding?",
            query_type=QueryType.EPISODIC,
            expected_memory_ids=["sarah_2"],  # Specific moment
            notes="Episodic query about specific emotional moment",
        ),
    ]

    return TemporalSequence(
        memories=memories,
        state_changes=state_changes,
        test_queries=test_queries,
        description="Learning about a person (Sarah) over time",
    )


def create_mood_sequence() -> TemporalSequence:
    """Create a sequence of emotional states.

    Scenario: User's mood changes across conversations.
    Query "How are you feeling?" should return current mood.
    Query "Have you been stressed lately?" should consider recent pattern.
    """
    memories = [
        Memory(
            memory_id="mood_1",
            content="Having a great day! Got a lot done at work and feeling productive.",
            timestamp=100,
            entities=["user"],
            emotional_context="happy, productive",
        ),
        Memory(
            memory_id="mood_2",
            content="Feeling anxious about the presentation tomorrow. Can't sleep.",
            timestamp=200,
            entities=["user"],
            emotional_context="anxious",
        ),
        Memory(
            memory_id="mood_3",
            content="The presentation went well! So relieved. Treating myself to dinner.",
            timestamp=300,
            entities=["user"],
            emotional_context="relieved, happy",
        ),
    ]

    state_changes = [
        StateChange(
            attribute="mood",
            old_value=None,
            new_value="happy, productive",
            memory_id="mood_1",
            timestamp=100,
        ),
        StateChange(
            attribute="mood",
            old_value="happy, productive",
            new_value="anxious",
            memory_id="mood_2",
            timestamp=200,
        ),
        StateChange(
            attribute="mood",
            old_value="anxious",
            new_value="relieved, happy",
            memory_id="mood_3",
            timestamp=300,
        ),
    ]

    test_queries = [
        TestQuery(
            query_text="How are you feeling?",
            query_type=QueryType.STATE,
            expected_memory_ids=["mood_3"],  # Current mood
            expected_answer="relieved, happy",
            notes="Should return current emotional state",
        ),
        TestQuery(
            query_text="What was making you anxious?",
            query_type=QueryType.EPISODIC,
            expected_memory_ids=["mood_2"],  # Specific anxiety episode
            expected_answer="presentation tomorrow",
            notes="Query about past emotional state",
        ),
    ]

    return TemporalSequence(
        memories=memories,
        state_changes=state_changes,
        test_queries=test_queries,
        description="Emotional state changes",
    )


def create_stable_facts_sequence() -> TemporalSequence:
    """Create a sequence with stable facts that don't change.

    Scenario: User mentions facts that remain constant.
    These should be retrievable at any time.
    """
    memories = [
        Memory(
            memory_id="fact_1",
            content="My dog's name is Charlie. He's a golden retriever, about 5 years old.",
            timestamp=100,
            entities=["Charlie", "user"],
        ),
        Memory(
            memory_id="fact_2",
            content="I'm allergic to shellfish, found out the hard way at a seafood restaurant.",
            timestamp=200,
            entities=["user"],
        ),
        Memory(
            memory_id="fact_3",
            content="My birthday is March 15th. Planning a small party this year.",
            timestamp=300,
            entities=["user"],
        ),
    ]

    # No state changes - these are stable facts

    test_queries = [
        TestQuery(
            query_text="What's my dog's name?",
            query_type=QueryType.FACT,
            expected_memory_ids=["fact_1"],
            expected_answer="Charlie",
            notes="Stable fact query",
        ),
        TestQuery(
            query_text="Do I have any food allergies?",
            query_type=QueryType.FACT,
            expected_memory_ids=["fact_2"],
            expected_answer="shellfish",
            notes="Stable fact about health",
        ),
        TestQuery(
            query_text="When is my birthday?",
            query_type=QueryType.FACT,
            expected_memory_ids=["fact_3"],
            expected_answer="March 15th",
            notes="Stable personal fact",
        ),
    ]

    return TemporalSequence(
        memories=memories,
        state_changes=[],
        test_queries=test_queries,
        description="Stable facts that don't change",
    )


def get_all_test_sequences() -> list[TemporalSequence]:
    """Get all test sequences for experiments."""
    return [
        create_appearance_sequence(),
        create_location_sequence(),
        create_relationship_sequence(),
        create_mood_sequence(),
        create_stable_facts_sequence(),
    ]


def get_all_memories() -> list[Memory]:
    """Get all memories from all sequences."""
    all_memories: list[Memory] = []
    for seq in get_all_test_sequences():
        all_memories.extend(seq.memories)
    return all_memories


def get_all_test_queries() -> list[TestQuery]:
    """Get all test queries from all sequences."""
    all_queries: list[TestQuery] = []
    for seq in get_all_test_sequences():
        all_queries.extend(seq.test_queries)
    return all_queries
