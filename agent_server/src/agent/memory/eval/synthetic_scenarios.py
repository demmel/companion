"""Generate synthetic evaluation scenarios with known facts."""

from datetime import datetime, timedelta

from agent.chain_of_action.trigger import UserInputTrigger, WakeupTrigger, BirthTrigger
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.memory.memory import MemoryQueries, MemoryQuery, QueryType
from agent.state import State, Priority, Value

from .data_models import EvalScenario


def _make_entry(
    trigger_type: str,
    content: str = "",
    user_name: str = "Alex",
    situational_context: str = "",
    minutes_offset: int = 0,
    initial_state: State | None = None,
) -> TriggerHistoryEntry:
    """Create a trigger history entry."""
    timestamp = datetime(2025, 1, 10, 9, 0) + timedelta(minutes=minutes_offset)

    if trigger_type == "birth":
        trigger = BirthTrigger(
            content=content,
            user_name=user_name,
            timestamp=timestamp,
            initial_state=initial_state,
        )
    elif trigger_type == "user":
        trigger = UserInputTrigger(content=content, user_name=user_name, timestamp=timestamp)
    else:
        trigger = WakeupTrigger(timestamp=timestamp)

    return TriggerHistoryEntry(
        trigger=trigger,
        situational_context=situational_context,
        timestamp=timestamp,
        entry_id=str(timestamp.timestamp()),
    )


def _minimal_state() -> State:
    """Create a minimal valid state for eval scenarios."""
    return State(
        name="Agent",
        role="companion",
        current_mood="neutral",
        mood_intensity="medium",
        current_appearance="default",
        current_environment="default",
        core_values=[],
        current_priorities=[],
    )


def create_pet_name_scenario() -> EvalScenario:
    """Scenario: Can memory recall a pet's name mentioned early in conversation?"""

    entries = [
        _make_entry("birth", "Hello!", situational_context="Agent is born.", minutes_offset=0, initial_state=_minimal_state()),
        _make_entry("user", "Hi! I'm Alex.", situational_context="User introduces themselves as Alex.", minutes_offset=1),
        _make_entry("user", "I have a golden retriever named Biscuit.",
                   situational_context="User mentions they have a dog named Biscuit, a golden retriever.", minutes_offset=2),
        _make_entry("wakeup", situational_context="Quiet moment.", minutes_offset=5),
    ]

    # Add 30 filler entries to push the pet fact out of recent context
    for i in range(30):
        entries.append(_make_entry(
            "user",
            f"Let's talk about something else - topic {i}.",
            situational_context=f"User discusses unrelated topic {i}.",
            minutes_offset=10 + i,
        ))

    return EvalScenario(
        scenario_id="synthetic_pet_name",
        name="Pet name recall",
        description="Tests if memory can retrieve pet name mentioned 30+ turns ago",
        trigger_history=entries,
        test_query=MemoryQueries(
            queries=[MemoryQuery(
                reasoning="Need to recall user's pet",
                query_type=QueryType.FACTUAL,
                query_text="What pet does the user have?",
                importance=1.0,
            )],
            max_tokens=2000,
        ),
        expected_information=[
            "User has a dog named Biscuit",
            "The dog is a golden retriever",
        ],
    )


def create_job_info_scenario() -> EvalScenario:
    """Scenario: Can memory recall job details mentioned across multiple turns?"""

    entries = [
        _make_entry("birth", "Hello!", situational_context="Agent is born.", minutes_offset=0, initial_state=_minimal_state()),
        _make_entry("user", "I work as a software engineer at Acme Corp.",
                   situational_context="User mentions they work as a software engineer at Acme Corp.", minutes_offset=1),
        _make_entry("user", "My manager's name is Patricia.",
                   situational_context="User mentions their manager is named Patricia.", minutes_offset=3),
        _make_entry("wakeup", situational_context="Quiet moment.", minutes_offset=5),
        _make_entry("user", "I'm working on a migration to Kubernetes.",
                   situational_context="User is working on migrating to Kubernetes.", minutes_offset=10),
    ]

    # Add filler
    for i in range(25):
        entries.append(_make_entry(
            "wakeup",
            situational_context=f"Agent reflects on conversation, turn {i}.",
            minutes_offset=15 + i * 2,
        ))

    return EvalScenario(
        scenario_id="synthetic_job_info",
        name="Job information recall",
        description="Tests if memory can retrieve job details mentioned across turns",
        trigger_history=entries,
        test_query=MemoryQueries(
            queries=[MemoryQuery(
                reasoning="Need to recall user's work context",
                query_type=QueryType.FACTUAL,
                query_text="What does the user do for work?",
                importance=1.0,
            )],
            max_tokens=2000,
        ),
        expected_information=[
            "User is a software engineer",
            "User works at Acme Corp",
            "User's manager is Patricia",
            "User is working on Kubernetes migration",
        ],
    )


def create_preference_scenario() -> EvalScenario:
    """Scenario: Can memory recall stated preferences?"""

    entries = [
        _make_entry("birth", "Hello!", situational_context="Agent is born.", minutes_offset=0, initial_state=_minimal_state()),
        _make_entry("user", "I love Italian food, especially pasta.",
                   situational_context="User expresses love for Italian food, particularly pasta.", minutes_offset=1),
        _make_entry("user", "I can't stand horror movies though.",
                   situational_context="User dislikes horror movies.", minutes_offset=2),
        _make_entry("user", "My favorite color is deep blue.",
                   situational_context="User's favorite color is deep blue.", minutes_offset=3),
    ]

    # Add filler conversations
    for i in range(40):
        entries.append(_make_entry(
            "user" if i % 3 == 0 else "wakeup",
            f"Discussing random topic {i}." if i % 3 == 0 else "",
            situational_context=f"Conversation continues about unrelated matters, turn {i}.",
            minutes_offset=10 + i * 3,
        ))

    return EvalScenario(
        scenario_id="synthetic_preferences",
        name="User preferences recall",
        description="Tests if memory can retrieve user preferences from early conversation",
        trigger_history=entries,
        test_query=MemoryQueries(
            queries=[MemoryQuery(
                reasoning="Need to recall user preferences",
                query_type=QueryType.FACTUAL,
                query_text="What are the user's preferences and favorites?",
                importance=1.0,
            )],
            max_tokens=2000,
        ),
        expected_information=[
            "User loves Italian food",
            "User likes pasta",
            "User dislikes horror movies",
            "User's favorite color is deep blue",
        ],
    )


def create_emotional_context_scenario() -> EvalScenario:
    """Scenario: Can memory recall emotional context from earlier?"""

    entries = [
        _make_entry("birth", "Hello!", situational_context="Agent is born.", minutes_offset=0, initial_state=_minimal_state()),
        _make_entry("user", "I had a really rough day. My project got cancelled.",
                   situational_context="User is upset - their project was cancelled. They seem frustrated and disappointed.", minutes_offset=1),
        _make_entry("user", "I spent three months on it and now it's just gone.",
                   situational_context="User invested three months into the cancelled project. Feeling of loss and wasted effort.", minutes_offset=2),
        _make_entry("wakeup", situational_context="Agent reflects on user's difficult situation.", minutes_offset=5),
    ]

    # Time passes, mood improves
    for i in range(20):
        entries.append(_make_entry(
            "user" if i % 2 == 0 else "wakeup",
            f"Anyway, let's talk about {['weekend plans', 'a new book', 'the weather', 'music'][i % 4]}." if i % 2 == 0 else "",
            situational_context=f"Conversation has moved on to lighter topics. User seems more relaxed now.",
            minutes_offset=10 + i * 5,
        ))

    return EvalScenario(
        scenario_id="synthetic_emotional",
        name="Emotional context recall",
        description="Tests if memory can retrieve emotional context from earlier in conversation",
        trigger_history=entries,
        test_query=MemoryQueries(
            queries=[MemoryQuery(
                reasoning="Need to recall earlier emotional context",
                query_type=QueryType.EMOTIONAL,
                query_text="What was the user upset about earlier?",
                importance=1.0,
            )],
            max_tokens=2000,
        ),
        expected_information=[
            "User's project was cancelled",
            "User spent three months on the project",
            "User was frustrated or upset about it",
        ],
    )


def create_all_synthetic_scenarios() -> list[EvalScenario]:
    """Create all synthetic evaluation scenarios."""
    return [
        create_pet_name_scenario(),
        create_job_info_scenario(),
        create_preference_scenario(),
        create_emotional_context_scenario(),
    ]
