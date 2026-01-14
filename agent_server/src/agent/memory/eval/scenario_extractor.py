"""Tools for extracting evaluation scenarios from conversation logs."""

import json
from pathlib import Path
from typing import Optional

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.memory.memory import MemoryQueries, MemoryQuery, QueryType

from .data_models import EvalScenario


def load_triggers_from_file(filepath: Path) -> list[TriggerHistoryEntry]:
    """Load trigger history entries from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for entry_data in data["entries"]:
        entry = TriggerHistoryEntry.model_validate(entry_data)
        entries.append(entry)

    return entries


def create_test_query(
    query_text: str,
    query_type: QueryType = QueryType.FACTUAL,
    max_tokens: int = 2000,
) -> MemoryQueries:
    """Create a simple test query."""
    return MemoryQueries(
        queries=[
            MemoryQuery(
                reasoning="Evaluation test query",
                query_type=query_type,
                query_text=query_text,
                importance=1.0,
            )
        ],
        max_tokens=max_tokens,
    )


def extract_scenario(
    triggers_filepath: Path,
    scenario_id: str,
    name: str,
    description: str,
    test_turn_index: int,
    test_query: MemoryQueries,
    expected_information: list[str],
) -> EvalScenario:
    """
    Extract an evaluation scenario from a conversation log.

    Args:
        triggers_filepath: Path to the triggers JSON file
        scenario_id: Unique ID for this scenario
        name: Human-readable name
        description: What this scenario tests
        test_turn_index: Index of the turn to test (use triggers up to this point)
        test_query: The query to run after replaying
        expected_information: Natural language descriptions of expected info

    Returns:
        An EvalScenario ready for evaluation
    """
    triggers = load_triggers_from_file(triggers_filepath)

    # Take triggers up to (but not including) the test turn
    trigger_history = triggers[:test_turn_index]

    return EvalScenario(
        scenario_id=scenario_id,
        name=name,
        description=description,
        trigger_history=trigger_history,
        test_query=test_query,
        expected_information=expected_information,
    )


def save_scenario(scenario: EvalScenario, filepath: Path) -> None:
    """Save a scenario to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(scenario.model_dump_json(indent=2))


def load_scenario(filepath: Path) -> EvalScenario:
    """Load a scenario from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return EvalScenario.model_validate(data)


def load_all_scenarios(scenarios_dir: Path) -> list[EvalScenario]:
    """Load all scenarios from a directory."""
    scenarios = []

    for filepath in scenarios_dir.glob("*.json"):
        scenario = load_scenario(filepath)
        scenarios.append(scenario)

    return scenarios
