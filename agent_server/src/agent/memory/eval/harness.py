"""Memory evaluation harness."""

import logging
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from agent.llm import LLM, SupportedModel
from agent.memory.memory import IMemory, MemoryQueries
from agent.memory.query_extraction import extract_memory_queries
from agent.chain_of_action.trigger_history import TriggerHistory
from agent.chain_of_action.state_replay import (
    derive_initial_state,
    replay_state,
)
from agent.config import Config

from .data_models import EvalScenario, EvalResult, EvalRun
from .llm_judge import judge_all_items, compute_recall

# Factory function type: takes trigger_history, returns memory instance
MemoryFactory = Callable[[TriggerHistory], IMemory]

def _get_cache_dir() -> Path:
    """Get the cache directory for eval results."""
    cache_dir = Path("eval_data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_file(scenario_id: str, memory_impl_name: str) -> Path:
    """Get cache file path for a scenario-memory pair."""
    return _get_cache_dir() / f"{scenario_id}_{memory_impl_name}.json"


def _load_cached_result(scenario_id: str, memory_impl_name: str) -> EvalResult | None:
    """Load cached eval result if available."""
    cache_file = _get_cache_file(scenario_id, memory_impl_name)
    if cache_file.exists():
        return EvalResult.model_validate_json(cache_file.read_text(encoding="utf-8"))
    return None


def _save_cached_result(result: EvalResult) -> None:
    """Save eval result to cache."""
    cache_file = _get_cache_file(result.scenario_id, result.memory_implementation)
    cache_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def evaluate_scenario(
    scenario: EvalScenario,
    memory_factory: MemoryFactory,
    llm: LLM,
    judge_model: SupportedModel,
    memory_implementation_name: str,
    query_extraction_model: SupportedModel,
    use_cache: bool = True,
) -> EvalResult:
    """
    Evaluate a single scenario against a memory implementation.

    Args:
        scenario: The evaluation scenario to run
        memory_factory: Factory function to create memory with trigger_history
        llm: LLM instance for memory operations and judging
        judge_model: Model to use for LLM judge
        memory_implementation_name: Name of the memory implementation
        use_cache: Whether to use cached results

    Returns:
        EvalResult with judgments and metrics
    """
    # Check cache first
    if use_cache:
        cached = _load_cached_result(scenario.scenario_id, memory_implementation_name)
        if cached is not None:
            return cached

    # Suppress verbose logging during eval
    logging.getLogger("agent.memory").setLevel(logging.ERROR)
    logging.getLogger("agent.structured_llm").setLevel(logging.ERROR)

    # Create empty trigger history and memory (like normal operation)
    trigger_history = TriggerHistory()
    memory = memory_factory(trigger_history)

    # Get model config (same as production)
    model_config = Config.get_model_config()

    # Token budget matches production default
    token_budget = int(
        llm.models[model_config.situational_analysis_model].context_window * 0.7
    )

    # Replay conversation matching real operation: query → store for each trigger
    initial_state = derive_initial_state(scenario.trigger_history[0])
    memory_context = ""

    num_entries = len(scenario.trigger_history)
    for i, (entry, state) in enumerate(replay_state(scenario.trigger_history, initial_state)):
        print(f"  replaying {i+1}/{num_entries}...", end=" ", flush=True)
        trigger_history.add_trigger_entry(entry)

        # Extract queries based on current memory context
        query_result = extract_memory_queries(
            trigger=entry.trigger,
            state=state,
            context=memory_context,
            llm=llm,
            model=query_extraction_model,
            max_queries=6,
        )

        # Query memory with extracted queries (same as production)
        memory_context = memory.query(
            MemoryQueries(
                queries=query_result.queries,
                max_tokens=token_budget,
            ),
            llm=llm,
            model=query_extraction_model,
        )

        # Store after query
        memory.store(entry, state, llm, query_extraction_model)
        print("done")

    # Final query with test query is the one we evaluate
    print(f"  final query...", end=" ", flush=True)
    start_time = time.perf_counter()
    output = memory.query(scenario.test_query, llm=llm, model=query_extraction_model)
    end_time = time.perf_counter()
    retrieval_time_ms = (end_time - start_time) * 1000
    print(f"{retrieval_time_ms:.0f}ms")

    # LLM judges if expected information is present
    print(f"  judging...", end=" ", flush=True)
    judgments = judge_all_items(
        memory_output=output,
        expected_items=scenario.expected_information,
        llm=llm,
        model=judge_model,
    )
    print("done")

    # 4. Compute recall
    recall = compute_recall(judgments)

    result = EvalResult(
        scenario_id=scenario.scenario_id,
        memory_implementation=memory_implementation_name,
        output=output,
        judgments=judgments,
        recall=recall,
        retrieval_time_ms=retrieval_time_ms,
    )

    # Cache the result
    if use_cache:
        _save_cached_result(result)

    return result


def run_evaluation(
    scenarios: list[EvalScenario],
    memory_factories: Mapping[str, MemoryFactory],
    llm: LLM,
    judge_model: SupportedModel,
    query_extraction_model: SupportedModel,
    use_cache: bool = True,
) -> EvalRun:
    """
    Run a complete evaluation across scenarios and implementations.

    Args:
        scenarios: List of scenarios to evaluate
        memory_factories: Dict of name -> factory function to create memory
        llm: LLM instance
        judge_model: Model to use for LLM judge
        use_cache: Whether to use cached results

    Returns:
        EvalRun with all results and aggregate metrics
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now(timezone.utc).isoformat()

    results: list[EvalResult] = []

    total = len(scenarios) * len(memory_factories)
    current = 0

    for scenario in scenarios:
        for impl_name, factory in memory_factories.items():
            current += 1
            print(f"[{current}/{total}] {scenario.scenario_id} + {impl_name}")

            result = evaluate_scenario(
                scenario=scenario,
                memory_factory=factory,
                llm=llm,
                judge_model=judge_model,
                memory_implementation_name=impl_name,
                query_extraction_model=query_extraction_model,
                use_cache=use_cache,
            )
            results.append(result)

            print(f"  -> recall={result.recall:.2f}")

    # Compute aggregates per implementation
    aggregate_recall: dict[str, float] = {}
    aggregate_time_ms: dict[str, float] = {}

    for impl_name in memory_factories:
        impl_results = [r for r in results if r.memory_implementation == impl_name]
        if impl_results:
            aggregate_recall[impl_name] = sum(r.recall for r in impl_results) / len(
                impl_results
            )
            aggregate_time_ms[impl_name] = sum(
                r.retrieval_time_ms for r in impl_results
            ) / len(impl_results)

    return EvalRun(
        run_id=run_id,
        timestamp=timestamp,
        memory_implementations=list(memory_factories.keys()),
        results=results,
        aggregate_recall=aggregate_recall,
        aggregate_time_ms=aggregate_time_ms,
    )


def print_eval_summary(eval_run: EvalRun) -> None:
    """Print a summary of evaluation results."""
    print(f"\n=== Evaluation Results ({eval_run.run_id}) ===\n")

    # Group results by scenario
    scenario_ids = set(r.scenario_id for r in eval_run.results)

    for scenario_id in sorted(scenario_ids):
        scenario_results = [r for r in eval_run.results if r.scenario_id == scenario_id]
        print(f"Scenario: {scenario_id}")

        for result in scenario_results:
            print(
                f"  {result.memory_implementation:20s}  "
                f"recall={result.recall:.2f}  "
                f"time={result.retrieval_time_ms:.1f}ms"
            )
        print()

    print("Aggregate (all scenarios):")
    for impl_name in eval_run.memory_implementations:
        recall = eval_run.aggregate_recall.get(impl_name, 0.0)
        time_ms = eval_run.aggregate_time_ms.get(impl_name, 0.0)
        print(f"  {impl_name:20s}  mean_recall={recall:.2f}  mean_time={time_ms:.1f}ms")
