"""CLI for running code-only agent benchmarks."""

import argparse
from pathlib import Path

from agent.experiments.framework import ExperimentRunner, ExperimentStorage, ExperimentAnalyzer
from agent.experiments.code_only_agent.benchmarks.base import (
    CodeAgentBenchmarkTestCase,
    CodeAgentMetricsCalculator,
)
from agent.experiments.code_only_agent.benchmarks.test_cases import get_all_benchmarks
from agent.experiments.code_only_agent.variant import LLMCodeAgentVariant
from agent.llm.models import SupportedModel
from agent.llm.router import create_llm
from agent.ui_output import ui_print


def main():
    """Run benchmark suite using experiment framework."""
    parser = argparse.ArgumentParser(
        description="Run code-only agent benchmark suite"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Filter by category (communication, filesystem, time_system, multi_step, edge_cases, code_logic, integration)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "very_hard"],
        help="Filter by difficulty level",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MISTRAL_SMALL_3_2_Q4",
        help="Model to use for benchmarks",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per test (default: 3)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-test output",
    )

    args = parser.parse_args()

    # Get model
    try:
        model = SupportedModel[args.model]
    except KeyError:
        ui_print(f"Unknown model: {args.model}")
        ui_print(f"Available models: {[m.name for m in SupportedModel]}")
        return 1

    # Create LLM
    ui_print(f"Initializing LLM with model: {model}")
    llm = create_llm()

    # Get benchmarks and filter
    all_benchmarks = get_all_benchmarks()
    filtered_benchmarks = all_benchmarks

    if args.category:
        filtered_benchmarks = [
            b for b in filtered_benchmarks if b.category == args.category
        ]
    if args.difficulty:
        filtered_benchmarks = [
            b for b in filtered_benchmarks if b.difficulty == args.difficulty
        ]

    ui_print(f"Running {len(filtered_benchmarks)} benchmarks with {args.runs} runs each")

    # Convert to framework test cases
    test_cases = [CodeAgentBenchmarkTestCase(b) for b in filtered_benchmarks]

    # Create variant
    variant = LLMCodeAgentVariant(llm, model)

    # Setup storage
    storage_dir = Path(__file__).parent / "benchmark_results"
    storage = ExperimentStorage(storage_dir)

    # Run experiment
    ui_print(f"\n{'='*60}")
    ui_print("Running experiments...")
    ui_print(f"{'='*60}\n")

    runner = ExperimentRunner[LLMCodeAgentVariant](
        storage,
        max_retries=0,
        progress_callback=ui_print if not args.quiet else None,
    )
    run_timestamp = runner.run_experiment(
        variants=[variant],
        test_cases=test_cases,
        num_runs=args.runs,
    )

    # Analyze results
    ui_print("\nAnalyzing results...")
    analyzer = ExperimentAnalyzer(storage)
    calculator = CodeAgentMetricsCalculator()
    results = analyzer.calculate_metrics(run_timestamp, calculator)

    # Generate report
    report = analyzer.generate_comparison_report(results, calculator)
    ui_print("\n" + report)

    ui_print(f"\nResults saved to: {storage_dir / run_timestamp}")

    return 0


if __name__ == "__main__":
    exit(main())
