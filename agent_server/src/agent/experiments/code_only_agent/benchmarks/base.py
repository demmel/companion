"""Base classes for code-only agent benchmarks."""

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Any

from pydantic import BaseModel, Field

from agent.experiments.framework import TestCase, MetricsCalculator
from agent.experiments.code_only_agent.agent import run_agent
from agent.experiments.code_only_agent.state import State, AgentTurn
from agent.experiments.code_only_agent.variant import LLMCodeAgentVariant
from agent.llm.models import SupportedModel
from agent.llm.router import LLM
from agent.ui_output import ui_print


class NoExpectedOutput(BaseModel):
    """Placeholder for test cases without expected output (validation happens during execution)."""

    pass


class BenchmarkResult(BaseModel):
    """Result of running a single benchmark."""

    task_name: str
    category: str
    difficulty: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    iterations_used: int
    max_iterations: int
    errors_encountered: list[str] = Field(default_factory=list)
    functions_called: dict[str, int] = Field(default_factory=dict)
    execution_time_seconds: float
    details: str
    user_input: str
    agent_response: str


@dataclass
class CodeAgentBenchmark:
    """Definition of a single benchmark task."""

    name: str
    category: str
    difficulty: str  # "easy", "medium", "hard", "very_hard"
    description: str
    user_input: str
    setup_fn: Optional[Callable[[Path], None]]  # Creates files/dirs for test
    validation_fn: Callable[[AgentTurn, State, Path], BenchmarkResult]
    setup_conversation: Optional[list[str]] = (
        None  # User inputs for conversation context before test
    )

    def __post_init__(self):
        """Validate difficulty level."""
        valid_difficulties = {"easy", "medium", "hard", "very_hard"}
        if self.difficulty not in valid_difficulties:
            raise ValueError(
                f"Invalid difficulty '{self.difficulty}'. Must be one of {valid_difficulties}"
            )


class CodeAgentBenchmarkTestCase(TestCase[LLMCodeAgentVariant]):
    """Adapts CodeAgentBenchmark to experiment framework TestCase."""

    def __init__(self, benchmark: CodeAgentBenchmark):
        self.benchmark = benchmark

    def name(self) -> str:
        """Return the test case name."""
        return self.benchmark.name

    def execute(self, variant: LLMCodeAgentVariant) -> BenchmarkResult:
        """Execute the benchmark with the given variant."""
        # Create temporary test directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)

            # Run setup if provided
            if self.benchmark.setup_fn:
                self.benchmark.setup_fn(test_path)

            # Create fresh state
            state = State()

            # Run setup conversation if provided
            if self.benchmark.setup_conversation:
                for setup_input in self.benchmark.setup_conversation:
                    run_agent(setup_input, state, variant.llm, variant.model)

            # Run agent
            start_time = datetime.now()
            turn = run_agent(
                self.benchmark.user_input, state, variant.llm, variant.model
            )
            execution_time = (datetime.now() - start_time).total_seconds()

            # Run validation
            result = self.benchmark.validation_fn(turn, state, test_path)

            # Add execution time
            result.execution_time_seconds = execution_time

            return result

    def expected_output(self) -> Optional[BaseModel]:
        """Return expected output (not used since validation happens in execute)."""
        return NoExpectedOutput()


class CodeAgentMetricsCalculator(MetricsCalculator[BenchmarkResult, NoExpectedOutput]):
    """Extracts metrics from BenchmarkResult for experiment framework."""

    def calculate(
        self, output: BenchmarkResult, expected: Optional[NoExpectedOutput]
    ) -> dict[str, float]:
        """Extract metrics from benchmark result."""
        return {
            "passed": 1.0 if output.passed else 0.0,
            "score": output.score,
            "iterations_used": float(output.iterations_used),
            "function_calls": float(sum(output.functions_called.values())),
            "error_count": float(len(output.errors_encountered)),
        }


class BenchmarkRunner:
    """Runs benchmark suite and collects results."""

    def __init__(self, llm: LLM, model: SupportedModel):
        self.llm = llm
        self.model = model

    def run_single(
        self, benchmark: CodeAgentBenchmark, verbose: bool = False
    ) -> BenchmarkResult:
        """Run a single benchmark task."""
        # Create temporary test directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)

            # Run setup if provided
            if benchmark.setup_fn:
                benchmark.setup_fn(test_path)

            # Create fresh state
            state = State()

            # Run setup conversation if provided
            if benchmark.setup_conversation:
                for setup_input in benchmark.setup_conversation:
                    run_agent(setup_input, state, self.llm, self.model)

            # Run agent
            start_time = datetime.now()
            turn = run_agent(benchmark.user_input, state, self.llm, self.model)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Run validation
            result = benchmark.validation_fn(turn, state, test_path)

            # Add execution time
            result.execution_time_seconds = execution_time

            if verbose:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                ui_print(
                    f"  {status} {benchmark.name} - Score: {result.score:.2f} ({result.iterations_used} iterations)"
                )

            return result

    def run_suite(
        self,
        benchmarks: list[CodeAgentBenchmark],
        verbose: bool = True,
        category_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run full benchmark suite and return aggregated results."""
        # Filter benchmarks
        filtered = benchmarks
        if category_filter:
            filtered = [b for b in filtered if b.category == category_filter]
        if difficulty_filter:
            filtered = [b for b in filtered if b.difficulty == difficulty_filter]

        if verbose:
            ui_print(f"\n{'='*60}")
            ui_print(f"Running {len(filtered)} benchmarks")
            if category_filter:
                ui_print(f"Category filter: {category_filter}")
            if difficulty_filter:
                ui_print(f"Difficulty filter: {difficulty_filter}")
            ui_print(f"{'='*60}\n")

        results = []
        for i, benchmark in enumerate(filtered, 1):
            if verbose:
                ui_print(f"[{i}/{len(filtered)}] {benchmark.category}/{benchmark.name}")
            result = self.run_single(benchmark, verbose=verbose)
            results.append(result)

        # Aggregate results
        summary = self._aggregate_results(results, filtered)

        return {
            "summary": summary,
            "results": results,
        }

    def _aggregate_results(
        self, results: list[BenchmarkResult], benchmarks: list[CodeAgentBenchmark]
    ) -> dict[str, Any]:
        """Aggregate results into summary statistics."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        total_score = sum(r.score for r in results)
        avg_score = total_score / total if total > 0 else 0.0

        # By category
        by_category = {}
        for benchmark in benchmarks:
            if benchmark.category not in by_category:
                by_category[benchmark.category] = {
                    "total": 0,
                    "passed": 0,
                    "total_score": 0.0,
                }

        for result in results:
            cat = by_category[result.category]
            cat["total"] += 1
            if result.passed:
                cat["passed"] += 1
            cat["total_score"] += result.score

        for cat_data in by_category.values():
            cat_data["score"] = (
                cat_data["total_score"] / cat_data["total"]
                if cat_data["total"] > 0
                else 0.0
            )
            del cat_data["total_score"]

        # By difficulty
        by_difficulty = {}
        for benchmark in benchmarks:
            if benchmark.difficulty not in by_difficulty:
                by_difficulty[benchmark.difficulty] = {
                    "total": 0,
                    "passed": 0,
                    "total_score": 0.0,
                }

        for result in results:
            diff = by_difficulty[result.difficulty]
            diff["total"] += 1
            if result.passed:
                diff["passed"] += 1
            diff["total_score"] += result.score

        for diff_data in by_difficulty.values():
            diff_data["score"] = (
                diff_data["total_score"] / diff_data["total"]
                if diff_data["total"] > 0
                else 0.0
            )
            del diff_data["total_score"]

        # Execution stats
        avg_iterations = (
            sum(r.iterations_used for r in results) / total if total > 0 else 0.0
        )
        total_execution_time = sum(r.execution_time_seconds for r in results)

        return {
            "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "model": str(self.model),
            "total_tasks": total,
            "passed": passed,
            "failed": total - passed,
            "overall_score": avg_score,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "avg_iterations_per_task": avg_iterations,
            "total_execution_time_seconds": total_execution_time,
        }

    def save_results(self, results_data: dict[str, Any], output_dir: Path) -> None:
        """Save results to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = results_data["summary"]
        run_id = summary["run_id"]

        # Create run directory
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed results
        results_list = []
        for result in results_data["results"]:
            result_dict = result.model_dump()
            results_list.append(result_dict)

        detailed_path = run_dir / "detailed_results.json"
        with open(detailed_path, "w") as f:
            json.dump(results_list, f, indent=2)

        ui_print(f"\nResults saved to: {run_dir}")
        ui_print(f"  - {summary_path}")
        ui_print(f"  - {detailed_path}")

    def print_summary(self, summary: dict[str, Any]) -> None:
        """Print formatted summary to console."""
        ui_print(f"\n{'='*60}")
        ui_print("BENCHMARK SUMMARY")
        ui_print(f"{'='*60}")
        ui_print(f"Run ID: {summary['run_id']}")
        ui_print(f"Model: {summary['model']}")
        ui_print(f"Timestamp: {summary['timestamp']}")
        ui_print("")
        ui_print(
            f"Overall: {summary['passed']}/{summary['total_tasks']} passed ({summary['overall_score']:.1%})"
        )
        ui_print(f"Average iterations: {summary['avg_iterations_per_task']:.1f}")
        ui_print(
            f"Total execution time: {summary['total_execution_time_seconds']:.1f}s"
        )

        ui_print(f"\n{'Category Performance:':-^60}")
        for cat_name, cat_data in sorted(summary["by_category"].items()):
            pct = cat_data["score"]
            passed = cat_data["passed"]
            total = cat_data["total"]
            bar = "█" * int(pct * 20)
            ui_print(f"  {cat_name:20} {passed:2}/{total:2} [{bar:<20}] {pct:.1%}")

        ui_print(f"\n{'Difficulty Performance:':-^60}")
        difficulty_order = ["easy", "medium", "hard", "very_hard"]
        for diff in difficulty_order:
            if diff in summary["by_difficulty"]:
                diff_data = summary["by_difficulty"][diff]
                pct = diff_data["score"]
                passed = diff_data["passed"]
                total = diff_data["total"]
                bar = "█" * int(pct * 20)
                ui_print(f"  {diff:12} {passed:2}/{total:2} [{bar:<20}] {pct:.1%}")

        ui_print(f"{'='*60}\n")
