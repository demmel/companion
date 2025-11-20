"""
Run structured output format experiment using the reusable framework.

Usage:
    python -m agent.experiments.structured_formats.run_experiment experiment [--num-runs N]
    python -m agent.experiments.structured_formats.run_experiment analyze <run_ts> [--strict-eval | --semantic-eval]
"""

import click
import logging
from pathlib import Path

from agent.llm import create_llm, SupportedModel
from agent.experiments.framework import (
    ExperimentStorage,
    ExperimentRunner,
    ExperimentAnalyzer,
)

from .formats import JSONFormat, XMLFormat, YAMLFormat, SExpFormat
from .test_cases import ALL_TEST_CASES, convert_to_framework_test_cases
from .framework_metrics import StructuredFormatMetricsCalculator
from .base_format import StructuredOutputFormat


def setup_logging(verbose: bool):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO

    # Simple format without module names
    log_format = (
        "%(message)s"
        if not verbose
        else "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="%H:%M:%S",
        force=True,  # Override any existing config
    )


setup_logging(verbose=False)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Structured output format experiment."""
    pass


@cli.command()
@click.option(
    "--num-runs",
    type=int,
    default=15,
    help="Number of runs per test case (default: 15)",
)
@click.option(
    "--output-dir",
    type=str,
    default="experiment_results",
    help="Directory to save results (default: experiment_results)",
)
@click.option(
    "--model",
    type=str,
    default="MISTRAL_SMALL_3_2_Q4",
    help="Model to use (default: MISTRAL_SMALL_3_2_Q4)",
)
def experiment(num_runs: int, output_dir: str, model: str):
    """Run experiment to collect raw data."""
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting structured output format experiment")
    logger.info(f"Runs per test case: {num_runs}")
    logger.info(f"Model: {model}")
    logger.info(f"Output directory: {output_dir_path}")

    # Initialize LLM
    llm = create_llm()
    model_enum = SupportedModel[model]

    # Initialize formats to test (these are the variants)
    formats = [
        JSONFormat(),
        XMLFormat(),
        YAMLFormat(),
        SExpFormat(),
    ]

    logger.info(f"Testing {len(formats)} formats: {[f.name() for f in formats]}")
    logger.info(f"Test cases: {len(ALL_TEST_CASES)}")

    # Convert test cases to framework test cases
    framework_test_cases = convert_to_framework_test_cases(
        ALL_TEST_CASES,
        llm=llm,
        model=model_enum,
        max_retries=3,
    )

    # Initialize framework components
    storage = ExperimentStorage(base_dir=output_dir_path)
    runner = ExperimentRunner[StructuredOutputFormat](storage=storage, max_retries=3)

    # Run experiment
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING EXPERIMENT")
    logger.info("=" * 80 + "\n")

    run_ts = runner.run_experiment(
        variants=formats, test_cases=framework_test_cases, num_runs=num_runs
    )

    logger.info(f"\nResults saved to: {output_dir_path / run_ts}")
    logger.info(
        f"To analyze: python -m agent.experiments.structured_formats.run_experiment analyze {run_ts}"
    )
    logger.info("\nExperiment complete!")


@cli.command()
@click.argument("run_ts")
@click.option(
    "--output-dir",
    type=str,
    default="experiment_results",
    help="Directory where results are saved (default: experiment_results)",
)
@click.option(
    "--strict-eval",
    is_flag=True,
    help="Use strict exact-match evaluation instead of flexible evaluation",
)
@click.option(
    "--semantic-eval",
    is_flag=True,
    help="Use semantic similarity evaluation with embeddings (most accurate for open-ended tasks)",
)
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed per-test-case metrics in addition to comparative analysis",
)
@click.option(
    "--model",
    type=str,
    default="MISTRAL_SMALL_3_2_Q4",
    help="Model to use for LLM-as-judge (default: MISTRAL_SMALL_3_2_Q4)",
)
def analyze(
    run_ts: str,
    output_dir: str,
    strict_eval: bool,
    semantic_eval: bool,
    detailed: bool,
    model: str,
):
    """Analyze saved experiment results and generate report."""
    output_dir_path = Path(output_dir)

    # Determine evaluation mode
    use_semantic_eval = semantic_eval

    if semantic_eval:
        eval_mode = "semantic similarity (embeddings)"
    elif strict_eval:
        eval_mode = "strict exact-match"
    else:
        eval_mode = "flexible (hardcoded synonyms)"

    logger.info("Analyzing experiment results")
    logger.info(f"Run: {run_ts}")
    logger.info(f"Evaluation mode: {eval_mode}")
    logger.info(f"LLM-as-judge model: {model}")

    # Initialize LLM for comparative metrics
    llm = create_llm()
    model_enum = SupportedModel[model]

    # Initialize framework components
    storage = ExperimentStorage(base_dir=output_dir_path)
    analyzer = ExperimentAnalyzer(storage=storage)
    calculator = StructuredFormatMetricsCalculator(
        llm=llm,
        model=model_enum,
        use_semantic_eval=use_semantic_eval,
    )

    logger.info("\n" + "=" * 80)
    logger.info("CALCULATING METRICS")
    logger.info("=" * 80 + "\n")

    results = analyzer.calculate_metrics(run_ts, calculator)

    # Generate and print comparison report
    report = analyzer.generate_comparison_report(results, calculator)
    print("\n" + report)

    # Optionally show detailed per-test-case breakdown
    if detailed:
        detailed_report = analyzer.generate_report(results)
        print("\n" + detailed_report)

    # Save report to file
    eval_suffix = (
        "_semantic" if semantic_eval else "_strict" if strict_eval else "_flexible"
    )
    report_file = output_dir_path / run_ts / f"comparison_report{eval_suffix}.txt"
    with open(report_file, "w") as f:
        f.write(report)

    logger.info(f"\nReport saved to: {report_file}")
    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    cli()
