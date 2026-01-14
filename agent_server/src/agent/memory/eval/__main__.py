"""Memory evaluation CLI."""

from pathlib import Path

import click

from agent.chain_of_action.trigger_history import TriggerHistory
from agent.llm import create_llm, SupportedModel
from agent.memory.sliding_window import SlidingWindowMemory
from agent.memory.dag.dag_memory_manager import DagMemoryManager

from .scenario_extractor import load_all_scenarios, load_scenario, save_scenario
from .harness import run_evaluation, print_eval_summary, MemoryFactory
from .synthetic_scenarios import create_all_synthetic_scenarios
from .data_models import EvalRun


def create_memory_factory(name: str, use_individual_formatting: bool) -> MemoryFactory:
    """Create a memory factory by name."""
    if name == "dag":
        return lambda th: DagMemoryManager.create(th, use_individual_formatting)
    elif name.startswith("sliding:"):
        size = int(name.split(":")[1])
        return lambda th: SlidingWindowMemory(window_size=size)
    else:
        raise ValueError(f"Unknown memory implementation: {name}")


def save_results(eval_run: EvalRun, results_dir: Path) -> Path:
    """Save evaluation results to JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = f"run_{eval_run.run_id}.json"
    filepath = results_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(eval_run.model_dump_json(indent=2))

    return filepath


@click.group()
def cli() -> None:
    """Memory evaluation tools."""
    pass


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("eval_data/scenarios"),
    help="Directory to save scenario files",
)
def generate(output_dir: Path) -> None:
    """Generate synthetic scenarios and save them to disk."""
    scenarios = create_all_synthetic_scenarios()

    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        filepath = output_dir / f"{scenario.scenario_id}.json"
        save_scenario(scenario, filepath)
        click.echo(f"Saved: {filepath}")

    click.echo(f"\nGenerated {len(scenarios)} scenarios in {output_dir}")


@cli.command()
@click.option(
    "--scenarios-dir",
    type=click.Path(path_type=Path),
    default=Path("eval_data/scenarios"),
    help="Directory containing scenario JSON files",
)
@click.option(
    "--scenario",
    type=click.Path(exists=True, path_type=Path),
    help="Single scenario file to run (overrides --scenarios-dir)",
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=Path("eval_data/results"),
    help="Directory to save results",
)
@click.option(
    "--memory",
    multiple=True,
    default=["dag", "sliding:20"],
    help="Memory implementations to test (dag, sliding:N). Can be specified multiple times.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Force re-evaluation, ignore cached results",
)
@click.option(
    "--query-model",
    type=click.Choice(["llama", "mistral", "claude"]),
    default="llama",
    help="Model for query extraction (llama=fast, mistral=balanced, claude=production)",
)
@click.option(
    "--individual-formatting/--compressed-formatting",
    default=True,
    help="Use individual memory formatting (default) or compressed container summaries",
)
def run(
    scenarios_dir: Path,
    scenario: Path | None,
    results_dir: Path,
    memory: tuple[str, ...],
    no_cache: bool,
    query_model: str,
    individual_formatting: bool,
) -> None:
    """Run memory evaluation on scenarios."""
    # Map query model choice to SupportedModel
    query_model_map = {
        "llama": SupportedModel.LLAMA_3B,
        "mistral": SupportedModel.MISTRAL_SMALL_3_2_Q4,
        "claude": SupportedModel.CLAUDE_SONNET_4_5,
    }
    query_extraction_model = query_model_map[query_model]

    if scenario:
        scenarios = [load_scenario(scenario)]
        click.echo(f"Loaded 1 scenario from {scenario}")
    else:
        scenarios = load_all_scenarios(scenarios_dir)
        click.echo(f"Loaded {len(scenarios)} scenarios from {scenarios_dir}")

    if not scenarios:
        click.echo("No scenarios found. Run 'generate' first or create scenarios via the UI.")
        return

    factories: dict[str, MemoryFactory] = {}
    for name in memory:
        factories[name] = create_memory_factory(name, individual_formatting)
    click.echo(f"Testing {len(factories)} memory implementations: {list(factories.keys())}")

    llm = create_llm()
    judge_model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    click.echo(f"\nRunning evaluation (query model: {query_model})...")
    eval_run = run_evaluation(
        scenarios=scenarios,
        memory_factories=factories,
        llm=llm,
        judge_model=judge_model,
        query_extraction_model=query_extraction_model,
        use_cache=not no_cache,
    )

    print_eval_summary(eval_run)

    results_path = save_results(eval_run, results_dir)
    click.echo(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    cli()
