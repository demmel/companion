#!/usr/bin/env python3
"""
Sequential Prompt Optimization CLI

Command-line interface for the sequential multi-prompt optimization framework.
Provides access to Level 1 (AgentEvaluator) and Sequential Optimization.
"""

import time
import click
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from agent.llm import SupportedModel, create_llm
from agent.progress import RichProgressReporter, NullProgressReporter

console = Console()

# Build model choices from enum
MODEL_CHOICES = [model.name.lower().replace("_", "-") for model in SupportedModel]
MODEL_MAP = {model.name.lower().replace("_", "-"): model for model in SupportedModel}


def setup_logging(
    path_manager, log_level: str, log_to_file: bool = True, rich_console=None
):
    """Configure logging for optimization runs using existing OptimizationPathManager

    Args:
        path_manager: OptimizationPathManager for file paths
        log_level: Logging level (DEBUG, INFO, etc.)
        log_to_file: Whether to log to file
        rich_console: Optional Rich console for coordinated output (prevents stderr artifacts)
    """

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_file:
        # File handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = path_manager.paths.logs_dir / f"optimization_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Capture all debug info to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        console.print(f"📝 Logging to: {log_file}")

    # Console handler - use Rich if available to avoid display artifacts
    if rich_console:
        from rich.logging import RichHandler

        console_handler = RichHandler(
            console=rich_console, show_time=True, show_path=False, rich_tracebacks=True
        )
        console_handler.setLevel(logging.ERROR)  # Only show errors on console
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        # Fallback to stderr if no Rich console provided
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)  # Only show errors on console
        console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)


@click.group()
def cli():
    """🚀 Sequential Prompt Optimization Framework"""
    console.print(
        Panel.fit(
            Text("🚀 Sequential Prompt Optimization Framework", style="bold blue"),
            border_style="blue",
        )
    )


@cli.command()
@click.option(
    "--domain", default="roleplay", help="Domain to evaluate (default: roleplay)"
)
@click.option(
    "--model",
    type=click.Choice(MODEL_CHOICES),
    default=MODEL_CHOICES[0],
    help="Model to use for agent",
)
@click.option("--scenario", help="Specific scenario to test (optional)")
def evaluate(domain: str, model: str, scenario: str):
    """🧪 Level 1: Run agent evaluation for a domain"""
    console.print(f"\n[bold cyan]Level 1: Agent Evaluation[/bold cyan]")
    console.print(f"Domain: {domain}")
    console.print(f"Model: {model}")

    try:
        # Import evaluation components
        from agent.eval.agent_evaluator import AgentEvaluator

        if domain == "roleplay":
            from agent.eval.domains.roleplay import RoleplayEvaluationConfig

            domain_config = RoleplayEvaluationConfig()
        else:
            console.print(f"[red]Unknown domain: {domain}[/red]")
            console.print("Available domains: roleplay")
            return

        llm = create_llm()
        supported_model = MODEL_MAP[model]

        # Get scenarios
        eval_config = domain_config.get_evaluation_config()
        scenarios = (
            [scenario] if scenario else eval_config.test_scenarios[:3]
        )  # Test first 3

        console.print(f"Testing {len(scenarios)} scenarios...")

        # Set up progress reporting
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as rich_progress:

            # Create progress reporter for evaluator
            progress_reporter = RichProgressReporter(rich_progress)
            evaluator = AgentEvaluator(
                domain_eval_config=domain_config,
                model=supported_model,
                llm=llm,
                progress=progress_reporter,
            )

            with progress_reporter.task(
                f"Evaluating {len(scenarios)} scenarios", total=len(scenarios)
            ) as eval_task:

                for i, test_scenario in enumerate(scenarios, 1):
                    eval_task.update(
                        (i - 1) / len(scenarios),
                        f"Scenario {i}/{len(scenarios)}: {test_scenario[:50]}...",
                    )

                    try:
                        result = evaluator.run_evaluation(test_scenario)

                        console.print(
                            f"\n[bold green]✅ Scenario {i} Results:[/bold green]"
                        )
                        console.print(
                            f"Overall Score: [bold]{result.overall_score:.1f}/10[/bold]"
                        )
                        console.print(f"Feedback: {result.feedback}")

                        if result.scores:
                            table = Table(title="Detailed Scores")
                            table.add_column("Criterion", style="cyan")
                            table.add_column("Score", style="green", justify="right")

                            for criterion, score in result.scores.items():
                                table.add_row(criterion, f"{score:.1f}/10")
                            console.print(table)

                        if result.suggested_improvements:
                            console.print(
                                "[bold yellow]Suggested Improvements:[/bold yellow]"
                            )
                            for improvement in result.suggested_improvements:
                                console.print(f"  • {improvement}")

                    except Exception as e:
                        console.print(f"[red]❌ Scenario {i} failed: {e}[/red]")
                        logging.getLogger(__name__).error(
                            f"Scenario {i} failed", exc_info=True
                        )

        console.print(f"\n[bold green]✅ Evaluation complete![/bold green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logging.getLogger(__name__).error("Evaluation failed", exc_info=True)


@cli.command()
@click.option(
    "--domain", default="roleplay", help="Domain to optimize (default: roleplay)"
)
@click.option("--scenario", help="Specific scenario to optimize for (required)")
@click.option("--max-cycles", default=3, help="Maximum optimization cycles")
@click.option(
    "--optimization-dir",
    default="optimization_data",
    help="Base directory for optimization files",
)
@click.option(
    "--model",
    type=click.Choice(MODEL_CHOICES),
    default=MODEL_CHOICES[0],
    help="Model to use for optimization",
)
def sequential(
    domain: str, scenario: str, max_cycles: int, optimization_dir: str, model: str
):
    """🔄 Sequential Multi-Prompt Optimization"""
    console.print(f"\n[bold cyan]Sequential Multi-Prompt Optimization[/bold cyan]")
    console.print(f"Domain: {domain}")
    console.print(f"Scenario: {scenario}")
    console.print(f"Max cycles: {max_cycles}")

    if not scenario:
        console.print(
            f"[red]Error: --scenario is required for sequential optimization[/red]"
        )
        console.print("Example: --scenario 'Roleplay as Elena, a mysterious vampire'")
        return

    try:
        # Import optimization components
        from agent.eval.sequential_optimizer import SequentialOptimizer
        from agent.eval.optimization_paths import OptimizationPathManager
        from agent.eval.preferences import SemanticPreferenceManager

        if domain == "roleplay":
            from agent.eval.domains.roleplay import RoleplayEvaluationConfig

            domain_config = RoleplayEvaluationConfig()
        else:
            console.print(f"[red]Unknown domain: {domain}[/red]")
            console.print("Available domains: roleplay")
            return

        # Create path manager and preference manager
        path_manager = OptimizationPathManager(base_dir=optimization_dir, domain=domain)

        llm = create_llm()
        supported_model = MODEL_MAP[model]

        prefs = SemanticPreferenceManager(
            llm=llm,
            model=supported_model,
            progress_reporter=NullProgressReporter(),
            preferences_dir=str(path_manager.paths.preferences_dir),
        )

        console.print(f"\n[bold blue]Starting sequential optimization...[/bold blue]")
        console.print(
            f"This will optimize agent, simulation, and evaluation prompts together."
        )
        console.print(f"The system will ask for user feedback when needed.")

        # Run sequential optimization with progress reporting
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as rich_progress:

            # Create progress reporter for optimization
            progress_reporter = RichProgressReporter(rich_progress)

            # Update logging to coordinate with Rich progress display
            setup_logging(
                path_manager,
                log_level="DEBUG",
                log_to_file=True,
                rich_console=rich_progress.console,
            )

            with progress_reporter.task(
                "Sequential optimization", total=max_cycles
            ) as opt_task:

                try:
                    # Create optimizer with progress reporter and path manager
                    optimizer = SequentialOptimizer(
                        domain_config,
                        prefs,
                        llm,
                        supported_model,
                        progress_reporter,
                        path_manager,
                    )

                    result = optimizer.run_sequential_optimization(
                        scenario=scenario, max_cycles=max_cycles
                    )

                    # Display results
                    console.print(
                        f"\n[bold green]🎯 Sequential Optimization Results:[/bold green]"
                    )

                    table = Table(title="Optimization Summary")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")

                    table.add_row("Run ID", result.run_id)
                    table.add_row("Domain", result.domain)
                    table.add_row("Success", "✅ Yes" if result.success else "❌ No")
                    table.add_row("Conversation Pairs", str(result.conversation_pairs))
                    table.add_row("Feedback Sessions", str(result.feedback_sessions))
                    table.add_row("Duration", f"{result.duration_seconds:.1f}s")

                    console.print(table)

                    # Show evaluator optimizations
                    if any(result.evaluator_optimizations.values()):
                        console.print(
                            f"\n[bold blue]📊 Evaluator Optimizations:[/bold blue]"
                        )
                        for (
                            eval_type,
                            was_optimized,
                        ) in result.evaluator_optimizations.items():
                            status = (
                                "✅ Optimized" if was_optimized else "❌ No changes"
                            )
                            console.print(f"  {eval_type}: {status}")

                    # Show prompt changes
                    console.print(f"\n[bold blue]📝 Prompt Changes:[/bold blue]")
                    for prompt_type in [
                        "agent",
                        "simulation",
                        "agent_eval",
                        "sim_eval",
                    ]:
                        initial = result.initial_prompts.get(prompt_type, "")
                        final = result.final_prompts.get(prompt_type, "")

                        if initial != final:
                            console.print(f"  ✅ {prompt_type}: Modified")
                        else:
                            console.print(f"  ❌ {prompt_type}: No changes")

                    if result.success:
                        console.print(
                            f"\n[bold green]🎉 Sequential optimization completed successfully![/bold green]"
                        )
                        console.print(
                            f"Optimized prompts have been saved to: {path_manager.paths.optimized_prompts_dir}"
                        )
                    else:
                        console.print(
                            f"\n[bold yellow]⚠️ Optimization completed but no significant improvements found.[/bold yellow]"
                        )

                except Exception as e:
                    console.print(f"[red]❌ Sequential optimization failed: {e}[/red]")
                    logging.getLogger(__name__).error(
                        "Sequential optimization failed", exc_info=True
                    )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logging.getLogger(__name__).error(
            "Sequential optimization error", exc_info=True
        )


@cli.command()
@click.option(
    "--preferences-dir", default="preferences", help="Directory for preference files"
)
@click.option(
    "--model",
    type=click.Choice(MODEL_CHOICES),
    default=MODEL_CHOICES[0],
    help="Model to use for preference analysis",
)
@click.option(
    "--show-contradictions", is_flag=True, help="Show detailed contradiction analysis"
)
@click.option("--show-philosophy", is_flag=True, help="Show synthesized philosophy")
def preferences(
    preferences_dir: str, model: str, show_contradictions: bool, show_philosophy: bool
):
    """📊 View learned user preferences (Advanced V3 System)"""
    console.print(f"\n[bold cyan]📊 Advanced User Preferences V3[/bold cyan]")

    try:
        from agent.eval.preferences import SemanticPreferenceManager
        from agent.llm import create_llm, SupportedModel

        llm = create_llm()
        supported_model = MODEL_MAP[model]
        prefs = SemanticPreferenceManager(
            llm=llm,
            model=supported_model,
            progress_reporter=NullProgressReporter(),
            preferences_dir=preferences_dir,
        )
        summary = prefs.get_summary()

        # Schema version and system status
        console.print(f"[bold blue]System Status:[/bold blue]")
        console.print(f"  Schema Version: {summary['schema_version']}")
        console.print(f"  Total Sessions: {summary['total_feedback_sessions']}")
        console.print(
            f"  Total Preferences Created: {summary['total_preferences_created']}"
        )

        # Maintenance status
        maintenance = summary["maintenance"]
        console.print(
            f"  Auto Contradiction Check: {'✅' if maintenance['auto_contradiction_check'] else '❌'}"
        )
        console.print(
            f"  Auto Philosophy Synthesis: {'✅' if maintenance['auto_philosophy_synthesis'] else '❌'}"
        )

        # Preference status by domain
        table = Table(title="Preference Status by Domain")
        table.add_column("Domain", style="cyan")
        table.add_column("Active", style="green")
        table.add_column("Deprecated", style="yellow")
        table.add_column("Removed", style="red")
        table.add_column("Confidence", style="blue")

        for domain in ["conversation", "simulation", "evaluation", "general"]:
            status_counts = summary["status_counts"].get(domain, {})
            active = status_counts.get("active", 0)
            deprecated = status_counts.get("deprecated", 0)
            removed = status_counts.get("removed", 0)
            confidence = summary["confidence_levels"].get(domain, 0.0)

            table.add_row(
                domain.title(),
                str(active),
                str(deprecated),
                str(removed),
                f"{confidence:.2f}",
            )

        console.print(table)

        # Philosophy summary
        philosophy = summary["philosophy"]
        if philosophy["status"] == "available":
            console.print(f"\n[bold blue]🧸 Philosophy Synthesis:[/bold blue]")
            console.print(f"  Status: {philosophy['status']}")
            console.print(f"  Confidence: {philosophy['confidence']:.2f}")
            console.print(f"  Top Values: {', '.join(philosophy['top_values'][:3])}")

            if show_philosophy:
                console.print(f"\n[bold green]Core Philosophy:[/bold green]")
                console.print(f"  {philosophy['core_philosophy']}")

                console.print(f"\n[bold green]Key Principles:[/bold green]")
                for principle in philosophy["key_principles"]:
                    console.print(f"  • {principle}")
        else:
            console.print(
                f"\n[bold yellow]🧸 Philosophy:[/bold yellow] {philosophy['recommendation']}"
            )

        # Contradictions summary
        contradictions = summary["contradictions"]
        if contradictions["active_count"] > 0:
            console.print(
                f"\n[bold red]⚠️ Active Contradictions: {contradictions['active_count']}[/bold red]"
            )

            if show_contradictions:
                for contradiction in contradictions["active_contradictions"]:
                    console.print(
                        f"  • {contradiction['type']}: {contradiction['description'][:80]}..."
                    )
                    console.print(
                        f"    Affects {contradiction['affected_preferences']} preferences (confidence: {contradiction['confidence']:.2f})"
                    )
        else:
            console.print(f"\n[bold green]✅ No Active Contradictions[/bold green]")

        if contradictions["resolved_count"] > 0:
            console.print(
                f"  Resolved: {contradictions['resolved_count']} contradictions"
            )

        # Show some recent active preferences
        for domain in ["conversation", "simulation", "evaluation", "general"]:
            domain_prefs = prefs.get_preferences(domain)
            if domain_prefs:
                console.print(
                    f"\n[bold green]{domain.title()} Preferences:[/bold green]"
                )
                recent_prefs = sorted(
                    domain_prefs, key=lambda p: p.last_updated, reverse=True
                )[:3]

                for pref in recent_prefs:
                    icon = "👍" if pref.preference_type == "positive" else "👎"
                    status_icon = "🟢" if pref.status.value == "active" else "🟡"
                    console.print(f"  {icon}{status_icon} {pref.description[:70]}...")
                    console.print(
                        f"     Confidence: {pref.confidence:.2f} | Reinforced: {pref.reinforcement_count}x"
                    )
                    if pref.tags:
                        console.print(f"     Tags: {', '.join(pref.tags[:3])}")

    except Exception as e:
        console.print(f"[red]Error loading preferences: {e}[/red]")


@cli.command()
@click.option(
    "--preferences-dir", default="preferences", help="Directory for preference files"
)
@click.option(
    "--model",
    type=click.Choice(MODEL_CHOICES),
    default=MODEL_CHOICES[0],
    help="Model to use for philosophy synthesis",
)
def synthesize_philosophy(preferences_dir: str, model: str):
    """🧸 Synthesize holistic philosophy from preferences"""
    console.print(f"\n[bold cyan]🧸 Philosophy Synthesis[/bold cyan]")

    try:
        from agent.eval.preferences import SemanticPreferenceManager
        from agent.llm import create_llm, SupportedModel

        llm = create_llm()
        supported_model = MODEL_MAP[model]
        prefs = SemanticPreferenceManager(
            llm=llm,
            model=supported_model,
            progress_reporter=NullProgressReporter(),
            preferences_dir=preferences_dir,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Synthesizing philosophy from preferences...", total=None
            )

            prefs._synthesize_philosophy()
            progress.remove_task(task)

        philosophy = prefs.get_philosophy_summary()
        if philosophy["status"] == "available":
            console.print(
                f"\n[bold green]✅ Philosophy Synthesis Complete![/bold green]"
            )
            console.print(f"[bold blue]Core Philosophy:[/bold blue]")
            console.print(f"  {philosophy['core_philosophy']}")

            console.print(f"\n[bold blue]Value Hierarchy:[/bold blue]")
            for i, value in enumerate(philosophy["top_values"], 1):
                console.print(f"  {i}. {value}")

            console.print(f"\n[bold blue]Key Principles:[/bold blue]")
            for principle in philosophy["key_principles"]:
                console.print(f"  • {principle}")

            console.print(f"\nConsistency: {philosophy['consistency_level']}")
            console.print(f"Confidence: {philosophy['confidence']:.2f}")
        else:
            console.print(f"[yellow]{philosophy['recommendation']}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error synthesizing philosophy: {e}[/red]")


@cli.command()
@click.option(
    "--preferences-dir", default="preferences", help="Directory for preference files"
)
@click.option(
    "--model",
    type=click.Choice(MODEL_CHOICES),
    default=MODEL_CHOICES[0],
    help="Model to use for contradiction analysis",
)
def check_contradictions(preferences_dir: str, model: str):
    """⚠️ Check for contradictions between preferences"""
    console.print(f"\n[bold cyan]⚠️ Contradiction Analysis[/bold cyan]")

    try:
        from agent.eval.preferences import SemanticPreferenceManager
        from agent.llm import create_llm, SupportedModel

        llm = create_llm()
        supported_model = MODEL_MAP[model]
        prefs = SemanticPreferenceManager(
            llm=llm,
            model=supported_model,
            progress_reporter=NullProgressReporter(),
            preferences_dir=preferences_dir,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Analyzing preferences for contradictions...", total=None
            )

            prefs._detect_and_resolve_contradictions()
            progress.remove_task(task)

        contradictions = prefs.get_contradictions_summary()

        if contradictions["active_count"] > 0:
            console.print(
                f"\n[bold red]⚠️ Found {contradictions['active_count']} active contradictions:[/bold red]"
            )

            for contradiction in contradictions["active_contradictions"]:
                console.print(
                    f"\n[bold yellow]{contradiction['type'].upper()}:[/bold yellow]"
                )
                console.print(f"  Description: {contradiction['description']}")
                console.print(
                    f"  Affects: {contradiction['affected_preferences']} preferences"
                )
                console.print(f"  Confidence: {contradiction['confidence']:.2f}")
        else:
            console.print(f"\n[bold green]✅ No contradictions detected![/bold green]")

        if contradictions["resolved_count"] > 0:
            console.print(
                f"\n[bold blue]📊 Previously resolved: {contradictions['resolved_count']} contradictions[/bold blue]"
            )

    except Exception as e:
        console.print(f"[red]Error checking contradictions: {e}[/red]")


@cli.command()
@click.argument("preference_id")
@click.argument("new_confidence", type=float)
@click.option(
    "--preferences-dir", default="preferences", help="Directory for preference files"
)
@click.option(
    "--model",
    type=click.Choice(MODEL_CHOICES),
    default=MODEL_CHOICES[0],
    help="Model to use",
)
@click.option(
    "--reason", default="Manual adjustment", help="Reason for confidence change"
)
def adjust_confidence(
    preference_id: str,
    new_confidence: float,
    preferences_dir: str,
    model: str,
    reason: str,
):
    """🎛️ Adjust confidence level of a specific preference"""
    console.print(f"\n[bold cyan]🎛️ Preference Confidence Adjustment[/bold cyan]")

    if not (0.0 <= new_confidence <= 1.0):
        console.print(f"[red]Error: Confidence must be between 0.0 and 1.0[/red]")
        return

    try:
        from agent.eval.preferences import SemanticPreferenceManager
        from agent.llm import create_llm, SupportedModel

        llm = create_llm()
        supported_model = MODEL_MAP[model]
        prefs = SemanticPreferenceManager(
            llm=llm,
            model=supported_model,
            progress_reporter=NullProgressReporter(),
            preferences_dir=preferences_dir,
        )

        success = prefs.adjust_preference_confidence(
            preference_id, new_confidence, reason
        )

        if success:
            console.print(
                f"[bold green]✅ Successfully updated preference confidence[/bold green]"
            )
        else:
            console.print(f"[red]❌ Preference not found: {preference_id}[/red]")
            console.print(
                f"Use 'optimize preferences' to list available preference IDs"
            )

    except Exception as e:
        console.print(f"[red]Error adjusting confidence: {e}[/red]")


@cli.command()
@click.argument("preference_id")
@click.option(
    "--preferences-dir", default="preferences", help="Directory for preference files"
)
@click.option(
    "--model",
    type=click.Choice(MODEL_CHOICES),
    default=MODEL_CHOICES[0],
    help="Model to use",
)
@click.option("--reason", default="Manual removal", help="Reason for removal")
def remove_preference(
    preference_id: str, preferences_dir: str, model: str, reason: str
):
    """🗑️ Remove a specific preference"""
    console.print(f"\n[bold cyan]🗑️ Preference Removal[/bold cyan]")

    try:
        from agent.eval.preferences import SemanticPreferenceManager
        from agent.llm import create_llm, SupportedModel

        llm = create_llm()
        supported_model = MODEL_MAP[model]
        prefs = SemanticPreferenceManager(
            llm=llm,
            model=supported_model,
            progress_reporter=NullProgressReporter(),
            preferences_dir=preferences_dir,
        )

        success = prefs.remove_preference(preference_id, reason)

        if success:
            console.print(
                f"[bold green]✅ Successfully removed preference[/bold green]"
            )
        else:
            console.print(f"[red]❌ Preference not found: {preference_id}[/red]")
            console.print(
                f"Use 'optimize preferences' to list available preference IDs"
            )

    except Exception as e:
        console.print(f"[red]Error removing preference: {e}[/red]")


@cli.command()
@click.option(
    "--optimization-dir",
    default="optimization_data",
    help="Base directory for optimization files",
)
def conversations(optimization_dir: str):
    """💬 View conversation dataset"""
    console.print(f"\n[bold cyan]💬 Conversation Dataset[/bold cyan]")

    try:
        from agent.eval.conversation_dataset import ConversationDataset

        # Try to find conversation dataset
        dataset_path = (
            Path(optimization_dir) / "test_conversations" / "conversation_dataset"
        )

        if not dataset_path.exists():
            console.print(
                f"[red]No conversation dataset found at: {dataset_path}[/red]"
            )
            console.print("Run sequential optimization first to create conversations.")
            return

        dataset = ConversationDataset(str(dataset_path))
        stats = dataset.get_stats()

        # Display stats
        table = Table(title="Dataset Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Conversations", str(stats["total_conversations"]))
        table.add_row("Total Evaluations", str(stats["total_evaluations"]))
        table.add_row("Unique Scenarios", str(stats["unique_scenarios"]))
        table.add_row(
            "Unique Prompt Combinations", str(stats["unique_prompt_combinations"])
        )

        console.print(table)

        # Show domain breakdown
        if stats["domain_counts"]:
            console.print(f"\n[bold blue]📊 By Domain:[/bold blue]")
            for domain, count in stats["domain_counts"].items():
                console.print(f"  {domain}: {count} conversations")

        # Show evaluator usage
        if stats["evaluator_counts"]:
            console.print(f"\n[bold blue]🧪 Evaluator Usage:[/bold blue]")
            for evaluator, count in stats["evaluator_counts"].items():
                console.print(f"  {evaluator}: {count} evaluations")

    except Exception as e:
        console.print(f"[red]Error loading conversation dataset: {e}[/red]")


@cli.command()
@click.option("--domain", default="roleplay", help="Domain to check")
@click.option(
    "--optimization-dir",
    default="optimization_data",
    help="Base directory for optimization files",
)
def status(domain: str, optimization_dir: str):
    """📈 Show optimization system status"""
    console.print(f"\n[bold cyan]📈 Sequential Optimization System Status[/bold cyan]")

    try:
        # Check if components exist
        console.print(f"[bold blue]Component Status:[/bold blue]")

        # Check domain config
        try:
            if domain == "roleplay":
                from agent.eval.domains.roleplay import RoleplayEvaluationConfig

                domain_config = RoleplayEvaluationConfig()
                console.print("  ✅ Domain configuration: Available")
            else:
                console.print("  ❌ Domain configuration: Unknown domain")
                return
        except Exception as e:
            console.print(f"  ❌ Domain configuration: Error - {e}")
            return

        # Check Level 1 (AgentEvaluator)
        try:
            from agent.eval.agent_evaluator import AgentEvaluator

            console.print("  ✅ Level 1 (AgentEvaluator): Available")
        except Exception as e:
            console.print(f"  ❌ Level 1 (AgentEvaluator): Error - {e}")

        # Check Sequential Optimizer
        try:
            from agent.eval.sequential_optimizer import SequentialOptimizer
            from agent.eval.optimization_paths import OptimizationPathManager
            from agent.eval.preferences import SemanticPreferenceManager

            path_manager = OptimizationPathManager(
                base_dir=optimization_dir, domain=domain
            )
            llm = create_llm()
            prefs = SemanticPreferenceManager(
                llm=llm,
                model=SupportedModel.MISTRAL_SMALL,
                progress_reporter=NullProgressReporter(),
                preferences_dir=str(path_manager.paths.preferences_dir),
            )
            optimizer = SequentialOptimizer(
                domain_config,
                prefs,
                llm,
                SupportedModel.MISTRAL_SMALL,
                NullProgressReporter(),
                path_manager,
            )

            console.print("  ✅ Sequential Optimizer: Available")
            console.print(
                f"    Optimization history: {len(optimizer.optimization_history)} runs"
            )
        except Exception as e:
            console.print(f"  ❌ Sequential Optimizer: Error - {e}")

        # Check conversation dataset
        try:
            from agent.eval.conversation_dataset import ConversationDataset

            dataset_path = (
                Path(optimization_dir) / "test_conversations" / "conversation_dataset"
            )
            if dataset_path.exists():
                dataset = ConversationDataset(str(dataset_path))
                stats = dataset.get_stats()
                console.print("  ✅ Conversation Dataset: Available")
                console.print(
                    f"    Total conversations: {stats['total_conversations']}"
                )
            else:
                console.print("  ❌ Conversation Dataset: Not found")
        except Exception as e:
            console.print(f"  ❌ Conversation Dataset: Error - {e}")

        # Check preferences (V3 Advanced System)
        try:
            from agent.eval.preferences import SemanticPreferenceManager
            from agent.eval.optimization_paths import OptimizationPathManager

            path_manager = OptimizationPathManager(
                base_dir=optimization_dir, domain=domain
            )
            llm = create_llm()
            prefs = SemanticPreferenceManager(
                llm=llm,
                model=SupportedModel.MISTRAL_SMALL,
                progress_reporter=NullProgressReporter(),
                preferences_dir=str(path_manager.paths.preferences_dir),
            )
            summary = prefs.get_summary()

            console.print("  ✅ Preference System V3: Available")
            console.print(f"    Schema Version: {summary['schema_version']}")
            console.print(
                f"    Total feedback sessions: {summary['total_feedback_sessions']}"
            )
            console.print(
                f"    Total preferences created: {summary['total_preferences_created']}"
            )

            # Show advanced features status
            philosophy = summary["philosophy"]
            contradictions = summary["contradictions"]
            maintenance = summary["maintenance"]

            if philosophy["status"] == "available":
                console.print(
                    f"    Philosophy: ✅ Synthesized (confidence: {philosophy['confidence']:.2f})"
                )
            else:
                console.print("    Philosophy: ❌ Not synthesized")

            if contradictions["active_count"] > 0:
                console.print(
                    f"    Contradictions: ⚠️ {contradictions['active_count']} active"
                )
            else:
                console.print("    Contradictions: ✅ None detected")

            console.print(
                f"    Auto-maintenance: {'✅' if maintenance['auto_contradiction_check'] else '❌'}"
            )

        except Exception as e:
            console.print(f"  ❌ Preference System: Error - {e}")

        # File system status
        console.print(f"\n[bold blue]File System:[/bold blue]")

        # Check optimization directory structure
        base_path = Path(optimization_dir)
        subdirs = ["preferences", "history", "test_conversations", "optimized_prompts"]

        for subdir in subdirs:
            path = base_path / subdir
            if path.exists():
                if path.is_dir():
                    file_count = len(list(path.glob("*")))
                    console.print(f"  ✅ {subdir}: {file_count} files")
                else:
                    console.print(f"  ✅ {subdir}: File exists")
            else:
                console.print(f"  ❌ {subdir}: Not found")

    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")


@cli.command()
@click.option("--domain", default="roleplay", help="Domain to check")
@click.option(
    "--optimization-dir",
    default="optimization_data",
    help="Base directory for optimization files",
)
@click.option("--run-id", help="Specific run ID to show (optional)")
@click.option(
    "--prompt-type",
    help="Specific prompt type to show (agent, simulation, agent_eval, sim_eval)",
)
@click.option("--show-diff", is_flag=True, help="Show detailed diffs between versions")
def prompt_evolution(
    domain: str, optimization_dir: str, run_id: str, prompt_type: str, show_diff: bool
):
    """📝 View prompt evolution and diffs from optimization runs"""
    console.print(f"\n[bold cyan]📝 Prompt Evolution Analysis[/bold cyan]")

    try:
        from agent.eval.optimization_paths import OptimizationPathManager
        from agent.eval.prompt_versioning import PromptVersionManager

        path_manager = OptimizationPathManager(base_dir=optimization_dir, domain=domain)
        version_manager = PromptVersionManager(path_manager, NullProgressReporter())

        # Get all prompt types to check
        prompt_types = ["agent", "simulation", "agent_eval", "sim_eval"]
        if prompt_type:
            prompt_types = [prompt_type]

        # If no run_id specified, show available runs
        if not run_id:
            console.print(f"[bold blue]Available optimization runs:[/bold blue]")

            # Get all version files to find run IDs
            all_run_ids = set()
            for ptype in ["agent", "simulation", "agent_eval", "sim_eval"]:
                versions = version_manager.get_prompt_versions(ptype)
                for version in versions:
                    all_run_ids.add(version.run_id)

            if not all_run_ids:
                console.print(
                    "[yellow]No optimization runs found. Run sequential optimization first.[/yellow]"
                )
                return

            # Show runs in a table
            table = Table(title="Available Optimization Runs")
            table.add_column("Run ID", style="cyan")
            table.add_column("Agent Versions", style="green")
            table.add_column("Simulation Versions", style="green")
            table.add_column("Agent Eval Versions", style="blue")
            table.add_column("Sim Eval Versions", style="blue")

            for rid in sorted(all_run_ids):
                agent_count = len(version_manager.get_prompt_versions("agent", rid))
                sim_count = len(version_manager.get_prompt_versions("simulation", rid))
                agent_eval_count = len(
                    version_manager.get_prompt_versions("agent_eval", rid)
                )
                sim_eval_count = len(
                    version_manager.get_prompt_versions("sim_eval", rid)
                )

                table.add_row(
                    rid,
                    str(agent_count),
                    str(sim_count),
                    str(agent_eval_count),
                    str(sim_eval_count),
                )

            console.print(table)
            console.print(
                f"\n[bold yellow]Use --run-id <run_id> to view evolution for a specific run[/bold yellow]"
            )
            return

        # Show evolution for specific run
        console.print(f"[bold blue]Run ID:[/bold blue] {run_id}")
        console.print(f"[bold blue]Domain:[/bold blue] {domain}")

        found_any = False
        for ptype in prompt_types:
            versions = version_manager.get_prompt_versions(ptype, run_id)

            if not versions:
                continue

            found_any = True
            console.print(f"\n[bold green]{'='*60}[/bold green]")
            console.print(f"[bold green]{ptype.upper()} PROMPT EVOLUTION[/bold green]")
            console.print(f"[bold green]{'='*60}[/bold green]")
            console.print(f"Total versions: {len(versions)}")

            # Show version summary
            table = Table(title=f"{ptype.title()} Prompt Versions")
            table.add_column("Version", style="cyan")
            table.add_column("Cycle", style="yellow")
            table.add_column("Step", style="green")
            table.add_column("Timestamp", style="blue")
            table.add_column("Changes", style="magenta")

            previous_content = None
            for i, version in enumerate(versions):
                timestamp_str = time.strftime(
                    "%H:%M:%S", time.localtime(version.timestamp)
                )

                content = version_manager.load_prompt_content(version)
                if previous_content:
                    changes = version_manager.get_compact_diff_summary(
                        previous_content, content
                    )
                else:
                    changes = "Initial version"

                table.add_row(
                    str(i + 1), str(version.cycle), version.step, timestamp_str, changes
                )

                previous_content = content

            console.print(table)

            # Show detailed diffs if requested
            if show_diff and len(versions) > 1:
                console.print(
                    f"\n[bold blue]Detailed Changes for {ptype.title()}:[/bold blue]"
                )

                previous_content = None
                for i, version in enumerate(versions):
                    content = version_manager.load_prompt_content(version)

                    if previous_content:
                        console.print(
                            f"\n[bold yellow]Changes in Version {i+1} (Cycle {version.cycle}, {version.step}):[/bold yellow]"
                        )

                        diff = version_manager.create_prompt_diff(
                            previous_content, content, f"Version {i}", f"Version {i+1}"
                        )

                        if diff == "No changes detected.":
                            console.print("[dim]No changes detected.[/dim]")
                        else:
                            # Show complete diff with proper color coding
                            diff_lines = diff.split("\n")
                            for line in diff_lines:
                                if line.startswith("+++") or line.startswith("---"):
                                    continue  # Skip file headers
                                elif line.startswith("@@"):
                                    console.print(f"[dim]{line}[/dim]")
                                elif line.startswith("+"):
                                    console.print(f"[green]+ {line[1:]}[/green]")
                                elif line.startswith("-"):
                                    console.print(f"[red]- {line[1:]}[/red]")
                                else:
                                    # Context line
                                    console.print(f"  {line}")

                    previous_content = content

        if not found_any:
            console.print(
                f"[yellow]No prompt versions found for run ID: {run_id}[/yellow]"
            )
            console.print("Check available runs with: optimize prompt-evolution")

        # Show evolution reports if they exist
        console.print(f"\n[bold blue]Evolution Reports:[/bold blue]")
        reports_found = False
        for ptype in prompt_types:
            report_file = (
                path_manager.paths.optimized_prompts_dir
                / f"{domain}_{ptype}_{run_id}_evolution_report.md"
            )
            if report_file.exists():
                console.print(f"  📋 {ptype}: {report_file}")
                reports_found = True

        if not reports_found:
            console.print("  No evolution reports found for this run.")

    except Exception as e:
        console.print(f"[red]Error analyzing prompt evolution: {e}[/red]")


if __name__ == "__main__":
    cli()
