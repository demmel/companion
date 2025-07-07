#!/usr/bin/env python3
"""
Prompt Optimization CLI

Command-line interface for the multi-tier prompt optimization framework.
Provides access to Level 1 (AgentEvaluator), Level 2 (PromptOptimizer), 
and Level 3 (MetaPromptOptimizer) systems.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path

console = Console()


@click.group()
def cli():
    """🚀 Multi-Tier Prompt Optimization Framework"""
    console.print(
        Panel.fit(
            Text("🚀 Multi-Tier Prompt Optimization Framework", style="bold blue"),
            border_style="blue",
        )
    )


@cli.command()
@click.option("--domain", default="roleplay", help="Domain to evaluate (default: roleplay)")
@click.option("--model", default="huihui_ai/mistral-small-abliterated", help="Model to use for agent")
@click.option("--scenario", help="Specific scenario to test (optional)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def evaluate(domain: str, model: str, scenario: str, verbose: bool):
    """🧪 Level 1: Run agent evaluation for a domain"""
    console.print(f"\n[bold cyan]Level 1: Agent Evaluation[/bold cyan]")
    console.print(f"Domain: {domain}")
    console.print(f"Model: {model}")
    
    try:
        # Import and setup
        from agent.eval.agent_evaluator import AgentEvaluator
        
        if domain == "roleplay":
            from agent.eval.domains.roleplay import RoleplayEvaluationConfig
            domain_config = RoleplayEvaluationConfig()
        else:
            console.print(f"[red]Unknown domain: {domain}[/red]")
            console.print("Available domains: roleplay")
            return
        
        evaluator = AgentEvaluator(domain_eval_config=domain_config, agent_model=model)
        
        # Get scenarios
        eval_config = domain_config.get_evaluation_config()
        scenarios = [scenario] if scenario else eval_config.test_scenarios[:3]  # Test first 3
        
        console.print(f"Testing {len(scenarios)} scenarios...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for i, test_scenario in enumerate(scenarios, 1):
                task = progress.add_task(f"Scenario {i}/{len(scenarios)}: {test_scenario[:50]}...", total=None)
                
                try:
                    result = evaluator.run_evaluation(test_scenario)
                    
                    console.print(f"\n[bold green]✅ Scenario {i} Results:[/bold green]")
                    console.print(f"Overall Score: [bold]{result.overall_score:.1f}/10[/bold]")
                    console.print(f"Feedback: {result.feedback}")
                    
                    if result.scores:
                        table = Table(title="Detailed Scores")
                        table.add_column("Criterion", style="cyan")
                        table.add_column("Score", style="green", justify="right")
                        
                        for criterion, score in result.scores.items():
                            table.add_row(criterion, f"{score:.1f}/10")
                        console.print(table)
                    
                    if result.suggested_improvements:
                        console.print("[bold yellow]Suggested Improvements:[/bold yellow]")
                        for improvement in result.suggested_improvements:
                            console.print(f"  • {improvement}")
                    
                except Exception as e:
                    console.print(f"[red]❌ Scenario {i} failed: {e}[/red]")
                    if verbose:
                        import traceback
                        console.print(f"[red]{traceback.format_exc()}[/red]")
                
                progress.remove_task(task)
        
        console.print(f"\n[bold green]✅ Evaluation complete![/bold green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")


@cli.command()
@click.option("--domain", default="roleplay", help="Domain to optimize (default: roleplay)")
@click.option("--prompt-type", type=click.Choice(["agent", "simulation", "evaluation"]), default="simulation", help="Type of prompt to optimize")
@click.option("--max-iterations", default=10, help="Maximum optimization iterations")
@click.option("--temperature", default=1.0, help="Initial temperature for simulated annealing")
@click.option("--cooling-rate", default=0.95, help="Cooling rate for simulated annealing")
@click.option("--optimization-dir", default="optimization_data", help="Base directory for optimization files")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def optimize(domain: str, prompt_type: str, max_iterations: int, temperature: float, 
             cooling_rate: float, optimization_dir: str, verbose: bool):
    """🎯 Level 2: Optimize prompts using intelligent optimization"""
    console.print(f"\n[bold cyan]Level 2: Intelligent Prompt Optimization[/bold cyan]")
    console.print(f"Domain: {domain}")
    console.print(f"Prompt type: {prompt_type}")
    console.print(f"Max iterations: {max_iterations}")
    
    try:
        # Import and setup
        from agent.eval.prompt_optimizer import IntelligentPromptOptimizer
        from agent.eval.optimization_paths import OptimizationPathManager
        
        if domain == "roleplay":
            from agent.eval.domains.roleplay import RoleplayEvaluationConfig
            domain_config = RoleplayEvaluationConfig()
        else:
            console.print(f"[red]Unknown domain: {domain}[/red]")
            return
        
        # Create path manager
        path_manager = OptimizationPathManager(base_dir=optimization_dir, domain=domain)
        
        optimizer = IntelligentPromptOptimizer(domain_config, path_manager)
        
        console.print(f"\nStarting optimization...")
        
        # Run optimization
        result = optimizer.optimize_prompt(
            prompt_type=prompt_type,
            initial_temperature=temperature,
            cooling_rate=cooling_rate,
            max_iterations=max_iterations
        )
        
        # Display results
        console.print(f"\n[bold green]🎯 Optimization Results:[/bold green]")
        
        table = Table(title="Optimization Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Success", "✅ Yes" if result.success else "❌ No")
        table.add_row("Original Score", f"{result.original_score:.2f}/10")
        table.add_row("Final Score", f"{result.optimized_score:.2f}/10")
        table.add_row("Improvement", f"{result.improvement:+.2f}")
        table.add_row("Iterations", str(result.iterations))
        table.add_row("Feedback Sessions", str(result.feedback_sessions_used))
        table.add_row("Duration", f"{result.duration_seconds:.1f}s")
        table.add_row("Final Confidence", f"{result.final_confidence:.2f}")
        
        console.print(table)
        
        if result.success:
            console.print(f"\n[bold green]📝 Optimized prompt saved![/bold green]")
            console.print(f"The improved prompt has been applied to the {prompt_type} configuration.")
        
        # Show optimization history summary
        if len(optimizer.optimization_history) > 1:
            console.print(f"\n[bold blue]📊 Optimization History:[/bold blue]")
            history_table = Table()
            history_table.add_column("Run", style="cyan")
            history_table.add_column("Type", style="yellow")
            history_table.add_column("Improvement", style="green")
            history_table.add_column("Success", style="magenta")
            
            for i, run in enumerate(optimizer.optimization_history[-5:], 1):  # Last 5 runs
                history_table.add_row(
                    str(i),
                    run.prompt_type,
                    f"{run.improvement:+.2f}",
                    "✅" if run.success else "❌"
                )
            console.print(history_table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")


@cli.command()
@click.option("--domain", default="roleplay", help="Domain to meta-optimize (default: roleplay)")
@click.option("--max-iterations", default=5, help="Maximum meta-optimization iterations")
@click.option("--optimization-dir", default="optimization_data", help="Base directory for optimization files")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def meta_optimize(domain: str, max_iterations: int, optimization_dir: str, verbose: bool):
    """🚀 Level 3: Meta-optimize the optimization process itself"""
    console.print(f"\n[bold cyan]Level 3: Meta-Prompt Optimization[/bold cyan]")
    console.print(f"Domain: {domain}")
    console.print(f"Max iterations: {max_iterations}")
    
    try:
        # Import and setup
        from agent.eval.meta_prompt_optimizer import IntelligentMetaOptimizer
        from agent.eval.optimization_paths import OptimizationPathManager
        
        if domain == "roleplay":
            from agent.eval.domains.roleplay import RoleplayEvaluationConfig
            domain_config = RoleplayEvaluationConfig()
        else:
            console.print(f"[red]Unknown domain: {domain}[/red]")
            return
        
        # Create path manager
        path_manager = OptimizationPathManager(base_dir=optimization_dir, domain=domain)
        
        meta_optimizer = IntelligentMetaOptimizer(domain_config, path_manager)
        
        # Show current status
        summary = meta_optimizer.get_meta_summary()
        console.print(f"\n[bold blue]📊 Current Status:[/bold blue]")
        console.print(f"  Optimization runs in history: {summary['total_optimization_runs']}")
        console.print(f"  Meta-iterations completed: {summary['total_meta_iterations']}")
        console.print(f"  Strategies tested: {summary['strategies_tested']}")
        console.print(f"  Has best strategy: {'✅' if summary['has_best_strategy'] else '❌'}")
        
        console.print(f"\nStarting meta-optimization...")
        console.print(f"[yellow]Note: This will run actual Level 2 optimizations to test strategies.[/yellow]")
        
        # Run meta-optimization
        result = meta_optimizer.run_meta_optimization(max_iterations=max_iterations)
        
        # Display results
        console.print(f"\n[bold green]🚀 Meta-Optimization Results:[/bold green]")
        
        table = Table(title="Meta-Optimization Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Strategy Effectiveness", f"{result.strategy_effectiveness:.2f}/10")
        table.add_row("User Satisfaction Proxy", f"{result.user_satisfaction_proxy:.2f}")
        table.add_row("Efficiency Score", f"{result.efficiency_score:.2f}")
        table.add_row("Robustness Score", f"{result.robustness_score:.2f}")
        table.add_row("Test Runs Completed", str(len(result.test_runs)))
        
        console.print(table)
        
        # Show best strategy
        best_strategy = result.strategy
        console.print(f"\n[bold blue]🎯 Best Strategy Found:[/bold blue]")
        console.print(f"Description: {best_strategy.strategy_description}")
        console.print(f"Interruption Sensitivity: {best_strategy.interruption_sensitivity:.2f}")
        console.print(f"Preference Weight: {best_strategy.preference_weight:.2f}")
        console.print(f"Exploration Temperature: {best_strategy.exploration_temperature:.2f}")
        console.print(f"Feedback Frequency: {best_strategy.feedback_frequency}")
        
        console.print(f"\n[bold green]💾 Best strategy saved to {history_dir}/best_optimization_strategy.json[/bold green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")


@cli.command()
@click.option("--preferences-dir", default="preferences", help="Directory for preference files")
def preferences(preferences_dir: str):
    """📊 View learned user preferences"""
    console.print(f"\n[bold cyan]📊 User Preferences[/bold cyan]")
    
    try:
        from agent.eval.semantic_preferences_v2 import SemanticPreferenceManager
        
        prefs = SemanticPreferenceManager(preferences_dir)
        summary = prefs.get_summary()
        
        # Overall summary
        table = Table(title="Preference Summary")
        table.add_column("Domain", style="cyan")
        table.add_column("Preferences", style="green")
        table.add_column("Confidence", style="yellow")
        
        for domain in ["conversation", "simulation", "evaluation"]:
            count = summary.get(f"{domain}_preferences_count", 0)
            confidence = summary["confidence_levels"].get(domain, 0.0)
            table.add_row(domain.title(), str(count), f"{confidence:.2f}")
        
        console.print(table)
        
        console.print(f"\nTotal feedback sessions: {summary['total_feedback_sessions']}")
        
        if summary['core_values']:
            console.print(f"\n[bold blue]🎯 Core Values:[/bold blue]")
            for value in summary['core_values']:
                console.print(f"  • {value}")
        
        # Show some recent preferences
        for domain in ["conversation", "simulation", "evaluation"]:
            domain_prefs = prefs.get_preferences(domain)
            if domain_prefs:
                console.print(f"\n[bold green]{domain.title()} Preferences:[/bold green]")
                recent_prefs = sorted(domain_prefs, key=lambda p: p.last_updated, reverse=True)[:3]
                
                for pref in recent_prefs:
                    icon = "👍" if pref.preference_type == "positive" else "👎"
                    console.print(f"  {icon} {pref.description[:80]}...")
                    console.print(f"     Confidence: {pref.confidence:.2f}")
        
    except Exception as e:
        console.print(f"[red]Error loading preferences: {e}[/red]")


@cli.command()
@click.option("--domain", default="roleplay", help="Domain to check")
@click.option("--history-dir", default="meta_optimization_history", help="Meta-optimization history directory")
def status(domain: str, history_dir: str):
    """📈 Show optimization system status"""
    console.print(f"\n[bold cyan]📈 Optimization System Status[/bold cyan]")
    
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
        
        # Check Level 2 (PromptOptimizer)
        try:
            from agent.eval.prompt_optimizer_v3 import IntelligentPromptOptimizer
            optimizer = IntelligentPromptOptimizer(domain_config)
            console.print("  ✅ Level 2 (PromptOptimizer): Available")
            console.print(f"    Optimization history: {len(optimizer.optimization_history)} runs")
        except Exception as e:
            console.print(f"  ❌ Level 2 (PromptOptimizer): Error - {e}")
        
        # Check Level 3 (MetaOptimizer)
        try:
            from agent.eval.meta_prompt_optimizer import IntelligentMetaOptimizer
            meta_optimizer = IntelligentMetaOptimizer(domain_config, history_dir)
            summary = meta_optimizer.get_meta_summary()
            console.print("  ✅ Level 3 (MetaOptimizer): Available")
            console.print(f"    Meta-iterations: {summary['total_meta_iterations']}")
            console.print(f"    Strategies tested: {summary['strategies_tested']}")
        except Exception as e:
            console.print(f"  ❌ Level 3 (MetaOptimizer): Error - {e}")
        
        # Check preferences
        try:
            from agent.eval.semantic_preferences_v2 import SemanticPreferenceManager
            prefs = SemanticPreferenceManager()
            summary = prefs.get_summary()
            console.print("  ✅ Preference System: Available")
            console.print(f"    Total feedback sessions: {summary['total_feedback_sessions']}")
        except Exception as e:
            console.print(f"  ❌ Preference System: Error - {e}")
        
        # File system status
        console.print(f"\n[bold blue]File System:[/bold blue]")
        
        # Check directories
        paths_to_check = [
            ("preferences", "Preferences directory"),
            (history_dir, "Meta-optimization history"),
            (".", "Current directory")
        ]
        
        for path_str, description in paths_to_check:
            path = Path(path_str)
            if path.exists():
                if path.is_dir():
                    file_count = len(list(path.glob("*")))
                    console.print(f"  ✅ {description}: {file_count} files")
                else:
                    console.print(f"  ✅ {description}: File exists")
            else:
                console.print(f"  ❌ {description}: Not found")
        
    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")


if __name__ == "__main__":
    cli()