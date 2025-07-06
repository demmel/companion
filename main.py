#!/usr/bin/env python3
"""
CLI Agent with Llama 3.3 70B
A general-purpose agent for various tasks.
"""


import click
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from agent.core import Agent
from agent.config import get_config
from agent.presenters import get_presenter_for_config

console = Console()


@click.command()
@click.option(
    "--model",
    default="aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m",
    help="Model to use (default: aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--check", is_flag=True, help="Check if model is available")
def main(model: str, verbose: bool, check: bool):
    """CLI Agent with Llama 3.3 70B - A general-purpose AI assistant"""

    console.print(
        Panel.fit(
            Text("🤖 CLI Agent with Llama 3.3 70B", style="bold blue"),
            border_style="blue",
        )
    )

    try:
        config = get_config("roleplay")  # Default to roleplay config
        agent = Agent(config=config, model=model, verbose=verbose)

        if check:
            console.print("Checking model availability...")
            if agent.llm.is_available():
                console.print(f"[green]✓ Model {model} is available[/green]")
            else:
                console.print(
                    f"[yellow]⚠ Model {model} not found. Attempting to pull...[/yellow]"
                )
                if agent.llm.pull_model():
                    console.print(f"[green]✓ Successfully pulled {model}[/green]")
                else:
                    console.print(f"[red]✗ Failed to pull {model}[/red]")
                    return

        console.print("\n[green]Agent ready![/green]")
        console.print("Available commands:")
        console.print("  • Type your message to chat with the agent")
        console.print("  • Type 'tools' to see available tools")
        console.print("  • Type 'prompt' to see current system prompt")
        console.print("  • Type 'state' to see current agent state")
        console.print("  • Type 'state <path> <value>' to modify state")
        console.print("  • Type 'reset' to clear conversation history")
        console.print("  • Type 'exit' to quit")

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

                if user_input.lower() in ["exit", "quit", "bye"]:
                    console.print("[yellow]👋 Goodbye![/yellow]")
                    break

                if user_input.lower() == "reset":
                    agent.reset_conversation()
                    console.print("[yellow]🔄 Conversation reset[/yellow]")
                    continue

                if user_input.lower() == "tools":
                    console.print("\n[bold]Available Tools:[/bold]")
                    console.print(agent.tools.get_tools_description())
                    continue

                if user_input.lower() == "prompt":
                    console.print("\n[bold]Current System Prompt:[/bold]")
                    current_prompt = agent._build_system_prompt(
                        include_tools=True, iteration_info=(0, 1)
                    )
                    console.print(f"[dim]{current_prompt}[/dim]")
                    continue

                if user_input.lower().startswith("state"):
                    parts = user_input.split(maxsplit=2)

                    if len(parts) == 1:
                        # Show current state
                        console.print("\n[bold]Current Agent State:[/bold]")
                        state = agent.get_state()
                        for key, value in state.items():
                            if key == "characters":
                                console.print(f"[cyan]{key}:[/cyan]")
                                for char_id, char_data in value.items():
                                    console.print(
                                        f"  [green]{char_id}:[/green] {char_data.get('name', 'Unknown')} - {char_data.get('personality', 'No personality')}"
                                    )
                                    if char_data.get("memories"):
                                        console.print(
                                            f"    [yellow]Memories:[/yellow] {len(char_data['memories'])} stored"
                                        )
                            else:
                                console.print(
                                    f"[cyan]{key}:[/cyan] {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}"
                                )
                        continue

                    elif len(parts) == 3:
                        # Set state value: state path value
                        _, path, value_str = parts
                        try:
                            # Try to parse value as JSON first, then fall back to string
                            import json

                            try:
                                value = json.loads(value_str)
                            except json.JSONDecodeError:
                                value = value_str

                            # Parse JSON path and set value
                            current_state = agent.get_state()

                            # Handle dot notation paths like "characters.char_id.name"
                            path_parts = path.split(".")
                            target = current_state

                            # Navigate to the parent of the target
                            for part in path_parts[:-1]:
                                if part not in target:
                                    target[part] = {}
                                target = target[part]

                            # Set the final value
                            final_key = path_parts[-1]
                            old_value = target.get(final_key, "Not set")
                            target[final_key] = value

                            # Update the agent state
                            agent.state = current_state

                            console.print(
                                f"[green]✓ Updated {path}:[/green] {old_value} → {value}"
                            )

                        except Exception as e:
                            console.print(f"[red]Error setting state: {e}[/red]")
                            console.print(
                                "[yellow]Usage: state <path> <value>[/yellow]"
                            )
                            console.print(
                                "[yellow]Example: state characters.abc123.name Alice[/yellow]"
                            )

                    else:
                        console.print("[yellow]Usage:[/yellow]")
                        console.print("  [cyan]state[/cyan] - Show current state")
                        console.print(
                            "  [cyan]state <path> <value>[/cyan] - Set state value"
                        )
                        console.print(
                            "  [cyan]Example:[/cyan] state characters.abc123.name Alice"
                        )

                    continue

                if not user_input.strip():
                    continue

                # Show context info before processing
                context_info = agent.get_context_info()
                context_color = (
                    "red"
                    if context_info.approaching_limit
                    else "yellow" if context_info.usage_percentage > 60 else "green"
                )
                console.print(
                    f"[{context_color}]Context: {context_info.estimated_tokens}/{context_info.context_limit} tokens ({context_info.usage_percentage:.1f}%)[/{context_color}]"
                )

                # Get presenter for this agent type and handle streaming
                presenter = get_presenter_for_config(agent.config.name, agent)
                presenter.process_stream(user_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if verbose:
                    import traceback

                    console.print(f"[red]{traceback.format_exc()}[/red]")

    except Exception as e:
        console.print(f"[red]Failed to initialize agent: {e}[/red]")
        if verbose:
            import traceback

            console.print(f"[red]{traceback.format_exc()}[/red]")


if __name__ == "__main__":
    main()
