"""
Generic presenter that mimics current CLI behavior
"""

from rich.console import Console
from agent.agent_events import (
    AgentTextEvent,
    ToolStartedEvent,
    ToolFinishedEvent,
    AgentErrorEvent,
)
from . import BasePresenter


class GenericPresenter(BasePresenter):
    """Generic presenter for non-roleplay agents"""

    def __init__(self, agent):
        super().__init__(agent)
        self.console = Console()
        self.active_tools = {}

    def process_stream(self, user_input: str) -> None:
        """Process events with generic presentation"""
        self.console.print(f"\n[bold green]🤖 Agent[/bold green]: ", end="")

        for event in self.agent.chat_stream(user_input):
            if isinstance(event, AgentTextEvent):
                self.console.print(event.content, end="", style="green")

            elif isinstance(event, ToolStartedEvent):
                self.console.print(
                    f"\n[dim cyan]🔧 {event.tool_name} ({event.tool_id})[/dim cyan]"
                )
                self.active_tools[event.tool_id] = event.tool_name

            elif isinstance(event, ToolFinishedEvent):
                tool_name = self.active_tools.get(event.tool_id, "Unknown")
                self.console.print(f"[dim green]✓ {tool_name} completed[/dim green]")
                self.console.print(
                    f"[dim blue]📋 {event.result[:100]}{'...' if len(event.result) > 100 else ''}[/dim blue]"
                )

            elif isinstance(event, AgentErrorEvent):
                self.console.print(f"[red]❌ Error: {event.message}[/red]")

        self.console.print()  # Final newline
