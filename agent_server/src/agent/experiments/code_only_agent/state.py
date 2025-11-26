"""State management for code-only agent."""

from dataclasses import dataclass, field
from typing import Optional

from agent.chain_of_action.prompts import format_section
from agent.experiments.code_only_agent.execution import OutputMessage
from agent.state import Value, Priority


@dataclass
class Iteration:
    reasoning: str
    code: Optional[str]  # None if agent decides it's done
    outputs: list[OutputMessage] = field(default_factory=list)


@dataclass
class AgentTurn:
    user_input: str
    iterations: list[Iteration]

    def get_speaks(self) -> list[str]:
        """Extract all speak messages from iterations."""
        from agent.experiments.code_only_agent.execution import SpeakMessage

        speaks = []
        for iteration in self.iterations:
            for output in iteration.outputs:
                if isinstance(output, SpeakMessage):
                    speaks.append(output.content)
        return speaks


class State:
    def __init__(self):
        from agent.experiments.code_only_agent.functions import (
            FunctionTree,
            FunctionCache,
        )

        # Personality (initialized on first turn, static after)
        self.name: str = ""
        self.role: str = ""
        self.core_values: list[Value] = []
        self.current_priorities: list[Priority] = []
        self.next_priority_id: int = 1

        # Agent-specific
        self.history: list[AgentTurn] = []
        self.function_tree = FunctionTree()
        self.function_cache = FunctionCache()

    def is_first_turn(self) -> bool:
        """Check if this is the first turn (personality not initialized)."""
        return not self.name

    def format_history(self, last_n_turns: int = 3) -> str:
        """Format recent agent turns for LLM."""
        if not self.history:
            return "(no history)"

        from agent.experiments.code_only_agent.execution import format_output_message

        recent = self.history[-last_n_turns:]
        lines = []
        for turn in recent:
            lines.append(f"User: {turn.user_input}")
            for i, iteration in enumerate(turn.iterations, 1):
                lines.append(f"  Iteration {i}:")
                lines.append(f"    Reasoning: {iteration.reasoning}")
                if iteration.code:
                    lines.append(f"    Code: {iteration.code}")
                if iteration.outputs:
                    for output in iteration.outputs:
                        lines.append(f"    {format_output_message(output)}")
        return "\n".join(lines)

    def format_for_llm(self) -> str:
        sections = []

        # Include personality if initialized
        if self.name:
            values_str = (
                ", ".join(v.content for v in self.core_values)
                if self.core_values
                else "None"
            )
            priorities_str = (
                ", ".join(p.content for p in self.current_priorities)
                if self.current_priorities
                else "None"
            )
            sections.append(
                format_section(
                    "Your Identity",
                    f"Name: {self.name}\n"
                    f"Role: {self.role}\n"
                    f"Values: {values_str}\n"
                    f"Priorities: {priorities_str}",
                )
            )

        history_section = self.format_history()
        if history_section != "(no history)":
            sections.append(format_section("Recent History", history_section))

        return "\n\n".join(sections)
