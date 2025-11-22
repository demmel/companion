"""State management for code-only agent."""

from dataclasses import dataclass
from typing import Optional

from agent.chain_of_action.prompts import format_section
from agent.experiments.code_only_agent.functions import FunctionTree, FunctionCache


@dataclass
class Iteration:
    reasoning: str
    code: Optional[str]  # None if agent decides it's done
    output: Optional[str]  # None if no code was run


@dataclass
class AgentTurn:
    user_input: str
    iterations: list[Iteration]


class State:
    def __init__(self):
        self.history: list[AgentTurn] = []
        self.function_tree = FunctionTree()
        self.function_cache = FunctionCache()
        self.speak_messages: list[str] = []  # Store messages from speak function

    def format_history(self, last_n_turns: int = 3) -> str:
        """Format recent agent turns for LLM."""
        if not self.history:
            return "(no history)"

        recent = self.history[-last_n_turns:]
        lines = []
        for turn in recent:
            lines.append(f"User: {turn.user_input}")
            for i, iteration in enumerate(turn.iterations, 1):
                lines.append(f"  Iteration {i}:")
                lines.append(f"    Reasoning: {iteration.reasoning}")
                if iteration.code:
                    lines.append(f"    Code: {iteration.code}")
                if iteration.output:
                    lines.append(f"    Output: {iteration.output}")
        return "\n".join(lines)

    def format_for_llm(self) -> str:
        sections = []

        history_section = self.format_history()
        if history_section != "(no history)":
            sections.append(format_section("Recent History", history_section))

        return "\n\n".join(sections)
