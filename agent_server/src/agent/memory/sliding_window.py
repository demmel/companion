"""Simple sliding window memory implementation for baseline comparison."""

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.chain_of_action.trigger import format_trigger_for_prompt
from agent.llm import LLM
from agent.memory.memory import IMemory, MemoryQueries
from agent.state import State


class SlidingWindowMemory(IMemory):
    """
    Dead simple baseline memory that keeps the N most recent entries.

    No fancy retrieval - just keeps recent history and returns it all.
    Used as a baseline to answer: "Does any fancy retrieval actually beat
    'just keep recent stuff'?"
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.entries: list[TriggerHistoryEntry] = []

    def store(
        self,
        trigger_history_entry: TriggerHistoryEntry,
        state: State,
        llm: LLM,
    ) -> None:
        """Store a new entry, keeping only the most recent N."""
        self.entries.append(trigger_history_entry)

        # Keep only most recent N
        if len(self.entries) > self.window_size:
            self.entries = self.entries[-self.window_size :]

    def query(self, memory_queries: MemoryQueries) -> str:
        """Return all stored entries formatted as text."""
        return self._format_entries(self.entries)

    def _format_entries(self, entries: list[TriggerHistoryEntry]) -> str:
        """Format entries into a readable string."""
        if not entries:
            return "No memories stored."

        formatted_parts: list[str] = []

        for i, entry in enumerate(entries):
            parts: list[str] = []
            parts.append(f"[Memory {i + 1}]")

            # Use compressed_summary if available, otherwise format the trigger
            if entry.compressed_summary:
                parts.append(entry.compressed_summary)
            else:
                # Format the trigger
                parts.append(format_trigger_for_prompt(entry.trigger))

                # Add situational context if available
                if entry.situational_context:
                    parts.append(entry.situational_context)

            formatted_parts.append("\n".join(parts))

        return "\n\n".join(formatted_parts)

    def clear(self) -> None:
        """Clear all stored entries."""
        self.entries = []
