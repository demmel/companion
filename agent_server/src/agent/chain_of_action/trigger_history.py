"""
Trigger-based history system - replaces conversation-based history with stream of consciousness approach.

Instead of back-and-forth conversation history, this tracks triggers and the agent's responses
to them, allowing for more flexible interaction patterns beyond just user messages.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from agent.chain_of_action.action.action_data import ActionData
from agent.chain_of_action.trigger import Trigger


class SummaryRecord(BaseModel):
    """Record of a summary and where it should appear in the UI"""

    summary_text: str
    insert_at_index: int  # Where this summary appears in the UI chronologically
    created_at: datetime = Field(default_factory=datetime.now)


class TriggerHistoryEntry(BaseModel):
    """Single entry in trigger-based history - a trigger and agent's response to it"""

    trigger: Trigger
    actions_taken: List[ActionData] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    entry_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    situational_context: Optional[str] = Field(default=None)
    compressed_summary: Optional[str] = Field(default=None)
    embedding_vector: Optional[List[float]] = Field(default=None)


class TriggerHistory:
    """
    Trigger-based history that tracks stimuli and agent responses instead of conversation turns.

    This allows the agent to respond to various types of triggers (user input, timers,
    self-reflection, etc.) in a more flexible stream of consciousness approach.
    """

    def __init__(self):
        self.entries: List[TriggerHistoryEntry] = []

    def add_trigger_response(
        self, trigger: Trigger, situational_context: str, actions: List[ActionData]
    ):
        """Add a new trigger and the agent's response to it"""
        entry = TriggerHistoryEntry(
            trigger=trigger,
            situational_context=situational_context,
            actions_taken=actions,
        )
        self.entries.append(entry)

    def add_trigger_entry(self, entry: TriggerHistoryEntry):
        """Add a pre-built trigger history entry"""
        self.entries.append(entry)

    def get_entries(self) -> List[TriggerHistoryEntry]:
        """Get all trigger history entries"""
        return self.entries.copy()

    def get_recent_entries(self) -> List[TriggerHistoryEntry]:
        """Get the most recent trigger history entries"""
        return self.entries

    def get_old_entries(self) -> List[TriggerHistoryEntry]:
        """Get entries that are not in the recent/stream of consciousness section"""
        return self.entries

    def get_entries_before_index(self, end_index: int) -> List[TriggerHistoryEntry]:
        """Get all entries before the specified index for summarization"""
        return self.entries[:end_index]

    def get_all_entries(self) -> List[TriggerHistoryEntry]:
        """Get all trigger history entries"""
        return self.entries.copy()

    def get_entry_by_id(self, entry_id: str) -> TriggerHistoryEntry:
        """Get a trigger by its entry ID"""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        raise ValueError(f"TriggerHistoryEntry with ID {entry_id} not found")

    def __len__(self) -> int:
        return len(self.entries)
