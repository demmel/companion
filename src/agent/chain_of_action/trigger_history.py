"""
Trigger-based history system - replaces conversation-based history with stream of consciousness approach.

Instead of back-and-forth conversation history, this tracks triggers and the agent's responses
to them, allowing for more flexible interaction patterns beyond just user messages.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from agent.chain_of_action.action.action_data import ActionData
from agent.chain_of_action.trigger import BaseTrigger, UserInputTrigger, Trigger
from agent.chain_of_action.action.base_action_data import BaseActionData
from agent.chain_of_action.action.action_types import ActionType
from agent.types import (
    Message,
    UserMessage,
    AgentMessage,
    TextContent,
    ThoughtContent,
    ToolCallFinished,
    ToolCallSuccess,
    TextToolContent,
)


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
        self.summaries: List[SummaryRecord] = []

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
        count = len(self.entries)
        if self.summaries:
            # Show all entries after the last summary
            # insert_at_index includes UI offset (1 + len(summaries)),
            # but we only want to subtract the actual entries summarized
            last_summary_index = self.summaries[-1].insert_at_index
            entries_summarized = last_summary_index - len(self.summaries)
            count -= entries_summarized
        return self.entries[-count:] if count > 0 else self.entries.copy()

    def get_old_entries(self) -> List[TriggerHistoryEntry]:
        """Get entries that are not in the recent/stream of consciousness section"""
        if not self.summaries:
            # No summaries, so no "old" entries - everything is recent
            return []

        # Get entries before the last summary cutoff
        last_summary_index = self.summaries[-1].insert_at_index
        entries_summarized = last_summary_index - len(self.summaries)
        return self.entries[:entries_summarized]

    def get_entries_before_index(self, end_index: int) -> List[TriggerHistoryEntry]:
        """Get all entries before the specified index for summarization"""
        return self.entries[:end_index]

    def get_recent_summary(self) -> Optional[SummaryRecord]:
        """Get the most recent summary record, if any"""
        return self.summaries[-1] if self.summaries else None

    def add_summary(self, summary_text: str, insert_at_index: int):
        """
        Add a new summary that should appear at the given index in the UI.

        Args:
            summary_text: The summary text
            insert_at_index: Where this summary should appear chronologically in the UI
        """
        summary_record = SummaryRecord(
            summary_text=summary_text, insert_at_index=insert_at_index
        )
        self.summaries.append(summary_record)

    def get_all_entries(self) -> List[TriggerHistoryEntry]:
        """Get all trigger history entries"""
        return self.entries.copy()

    def get_all_summaries(self) -> List[SummaryRecord]:
        """Get all summary records"""
        return self.summaries.copy()

    def get_most_recent_summary(self) -> Optional[str]:
        """Get the most recent summary text, if any"""
        return self.summaries[-1].summary_text if self.summaries else None

    def __len__(self) -> int:
        return len(self.entries)
