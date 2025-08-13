"""
Trigger-based history system - replaces conversation-based history with stream of consciousness approach.

Instead of back-and-forth conversation history, this tracks triggers and the agent's responses
to them, allowing for more flexible interaction patterns beyond just user messages.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from agent.chain_of_action.trigger import BaseTriger, UserInputTrigger, Trigger
from agent.chain_of_action.context import ActionResult
from agent.chain_of_action.action_types import ActionType
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
    actions_taken: List[ActionResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    entry_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    compressed_summary: Optional[str] = (
        None  # Individual compressed summary for this trigger
    )


class TriggerHistory:
    """
    Trigger-based history that tracks stimuli and agent responses instead of conversation turns.

    This allows the agent to respond to various types of triggers (user input, timers,
    self-reflection, etc.) in a more flexible stream of consciousness approach.
    """

    def __init__(self):
        self.entries: List[TriggerHistoryEntry] = []
        self.summaries: List[SummaryRecord] = []

    def add_trigger_response(self, trigger: Trigger, actions: List[ActionResult]):
        """Add a new trigger and the agent's response to it"""
        entry = TriggerHistoryEntry(
            trigger=trigger,
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
            last_summary_index = self.summaries[-1].insert_at_index
            count -= last_summary_index
        return self.entries[-count:] if count > 0 else self.entries.copy()

    def get_entries_before_index(self, end_index: int) -> List[TriggerHistoryEntry]:
        """Get all entries before the specified index for summarization"""
        return self.entries[:end_index]

    def get_recent_summary(self) -> Optional[SummaryRecord]:
        """Get the most recent summary record, if any"""
        return self.summaries[-1] if self.summaries else None

    def to_conversation_history(self) -> List[Message]:
        """
        Convert trigger history to legacy conversation history format for compatibility.

        Maps:
        - UserInputTrigger -> UserMessage
        - ActionResults -> AgentMessage with content and tool calls
        """
        messages = []

        for entry in self.entries:
            # Convert trigger to message
            if isinstance(entry.trigger, UserInputTrigger):
                user_msg = UserMessage(
                    content=[TextContent(text=entry.trigger.content)]
                )
                messages.append(user_msg)

            # Convert actions to agent message
            if entry.actions_taken:
                content = []
                tool_calls = []

                for action in entry.actions_taken:
                    if action.action == ActionType.THINK:
                        content.append(ThoughtContent(text=action.result_summary))
                    elif action.action == ActionType.SPEAK:
                        content.append(TextContent(text=action.result_summary))
                    elif action.action in [
                        ActionType.UPDATE_APPEARANCE,
                        ActionType.UPDATE_MOOD,
                    ]:
                        # These become tool calls
                        tool_result = ToolCallSuccess(
                            content=TextToolContent(text=action.result_summary),
                            llm_feedback=action.result_summary,
                        )

                        tool_calls.append(
                            ToolCallFinished(
                                tool_name=action.action.value,
                                tool_id=f"{action.action.value}_{entry.entry_id}",
                                parameters=action.metadata or {},
                                result=tool_result,
                            )
                        )

                if content or tool_calls:
                    agent_msg = AgentMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                    )
                    messages.append(agent_msg)

        return messages

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
