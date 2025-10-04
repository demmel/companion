# Timeline pagination types
from typing import Literal
from agent.api_types.actions import Action, convert_action_to_dto
from agent.api_types.triggers import Trigger, convert_trigger_to_dto
from agent.chain_of_action.trigger_history import (
    SummaryRecord,
    TriggerHistoryEntry as BackendTriggerHistoryEntry,
)
from pydantic import BaseModel


class Summary(BaseModel):
    """DTO for summary records"""

    summary_text: str
    insert_at_index: int
    created_at: str


class TriggerHistoryEntry(BaseModel):
    """DTO for complete trigger-response entries"""

    trigger: Trigger
    actions_taken: list[Action]
    timestamp: str
    entry_id: str


class TimelineEntryTrigger(BaseModel):
    """Timeline entry for a trigger (complete trigger-response pair)"""

    type: Literal["trigger"] = "trigger"
    entry: TriggerHistoryEntry


class TimelineEntrySummary(BaseModel):
    """Timeline entry for a summary"""

    type: Literal["summary"] = "summary"
    summary: Summary


# Union type for timeline entries
TimelineEntry = TimelineEntryTrigger | TimelineEntrySummary


class PaginationInfo(BaseModel):
    """Pagination information for timeline responses"""

    total_items: int
    page_size: int
    has_next: bool
    has_previous: bool
    next_cursor: str | None = None
    previous_cursor: str | None = None


class TimelineResponse(BaseModel):
    """Response model for paginated timeline data"""

    entries: list[TimelineEntry]
    pagination: PaginationInfo


def convert_summary_to_dto(summary: SummaryRecord) -> Summary:
    """Convert backend SummaryRecord to DTO"""
    return Summary(
        summary_text=summary.summary_text,
        insert_at_index=summary.insert_at_index,
        created_at=summary.created_at.isoformat(),
    )


def convert_trigger_history_entry_to_dto(
    entry: BackendTriggerHistoryEntry,
) -> TriggerHistoryEntry:
    """Convert backend TriggerHistoryEntry to DTO"""
    return TriggerHistoryEntry(
        trigger=convert_trigger_to_dto(entry.trigger),
        actions_taken=[convert_action_to_dto(action) for action in entry.actions_taken],
        timestamp=entry.timestamp.isoformat(),
        entry_id=entry.entry_id,
    )
