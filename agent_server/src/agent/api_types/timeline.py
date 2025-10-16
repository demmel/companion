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
    situational_context: str
    compressed_summary: str | None = None


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
        situational_context=entry.situational_context,
        compressed_summary=entry.compressed_summary,
    )


def build_timeline_page(
    all_entries: list[BackendTriggerHistoryEntry],
    page_size: int,
    before_index: int | None = None,
    after_index: int | None = None,
) -> tuple[list[TimelineEntry], PaginationInfo]:
    """
    Build a page of timeline entries with pagination info.

    Used by both REST /api/timeline endpoint and WebSocket hydration.

    Args:
        all_entries: All trigger history entries in chronological order
        page_size: Number of items per page
        before_index: Get entries before this index (older entries)
        after_index: Get entries after this index (newer entries)

    Returns:
        Tuple of (timeline_entries, pagination_info)
    """
    # Build timeline entries (currently just triggers, summaries removed)
    timeline_entries: list[TimelineEntry] = []
    for entry in all_entries:
        entry_dto = convert_trigger_history_entry_to_dto(entry)
        timeline_entries.append(TimelineEntryTrigger(entry=entry_dto))

    total_items = len(timeline_entries)

    # Handle pagination
    if before_index is not None:
        # Get entries before the specified index (older entries)
        end_index = min(before_index, total_items)
        start_index = max(0, end_index - page_size)
    elif after_index is not None:
        # Get entries after the specified index (newer entries)
        start_index = min(after_index, total_items)
        end_index = min(start_index + page_size, total_items)
    else:
        # Default to showing the last page (most recent items)
        start_index = max(0, total_items - page_size)
        end_index = total_items

    # Get page of items
    page_entries = timeline_entries[start_index:end_index]

    # Calculate pagination info
    has_next = end_index < total_items
    has_previous = start_index > 0

    next_cursor = str(end_index) if has_next else None
    previous_cursor = str(start_index) if has_previous else None

    pagination = PaginationInfo(
        total_items=total_items,
        page_size=page_size,
        has_next=has_next,
        has_previous=has_previous,
        next_cursor=next_cursor,
        previous_cursor=previous_cursor,
    )

    return (page_entries, pagination)
