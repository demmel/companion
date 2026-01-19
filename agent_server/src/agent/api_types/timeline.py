# Timeline pagination types
import itertools
from typing import Literal
from agent.api_types.actions import Action, convert_action_to_dto
from agent.api_types.triggers import Trigger, convert_trigger_to_dto
from agent.chain_of_action.trigger_history_entry import (
    SummaryRecord,
    TriggerHistoryEntry as BackendTriggerHistoryEntry,
)
from agent.storage.interface import ITriggerHistory
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
    trigger_history: ITriggerHistory,
    page_size: int,
    before_index: int | None = None,
    after_index: int | None = None,
) -> tuple[list[TimelineEntry], PaginationInfo]:
    """
    Build a page of timeline entries with pagination info.

    Uses efficient queries to fetch only the needed entries.

    Used by both REST /api/timeline endpoint and WebSocket hydration.

    Args:
        trigger_history: The trigger history to fetch entries from
        page_size: Number of items per page
        before_index: Get entries before this index (older entries)
        after_index: Get entries after this index (newer entries)

    Returns:
        Tuple of (timeline_entries, pagination_info)
    """
    total_items = trigger_history.get_entry_count()

    # Calculate start and end indices based on cursors
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

    # Fetch only the needed entries using iter_entries
    fetch_count = end_index - start_index
    entries = list(
        itertools.islice(
            trigger_history.iter_entries(reverse=False, start=start_index), fetch_count
        )
    )

    # Convert to timeline entries
    page_entries: list[TimelineEntry] = [
        TimelineEntryTrigger(entry=convert_trigger_history_entry_to_dto(entry))
        for entry in entries
    ]

    # Calculate pagination info
    has_next = end_index < total_items
    has_previous = start_index > 0

    pagination = PaginationInfo(
        total_items=total_items,
        page_size=page_size,
        has_next=has_next,
        has_previous=has_previous,
        next_cursor=str(end_index) if has_next else None,
        previous_cursor=str(start_index) if has_previous else None,
    )

    return (page_entries, pagination)
