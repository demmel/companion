"""
API Data Transfer Objects (DTOs) for frontend communication.

These types decouple the API contract from internal backend types,
allowing for independent evolution of internal structures.
"""

from typing import Union, Literal, Annotated, Optional, List
from pydantic import BaseModel, Field

from agent.chain_of_action.action.action_data import ActionData, create_context_given
from agent.chain_of_action.action.base_action_data import BaseActionData


# Trigger DTOs
class UserInputTriggerDTO(BaseModel):
    """DTO for user input triggers"""

    type: Literal["user_input"] = "user_input"
    content: str
    user_name: str
    timestamp: str
    image_urls: Optional[List[str]] = None


class WakeupTriggerDTO(BaseModel):
    """DTO for wakeup triggers"""

    type: Literal["wakeup"] = "wakeup"
    timestamp: str


# Discriminated union for all trigger types
TriggerDTO = Union[UserInputTriggerDTO, WakeupTriggerDTO]


# Action DTOs
class ActionStatusSuccess(BaseModel):
    type: Literal["success"] = "success"
    result: str


class ActionStatusError(BaseModel):
    type: Literal["error"] = "error"
    error: Optional[str] = None


class ActionStatusStreaming(BaseModel):
    type: Literal["streaming"] = "streaming"
    result: str


ActionStatus = Union[ActionStatusSuccess, ActionStatusError, ActionStatusStreaming]


class BaseActionDTO(BaseModel):
    """Base class for action DTOs"""

    context_given: str
    reasoning: str
    status: ActionStatus
    duration_ms: int


class ThinkActionDTO(BaseActionDTO):
    """DTO for think actions"""

    type: Literal["think"] = "think"


class SpeakActionDTO(BaseActionDTO):
    """DTO for speak actions"""

    type: Literal["speak"] = "speak"


class UpdateAppearanceActionDTO(BaseActionDTO):
    """DTO for appearance update actions"""

    type: Literal["update_appearance"] = "update_appearance"
    image_description: str
    image_url: str


class UpdateEnvironmentActionDTO(BaseActionDTO):
    """DTO for environment update actions"""

    type: Literal["update_environment"] = "update_environment"
    image_description: str
    image_url: str


class UpdateMoodActionDTO(BaseActionDTO):
    """DTO for mood update actions"""

    type: Literal["update_mood"] = "update_mood"


class WaitActionDTO(BaseActionDTO):
    """DTO for wait actions"""

    type: Literal["wait"] = "wait"


class AddPriorityActionDTO(BaseActionDTO):
    """DTO for add priority actions"""

    type: Literal["add_priority"] = "add_priority"


class RemovePriorityActionDTO(BaseActionDTO):
    """DTO for remove priority actions"""

    type: Literal["remove_priority"] = "remove_priority"


class FetchUrlActionDTO(BaseActionDTO):
    """DTO for fetch URL actions"""

    type: Literal["fetch_url"] = "fetch_url"
    url: str
    looking_for: str


class SearchResultDTO(BaseModel):
    """DTO for individual search results"""

    url: str
    title: str
    snippet: str


class SearchWebActionDTO(BaseActionDTO):
    """DTO for web search actions"""

    type: Literal["search_web"] = "search_web"
    query: str
    purpose: str
    search_results: List[SearchResultDTO]


# Discriminated union for all action types
ActionDTO = Union[
    ThinkActionDTO,
    SpeakActionDTO,
    UpdateAppearanceActionDTO,
    UpdateEnvironmentActionDTO,
    UpdateMoodActionDTO,
    WaitActionDTO,
    AddPriorityActionDTO,
    FetchUrlActionDTO,
    SearchWebActionDTO,
    RemovePriorityActionDTO,
]


class SummaryDTO(BaseModel):
    """DTO for summary records"""

    summary_text: str
    insert_at_index: int
    created_at: str


class TriggerHistoryEntryDTO(BaseModel):
    """DTO for complete trigger-response entries"""

    trigger: TriggerDTO
    actions_taken: List[ActionDTO]
    timestamp: str
    entry_id: str


class TriggerHistoryResponse(BaseModel):
    """Response model for trigger history data"""

    entries: List[TriggerHistoryEntryDTO]
    summaries: List[SummaryDTO]
    total_entries: int
    recent_entries_count: int


# Streaming event types for real-time updates
class TriggerStartedEvent(BaseModel):
    """Event emitted when a new trigger starts processing"""

    type: Literal["trigger_started"] = "trigger_started"
    trigger: TriggerDTO
    entry_id: str
    timestamp: str


class ActionStartedEvent(BaseModel):
    """Event emitted when an action starts within a trigger"""

    type: Literal["action_started"] = "action_started"
    entry_id: str
    action_type: str
    context_given: str
    reasoning: str
    sequence_number: int
    action_number: int
    timestamp: str


class ActionCompletedEvent(BaseModel):
    """Event emitted when an action completes within a trigger"""

    type: Literal["action_completed"] = "action_completed"
    action: ActionDTO
    entry_id: str
    sequence_number: int
    action_number: int
    timestamp: str


class TriggerCompletedEvent(BaseModel):
    """Event emitted when all actions for a trigger are complete"""

    type: Literal["trigger_completed"] = "trigger_completed"
    entry_id: str
    total_actions: int
    successful_actions: int
    timestamp: str
    # Context information for UI updates
    estimated_tokens: int
    context_limit: int
    usage_percentage: float
    approaching_limit: bool


class ActionProgressEvent(BaseModel):
    """Event emitted for streaming action progress"""

    type: Literal["action_progress"] = "action_progress"
    entry_id: str
    action_type: str
    partial_result: str
    sequence_number: int
    action_number: int
    timestamp: str


class AgentErrorEvent(BaseModel):
    """Event emitted when agent encounters an error"""

    type: Literal["error"] = "error"
    message: str
    entry_id: str = ""


class SummarizationStartedEvent(BaseModel):
    """Event emitted when auto-summarization starts"""

    type: Literal["summarization_started"] = "summarization_started"
    entries_to_summarize: int
    recent_entries_kept: int
    context_usage_before: float


class SummarizationFinishedEvent(BaseModel):
    """Event emitted when auto-summarization completes"""

    type: Literal["summarization_finished"] = "summarization_finished"
    summary: str
    entries_summarized: int
    entries_after: int
    context_usage_after: float


# Union type for all trigger-based events
TriggerEvent = Union[
    TriggerStartedEvent,
    ActionStartedEvent,
    ActionProgressEvent,
    ActionCompletedEvent,
    TriggerCompletedEvent,
    AgentErrorEvent,
    SummarizationStartedEvent,
    SummarizationFinishedEvent,
]

# Alias for backward compatibility
AgentEvent = TriggerEvent


# Conversion functions from backend types to DTOs
def convert_trigger_to_dto(trigger) -> TriggerDTO:
    """Convert backend TriggerEvent to DTO"""
    from agent.chain_of_action.trigger import UserInputTrigger, WakeupTrigger

    if isinstance(trigger, UserInputTrigger):
        # Convert file paths to URLs for client
        image_urls = None
        if trigger.image_paths:
            from pathlib import Path

            image_urls = [
                f"/uploaded_images/{Path(path).name}" for path in trigger.image_paths
            ]

        return UserInputTriggerDTO(
            content=trigger.content,
            user_name=trigger.user_name,
            timestamp=trigger.timestamp.isoformat(),
            image_urls=image_urls,
        )
    elif isinstance(trigger, WakeupTrigger):
        return WakeupTriggerDTO(
            timestamp=trigger.timestamp.isoformat(),
        )
    else:
        raise ValueError(f"Unsupported trigger type: {type(trigger)}")


def convert_action_to_dto(action: ActionData) -> ActionDTO:
    """Convert backend ActionResult to DTO"""
    from agent.chain_of_action.action.action_types import ActionType

    # Convert success/result_summary to ActionStatus
    if action.result.type == "success":
        status = ActionStatusSuccess(result=action.result.content.result_summary())
    else:
        status = ActionStatusError(error=action.result.error)

    context_given = create_context_given(action)
    base_data = {
        "context_given": context_given,
        "reasoning": action.reasoning,
        "status": status,
        "duration_ms": int(action.duration_ms) if action.duration_ms is not None else 0,
    }

    if action.type == ActionType.THINK:
        return ThinkActionDTO(**base_data)
    elif action.type == ActionType.SPEAK:
        return SpeakActionDTO(**base_data)
    elif action.type == ActionType.UPDATE_APPEARANCE:
        image_description = None
        image_url = None

        if action.result.type == "success":
            image_description = action.result.content.image_description
            image_url = action.result.content.image_result.image_url
        else:
            raise ValueError("Invalid result type for UpdateAppearanceAction")

        return UpdateAppearanceActionDTO(
            **base_data, image_description=image_description, image_url=image_url
        )
    elif action.type == ActionType.UPDATE_ENVIRONMENT:
        image_description = None
        image_url = None

        if action.result.type == "success":
            image_description = action.result.content.image_description
            image_url = action.result.content.image_result.image_url
        else:
            raise ValueError("Invalid result type for UpdateEnvironmentAction")

        return UpdateEnvironmentActionDTO(
            **base_data, image_description=image_description, image_url=image_url
        )
    elif action.type == ActionType.UPDATE_MOOD:
        return UpdateMoodActionDTO(**base_data)
    elif action.type == ActionType.WAIT:
        return WaitActionDTO(**base_data)
    elif action.type == ActionType.ADD_PRIORITY:
        return AddPriorityActionDTO(**base_data)
    elif action.type == ActionType.REMOVE_PRIORITY:
        return RemovePriorityActionDTO(**base_data)
    elif action.type == ActionType.FETCH_URL:
        url = action.input.url
        looking_for = action.input.looking_for
        return FetchUrlActionDTO(**base_data, url=url, looking_for=looking_for)
    elif action.type == ActionType.SEARCH_WEB:
        query = action.input.query
        purpose = action.input.purpose
        search_results = []

        if action.result.type == "success":
            # Convert SearchResult objects to SearchResultDTO objects
            search_results = [
                SearchResultDTO(
                    url=result.url, title=result.title, snippet=result.snippet
                )
                for result in action.result.content.search_results
            ]

        return SearchWebActionDTO(
            **base_data, query=query, purpose=purpose, search_results=search_results
        )
    else:
        raise ValueError(f"Unsupported action type: {action.action}")


def convert_summary_to_dto(summary) -> SummaryDTO:
    """Convert backend SummaryRecord to DTO"""
    return SummaryDTO(
        summary_text=summary.summary_text,
        insert_at_index=summary.insert_at_index,
        created_at=summary.created_at.isoformat(),
    )


def convert_trigger_history_entry_to_dto(entry) -> TriggerHistoryEntryDTO:
    """Convert backend TriggerHistoryEntry to DTO"""
    return TriggerHistoryEntryDTO(
        trigger=convert_trigger_to_dto(entry.trigger),
        actions_taken=[convert_action_to_dto(action) for action in entry.actions_taken],
        timestamp=entry.timestamp.isoformat(),
        entry_id=entry.entry_id,
    )


# Timeline pagination types
class TimelineEntryTrigger(BaseModel):
    """Timeline entry for a trigger (complete trigger-response pair)"""

    type: Literal["trigger"] = "trigger"
    entry: TriggerHistoryEntryDTO


class TimelineEntrySummary(BaseModel):
    """Timeline entry for a summary"""

    type: Literal["summary"] = "summary"
    summary: SummaryDTO


# Union type for timeline entries
TimelineEntry = Union[TimelineEntryTrigger, TimelineEntrySummary]


class PaginationInfo(BaseModel):
    """Pagination information for timeline responses"""

    total_items: int
    page_size: int
    has_next: bool
    has_previous: bool
    next_cursor: Optional[str] = None
    previous_cursor: Optional[str] = None


class TimelineResponse(BaseModel):
    """Response model for paginated timeline data"""

    entries: List[TimelineEntry]
    pagination: PaginationInfo
