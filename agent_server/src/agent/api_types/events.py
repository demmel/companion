# Streaming event types for real-time updates
from agent.api_types.actions import Action
from agent.api_types.triggers import Trigger
from agent.api_types.timeline import TimelineEntry, PaginationInfo
from pydantic import BaseModel
from typing_extensions import Literal


class TriggerStartedEvent(BaseModel):
    """Event emitted when a new trigger starts processing"""

    type: Literal["trigger_started"] = "trigger_started"
    trigger: Trigger
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
    action: Action
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
TriggerEvent = (
    TriggerStartedEvent
    | ActionStartedEvent
    | ActionProgressEvent
    | ActionCompletedEvent
    | TriggerCompletedEvent
    | AgentErrorEvent
    | SummarizationStartedEvent
    | SummarizationFinishedEvent
)


# Alias for backward compatibility
AgentEvent = TriggerEvent


class EventEnvelope(BaseModel):
    """Envelope for events with sequence tracking for streaming protocol"""

    type: Literal["event_envelope"] = "event_envelope"
    event_sequence: int
    trigger_id: str
    event: AgentEvent


class HydrationResponse(BaseModel):
    """Response to hydration request containing timeline page and pagination info"""

    type: Literal["hydration_response"] = "hydration_response"
    entries: list[TimelineEntry]
    pagination: PaginationInfo


# Discriminated union for all server events sent to clients
AgentServerEvent = HydrationResponse | EventEnvelope
