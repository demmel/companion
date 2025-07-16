"""
Agent-level streaming events for the chat_stream API
"""

from typing import Dict, Any, Optional
from enum import Enum
from typing_extensions import Literal

from pydantic import BaseModel

from agent.types import ToolResult


class AgentTextEvent(BaseModel):
    """Text content from agent response"""

    content: str
    is_thought: bool = False
    type: Literal["text"] = "text"


class ToolStartedEvent(BaseModel):
    """Tool execution started"""

    tool_name: str
    tool_id: str
    parameters: Dict[str, Any]
    type: Literal["tool_started"] = "tool_started"


class ToolProgressEvent(BaseModel):
    """Progress update from tool execution"""

    tool_id: str
    data: str
    progress: Optional[float] = None  # 0.0 to 1.0
    type: Literal["tool_progress"] = "tool_progress"


class ToolFinishedEvent(BaseModel):
    """Tool execution completed"""

    tool_id: str
    result: ToolResult
    type: Literal["tool_finished"] = "tool_finished"


class UserInputRequestEvent(BaseModel):
    """Placeholder for interactive tools that need user input"""

    # TODO: Define structure when we implement interactive tools
    type: Literal["user_input_request"] = "user_input_request"


class AgentErrorEvent(BaseModel):
    """Error in agent processing"""

    message: str
    tool_name: str = ""
    tool_id: str = ""
    type: Literal["error"] = "error"


class SummarizationStartedEvent(BaseModel):
    """Auto-summarization process started"""

    messages_to_summarize: int
    recent_messages_kept: int
    context_usage_before: float
    type: Literal["summarization_started"] = "summarization_started"


class SummarizationFinishedEvent(BaseModel):
    """Auto-summarization process completed"""

    summary: str
    messages_summarized: int
    messages_after: int
    context_usage_after: float
    type: Literal["summarization_finished"] = "summarization_finished"


class ResponseCompleteEvent(BaseModel):
    """Agent response completed with context information"""

    message_count: int
    conversation_messages: int
    estimated_tokens: int
    context_limit: int
    usage_percentage: float
    approaching_limit: bool
    type: Literal["response_complete"] = "response_complete"


# Union type for all agent events
AgentEvent = (
    AgentTextEvent
    | ToolStartedEvent
    | ToolProgressEvent
    | ToolFinishedEvent
    | UserInputRequestEvent
    | AgentErrorEvent
    | SummarizationStartedEvent
    | SummarizationFinishedEvent
    | ResponseCompleteEvent
)
