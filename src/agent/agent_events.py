"""
Agent-level streaming events for the chat_stream API
"""

from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel


class AgentEventType(Enum):
    """Types of agent streaming events"""

    TEXT = "text"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESS = "tool_progress"
    TOOL_FINISHED = "tool_finished"
    USER_INPUT_REQUEST = "user_input_request"  # Placeholder for future implementation
    ERROR = "error"
    RESPONSE_COMPLETE = "response_complete"


class ToolResultType(Enum):
    """Types of tool execution results"""

    SUCCESS = "success"
    ERROR = "error"


class AgentTextEvent(BaseModel):
    """Text content from agent response"""

    content: str
    type: AgentEventType = AgentEventType.TEXT


class ToolStartedEvent(BaseModel):
    """Tool execution started"""

    tool_name: str
    tool_id: str
    parameters: Dict[str, Any]
    type: AgentEventType = AgentEventType.TOOL_STARTED


class ToolProgressEvent(BaseModel):
    """Progress update from tool execution"""

    tool_id: str
    data: str
    progress: Optional[float] = None  # 0.0 to 1.0
    type: AgentEventType = AgentEventType.TOOL_PROGRESS


class ToolFinishedEvent(BaseModel):
    """Tool execution completed"""

    tool_id: str
    result_type: ToolResultType
    result: str  # actual result for success, error message for error
    type: AgentEventType = AgentEventType.TOOL_FINISHED


class UserInputRequestEvent(BaseModel):
    """Placeholder for interactive tools that need user input"""

    # TODO: Define structure when we implement interactive tools
    pass


class AgentErrorEvent(BaseModel):
    """Error in agent processing"""

    message: str
    tool_name: str = ""
    tool_id: str = ""
    type: AgentEventType = AgentEventType.ERROR


class ResponseCompleteEvent(BaseModel):
    """Agent response completed with context information"""

    message_count: int
    conversation_messages: int
    estimated_tokens: int
    context_limit: int
    usage_percentage: float
    approaching_limit: bool
    type: AgentEventType = AgentEventType.RESPONSE_COMPLETE


# Union type for all agent events
AgentEvent = (
    AgentTextEvent
    | ToolStartedEvent
    | ToolProgressEvent
    | ToolFinishedEvent
    | UserInputRequestEvent
    | AgentErrorEvent
    | ResponseCompleteEvent
)
