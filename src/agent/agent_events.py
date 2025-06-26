"""
Agent-level streaming events for the chat_stream API
"""

from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


class AgentEventType(Enum):
    """Types of agent streaming events"""

    TEXT = "text"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESS = "tool_progress"
    TOOL_FINISHED = "tool_finished"
    USER_INPUT_REQUEST = "user_input_request"  # Placeholder for future implementation
    ERROR = "error"


class ToolResultType(Enum):
    """Types of tool execution results"""
    
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class AgentTextEvent:
    """Text content from agent response"""

    content: str
    type: AgentEventType = AgentEventType.TEXT


@dataclass
class ToolStartedEvent:
    """Tool execution started"""

    tool_name: str
    tool_id: str
    parameters: Dict[str, Any]
    type: AgentEventType = AgentEventType.TOOL_STARTED


@dataclass
class ToolProgressEvent:
    """Progress update from tool execution"""

    tool_id: str
    data: str
    progress: Optional[float] = None  # 0.0 to 1.0
    type: AgentEventType = AgentEventType.TOOL_PROGRESS


@dataclass
class ToolFinishedEvent:
    """Tool execution completed"""

    tool_id: str
    result_type: ToolResultType
    result: str  # actual result for success, error message for error
    type: AgentEventType = AgentEventType.TOOL_FINISHED


@dataclass
class UserInputRequestEvent:
    """Placeholder for interactive tools that need user input"""

    # TODO: Define structure when we implement interactive tools
    pass


@dataclass
class AgentErrorEvent:
    """Error in agent processing"""

    message: str
    tool_name: str = ""
    tool_id: str = ""
    type: AgentEventType = AgentEventType.ERROR


# Union type for all agent events
AgentEvent = (
    AgentTextEvent
    | ToolStartedEvent
    | ToolProgressEvent
    | ToolFinishedEvent
    | UserInputRequestEvent
    | AgentErrorEvent
)
