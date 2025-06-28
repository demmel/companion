from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ToolCallResultType(str, Enum):
    """Enum for tool call result types"""

    SUCCESS = "success"
    ERROR = "error"


class ToolCallResult(BaseModel):
    """Result of a tool call"""

    type: ToolCallResultType
    content: str


class ToolCallBase(BaseModel):
    """Base class for tool call events"""

    tool_name: str
    tool_id: str
    parameters: Dict[str, Any]


class ToolCallRunning(ToolCallBase):
    """Event when a tool call starts"""

    type: Literal["started"] = "started"
    progress: Optional[Any] = None  # Optional progress data


class ToolCallFinished(ToolCallBase):
    """Event when a tool call finishes"""

    type: Literal["finished"] = "finished"
    result: ToolCallResult


ToolCall = ToolCallRunning | ToolCallFinished


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class AgentMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: List[ToolCall]


Message = UserMessage | AgentMessage
