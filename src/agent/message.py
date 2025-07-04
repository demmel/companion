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


class SummarizationContent(BaseModel):
    """Structured content for summarization system messages"""
    type: Literal["summarization"] = "summarization"
    title: str
    summary: str
    messages_summarized: int
    context_usage_before: float
    context_usage_after: float


class TextContent(BaseModel):
    """Structured content for text system messages"""
    type: Literal["text"] = "text"
    text: str


# Union type for system message content (can be structured or string for backward compatibility)
SystemContent = SummarizationContent | TextContent | str


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: SystemContent


Message = UserMessage | AgentMessage | SystemMessage
