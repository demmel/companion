from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

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


class TextContent(BaseModel):
    """Structured content for text messages"""
    type: Literal["text"] = "text"
    text: str


class SummarizationContent(BaseModel):
    """Structured content for summarization system messages"""
    type: Literal["summarization"] = "summarization"
    title: str
    summary: str
    messages_summarized: int
    context_usage_before: float
    context_usage_after: float


# Role-specific content item types
UserContentItem = TextContent
AgentContentItem = TextContent
SystemContentItem = Union[SummarizationContent, TextContent]

# Role-specific content as lists
UserContent = List[UserContentItem]
AgentContent = List[AgentContentItem]
SystemContent = List[SystemContentItem]


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: UserContent


class AgentMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: AgentContent
    tool_calls: List[ToolCall]


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: SystemContent


Message = UserMessage | AgentMessage | SystemMessage
