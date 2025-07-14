from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# Tool content types (similar to message content types)
class TextToolContent(BaseModel):
    """Text tool content"""

    type: Literal["text"] = "text"
    text: str


class ImageGenerationToolContent(BaseModel):
    """Image generation tool content with LLM optimization metadata"""

    type: Literal["image_generated"] = "image_generated"
    prompt: str  # Final optimized prompt used
    image_path: str  # File system path to the generated image
    image_url: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None

    # New optimization metadata
    original_description: Optional[str] = None  # Original natural description
    optimization_confidence: Optional[float] = None  # LLM confidence in optimization
    camera_angle: Optional[str] = None  # Camera angle chosen
    viewpoint: Optional[str] = None  # Viewpoint chosen
    optimization_notes: Optional[str] = None  # Notes about optimization choices


# Union of all tool content types
ToolContent = Union[TextToolContent, ImageGenerationToolContent]


# Proper tagged union for tool call results
class ToolCallSuccess(BaseModel):
    """Successful tool execution result"""

    type: Literal["success"] = "success"
    content: ToolContent
    llm_feedback: str = (
        "Tool executed successfully"  # Simple feedback for LLM iterations
    )


class ToolCallError(BaseModel):
    """Failed tool execution result"""

    type: Literal["error"] = "error"
    error: str


# Tagged union for tool results
ToolResult = Union[ToolCallSuccess, ToolCallError]


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
    result: ToolResult


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
    """Message from the user"""

    role: Literal["user"] = "user"
    content: UserContent


class AgentMessage(BaseModel):
    """Message from the agent"""

    role: Literal["assistant"] = "assistant"
    content: AgentContent
    tool_calls: List[ToolCall]


class SystemMessage(BaseModel):
    """System message for summarization or other system-level content"""

    role: Literal["system"] = "system"
    content: SystemContent


Message = UserMessage | AgentMessage | SystemMessage
