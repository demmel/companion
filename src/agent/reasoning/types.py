"""
Core types for the reasoning system to avoid circular imports
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class AnalysisType(Enum):
    """Type of analysis being performed"""

    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"


class ProposedToolCall(BaseModel):
    """A tool that should be called based on the reasoning"""

    call_id: str = Field(description="Unique identifier for the tool call")
    tool_name: str = Field(description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(description="Parameters to pass to the tool")
    reasoning: str = Field(
        description="Specific strategic explanation of HOW this tool will be useful in future conversations, WHY it matters for character development, and WHAT opportunities it creates for future storylines. Avoid generic explanations like 'important to remember' - be specific about the strategic value."
    )


class ReasoningResult(BaseModel):
    """What we learned from reasoning about the text"""

    understanding: str = Field(
        description="What we understood about the text and what the person is trying to achieve"
    )
    situational_awareness: str = Field(
        description="What this means for the conversation and how it connects to previous context"
    )
    emotional_context: str = Field(
        description="Emotions, intentions, or needs present in the message"
    )
    key_information: List[str] = Field(
        description="Important details that should be remembered"
    )
    proposed_tools: List[ProposedToolCall] = Field(
        description="ONLY tools for genuinely NEW information that passes both 'first-time revelation' and 'strategic importance' tests. Group related details into coherent memories rather than fragmenting them. Each tool must have specific strategic reasoning about future value.",
        default=[],
    )
    follow_up_opportunities: List[str] = Field(
        description="Future conversation opportunities this creates"
    )
    should_end_turn: bool = Field(
        description="Whether this analysis suggests ending the conversation turn",
        default=False,
    )
