"""
Basic reasoning experiment - understand text and decide what to do about it
"""

from typing import List

from agent.llm import LLM, SupportedModel
from agent.tools import BaseTool, ToolRegistry
from agent.structured_llm import structured_llm_call
from .types import AnalysisType, ReasoningResult
from agent.types import (
    Message,
    AgentMessage,
    ToolCallContent,
    ThoughtContent,
    UserMessage,
    SystemMessage,
    TextContent,
)


def analyze_conversation_turn(
    text: str,
    analysis_type: AnalysisType,
    conversation_context: List[Message],
    tool_registry: ToolRegistry,
    llm: LLM,
    model: SupportedModel,
    include_thoughts: bool = True,
) -> ReasoningResult:
    """
    Analyze a conversation turn - understand it and decide what to do

    This function provides different reasoning approaches based on what we're analyzing:
    - USER_INPUT: Understanding what the user is trying to communicate/achieve
    - AGENT_RESPONSE: Self-reflection on whether our response is complete
    """

    # Serialize conversation context for prompt
    context_text = _serialize_conversation_context(
        conversation_context, include_thoughts
    )

    # Get tool descriptions from registry
    tools_description = tool_registry.get_tools_description()

    # Build analysis-type-specific prompts
    if analysis_type == AnalysisType.USER_INPUT:
        system_prompt, user_prompt = _build_user_input_analysis_prompt(
            text, context_text, tools_description
        )
    elif analysis_type == AnalysisType.AGENT_RESPONSE:
        system_prompt, user_prompt = _build_agent_response_analysis_prompt(
            text, context_text, tools_description
        )
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    # Use structured LLM call to get reliable output
    return structured_llm_call(
        system_prompt=system_prompt,
        user_input=user_prompt,
        response_model=ReasoningResult,
        model=model,
        llm=llm,
        temperature=0.3,
    )


def _serialize_conversation_context(
    messages: List[Message], include_thoughts: bool = True
) -> str:
    """Convert List[ReasoningMessage] to formatted string for prompt inclusion"""
    if not messages:
        return "No previous conversation history."

    lines = []
    for msg in messages[-10:]:  # Only include last 10 messages to avoid context bloat
        role_display = msg.role.upper()

        # Handle different message types with proper union type checking
        if isinstance(msg, UserMessage):
            content_parts = []
            for content_item in msg.content:
                if isinstance(content_item, TextContent):
                    content_parts.append(content_item.text)
            content_text = " ".join(content_parts) if content_parts else "[No content]"

        elif isinstance(msg, AgentMessage):
            content_parts = []
            for content_item in msg.content:
                if isinstance(content_item, TextContent):
                    content_parts.append(content_item.text)
                elif isinstance(content_item, ThoughtContent) and include_thoughts:
                    content_parts.append(
                        f"[REASONING: {content_item.reasoning.understanding}]"
                    )
                elif isinstance(content_item, ToolCallContent):
                    content_parts.append(f"TOOL_CALL: {content_item.tool_name}")
            content_text = " ".join(content_parts) if content_parts else "[No content]"

        elif isinstance(msg, SystemMessage):
            content_parts = []
            for content_item in msg.content:
                if isinstance(content_item, TextContent):
                    content_parts.append(content_item.text)
            content_text = (
                " ".join(content_parts) if content_parts else "[System message]"
            )

        else:
            content_text = "[Unknown message type]"

        lines.append(f"{role_display}: {content_text}")

    return "\n".join(lines)


def _build_user_input_analysis_prompt(
    text: str, context_text: str, tools_description: str
) -> tuple[str, str]:
    """Build prompts for analyzing user input"""

    system_prompt = """You are a reasoning agent that deeply analyzes user messages to understand their meaning and determine appropriate responses.

Your task is to analyze what the user is trying to communicate, what they need, and what actions would be most helpful in response.

Focus on understanding FIRST, then planning actions. Only suggest tools when they would genuinely enhance the conversation or capture truly NEW information."""

    user_prompt = f"""
================================================================================
                               CONVERSATION CONTEXT  
================================================================================
{context_text}

================================================================================
                            USER MESSAGE TO ANALYZE
================================================================================
"{text}"

================================================================================
                                AVAILABLE TOOLS
================================================================================
{tools_description}

**REASONING TASK:**
Provide deep analysis of what the user is trying to achieve:

1. **Understanding**: What is the user trying to communicate?
   - What are they asking for or trying to achieve?
   - What emotions, intentions, or needs are present?
   - What key information are they providing?

2. **Situational Awareness**: What does this mean for our conversation?
   - How does this connect to our previous interactions?
   - What context or background is relevant?
   - What should I remember about this moment?

3. **Response Planning**: How should I respond?
   - What tools (if any) would enhance my response?
   - For memory tools: Is this information genuinely NEW or already established in our conversation?
   - What specific strategic value would each tool provide for THIS character in THIS moment?
   - Should I end my turn here or continue with more actions?

CRITICAL: Only propose tools for information that is NOT already established in the conversation context. Explain WHY each tool matters strategically, not just that it's "important to remember."

Provide structured output with your analysis."""

    return system_prompt, user_prompt


def _build_agent_response_analysis_prompt(
    text: str, context_text: str, tools_description: str
) -> tuple[str, str]:
    """Build prompts for analyzing agent's own response"""

    system_prompt = """You are a self-reflection agent that analyzes your own responses to determine if they are complete and appropriate.

Your task is to evaluate whether your response fully addresses what was needed, and whether you should continue with additional actions or end your turn.

Focus on natural conversation flow and avoiding over-acting. Most responses should end the turn unless there's a compelling reason to continue."""

    user_prompt = f"""
================================================================================
                               CONVERSATION CONTEXT
================================================================================
{context_text}

================================================================================
                            MY RESPONSE TO ANALYZE
================================================================================
"{text}"

================================================================================
                                AVAILABLE TOOLS
================================================================================
{tools_description}

**SELF-REFLECTION TASK:**
Evaluate the completeness and appropriateness of my response:

1. **Understanding**: How well did I address what was needed?
   - Did I answer the user's question or request fully?
   - Are there important aspects I missed or should clarify?
   - Is my response appropriate for the emotional context?

2. **Situational Awareness**: What is the current state of our interaction?
   - Does my response create a natural pause for user input?
   - Does it end with a question, dramatic revelation, or clear handoff?
   - Have I provided sufficient detail without overwhelming?

3. **Continuation Planning**: Should I continue or end my turn?
   - For memory tools: Compare my response against the conversation context - which specific details appear for the FIRST TIME in our conversation history?
   - For scene/action tools: Would additional actions enhance the moment or feel like over-acting?
   - Does this response naturally invite user reaction, or does it need more from me?

**TERMINATION GUIDELINES:**
- Set should_end_turn to TRUE if: Response creates natural pause, ends with question/revelation, or provides complete answer
- Set should_end_turn to FALSE if: Response feels incomplete, user needs more context, or important new information requires memory storage

**MEMORY ANALYSIS PROCESS:**
1. **Read my response carefully** - What factual details does it contain?
2. **Scan the conversation context** - Have any of these details been mentioned before in previous messages?
3. **Group related new information** - Combine connected details into coherent stories/facts rather than fragmenting them
4. **Assess strategic importance** - Will this specific information be useful for future character interactions or plot development?
5. **Only propose memory tools** for details that pass BOTH the "first-time revelation" AND "strategic importance" tests

**GRANULARITY GUIDELINES:**
- Group related details into single coherent memories (e.g., "painter story" not 4 separate details)
- Focus on information that will meaningfully impact future interactions
- Avoid storing minor details that don't advance character or plot development

**REASONING QUALITY REQUIREMENTS:**
- Explain specific strategic value: HOW will this information be useful in future conversations?
- Connect to character development: WHY does this matter for Elena's character arc?
- Consider plot implications: WHAT opportunities does this create for future storylines?

**Example:**
- GOOD: "Remember the tragic painter story (Montmartre 1889, vampire love affair, abrupt ending) because this establishes Elena's pattern of difficult romantic choices and gives the user a concrete historical example to reference when exploring Elena's past relationships"
- BAD: "Remember this because it provides significant context about Elena's history"

Provide structured output with your analysis."""

    return system_prompt, user_prompt


if __name__ == "__main__":
    # Simple test when we're ready
    pass
