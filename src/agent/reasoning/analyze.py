"""
Basic reasoning experiment - understand text and decide what to do about it
"""

from typing import List

from agent.llm import LLM, SupportedModel
from agent.state import State
from agent.tools import BaseTool, ToolRegistry
from agent.structured_llm import direct_structured_llm_call
from .types import AnalysisType, ReasoningResult
from .prompts import (
    build_understanding_prompt,
    build_reflection_prompt,
)
from agent.types import (
    Message,
    AgentMessage,
    ToolCallContent,
    ThoughtContent,
    UserMessage,
    SystemMessage,
    TextContent,
    SummarizationContent,
)


def analyze_conversation_turn(
    text: str,
    analysis_type: AnalysisType,
    conversation_context: List[Message],
    tool_registry: ToolRegistry,
    llm: LLM,
    model: SupportedModel,
    state: State,
    include_thoughts: bool = True,
) -> str:
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

    # Build agent-specific direct prompts
    if analysis_type == AnalysisType.USER_INPUT:
        direct_prompt = build_understanding_prompt(
            text, context_text, tools_description, state
        )
    elif analysis_type == AnalysisType.AGENT_RESPONSE:
        direct_prompt = build_reflection_prompt(
            text, context_text, tools_description, state
        )
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    # Use direct generation for unstructured thoughts
    thoughts = llm.generate_complete(
        model=model,
        prompt=direct_prompt,
        temperature=0.6,
    )

    return thoughts


def _serialize_conversation_context(
    messages: List[Message], include_thoughts: bool = True
) -> str:
    """Convert conversation messages to markdown format preserving chronological order"""
    if not messages:
        return "No previous conversation history."

    lines = ["## Conversation History\n"]

    # Remove truncation - include all messages for full context
    for i, msg in enumerate(messages):
        # Add 2 blank lines between messages (except the first one)
        if i > 0:
            lines.append("")
            lines.append("")
        # Handle different message types with proper union type checking
        if isinstance(msg, UserMessage):
            # Extract user name from context or default to "User"
            user_name = "David"  # TODO: Extract actual user name if available

            content_parts = []
            for content_item in msg.content:
                if isinstance(content_item, TextContent):
                    content_parts.append(content_item.text)

            if content_parts:
                content_text = " ".join(content_parts)
                lines.append(f"### {user_name}")
                lines.append(content_text)

        elif isinstance(msg, AgentMessage):
            lines.append("### Me")

            # Process content items in chronological order, detecting type changes
            prev_type = None
            current_section = []

            for content_item in msg.content:
                current_type = type(content_item).__name__

                # If content type changed, flush previous section
                if prev_type and prev_type != current_type:
                    if current_section:
                        lines.extend(current_section)
                        lines.append("")  # 1 blank line between sections
                        current_section = []

                if isinstance(content_item, ThoughtContent) and include_thoughts:
                    thought_parts = ["**My Thoughts:**"]
                    thought_parts.append(content_item.text)
                    current_section = ["\n".join(thought_parts)]

                elif isinstance(content_item, TextContent):
                    if current_section:
                        # Already in text section - just append content
                        current_section.append(content_item.text)
                    else:
                        # Start new text section with separate header and content
                        current_section = [
                            "**What I said:**",
                            "```",
                            content_item.text,
                            "```",
                        ]

                elif isinstance(content_item, ToolCallContent):
                    # Format tool call with parameters and results
                    tool_name = content_item.tool_name
                    params = content_item.parameters
                    result = content_item.result

                    action_lines = [
                        f"**Action:** {tool_name.title().replace('_', ' ')}"
                    ]

                    # Show parameters as bullet list
                    if params:
                        for key, value in params.items():
                            action_lines.append(f"- {key}: {value}")

                    # Show result
                    if result:
                        if result.type == "success":
                            # Handle different content types
                            content = result.content
                            if content.type == "text":
                                result_text = content.text
                            elif content.type == "image_generated":
                                # Show interesting optimization details
                                details = [f"Optimized prompt: '{content.prompt}'"]
                                if content.optimization_notes:
                                    details.append(
                                        f"Notes: {content.optimization_notes}"
                                    )
                                if content.camera_angle:
                                    details.append(f"Camera: {content.camera_angle}")
                                if content.viewpoint:
                                    details.append(f"Viewpoint: {content.viewpoint}")
                                result_text = " | ".join(details)
                            else:
                                result_text = str(content)
                            action_lines.append(f"- ✅ Result: {result_text}")
                        elif result.type == "error":
                            action_lines.append(f"- ❌ Error: {result.error}")

                    current_section = ["\n".join(action_lines)]

                prev_type = current_type

            # Flush final section
            if current_section:
                lines.extend(current_section)

        elif isinstance(msg, SystemMessage):
            content_parts = []
            for content_item in msg.content:
                if isinstance(content_item, TextContent):
                    content_parts.append(content_item.text)
                elif isinstance(content_item, SummarizationContent):
                    content_parts.append(
                        f"**Conversation Summary:**\n\n{content_item.summary}"
                    )

            if content_parts:
                content_text = "\n\n".join(
                    content_parts
                )  # Use double newlines to separate content blocks
                lines.append("### System")
                lines.append(f"*{content_text}*")

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
