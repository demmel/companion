from collections.abc import Iterator
import json
import logging
from typing import List
from agent.agent_events import (
    AgentErrorEvent,
    AgentEvent,
    AgentTextEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
)
from agent.conversation_history import ConversationHistory
from agent.llm import LLM, SupportedModel
from agent.llm import Message as LLMMessage
from agent.reasoning.analyze import analyze_conversation_turn
from agent.reasoning.prompts import build_chloe_response_prompt
from agent.state import ChloeState, build_chloe_state_description
from agent.state_analysis import analyze_thoughts_for_state_updates, apply_state_updates
from agent.types import (
    AgentMessage,
    Message,
    ThoughtContent,
    ToolCallContent,
    TextContent,
    ToolCallError,
    ToolCallFinished,
    UserMessage,
)
from agent.reasoning.types import AnalysisType
from agent.tools import ToolExecutionError, ToolRegistry

logger = logging.getLogger(__name__)


def _build_image_description(appearance: str, environment: str) -> str:
    """Build an image generation prompt from appearance and environment descriptions"""

    # Simply combine the descriptions without bias
    if appearance and environment:
        return f"{appearance} in {environment}"
    elif appearance:
        return appearance
    elif environment:
        return f"A person in {environment}"
    else:
        return "A person"


def run_reasoning_loop(
    history: ConversationHistory,
    user_input: str,
    tools: ToolRegistry,
    llm: LLM,
    model: SupportedModel,
    chloe_state: ChloeState,
) -> Iterator[AgentEvent]:

    # Build Chloe's state description for reasoning context
    state_description = build_chloe_state_description(chloe_state)

    # Generate Chloe's thoughts about the user input
    thoughts = analyze_conversation_turn(
        user_input,
        AnalysisType.USER_INPUT,
        history.get_summarized_history(),
        tools,
        llm,
        model,
        True,  # Include thoughts for reasoning continuity
        state_description,
    )

    # Add user message to history
    user_message = UserMessage(content=[TextContent(text=user_input)])
    history.add_message(user_message)

    # Emit thoughts as thought event
    yield AgentTextEvent(
        content=thoughts,
        is_thought=True,
    )

    # Create agent message with thoughts
    current_message = AgentMessage(
        role="assistant",
        content=[ThoughtContent(text=thoughts)],
        tool_calls=[],
    )

    # Analyze thoughts for state updates
    try:
        state_updates = analyze_thoughts_for_state_updates(
            thoughts, chloe_state, llm, model
        )
        # Apply updates to Chloe's state
        chloe_state = apply_state_updates(chloe_state, state_updates)
        current_message.content.append(
            ThoughtContent(
                text=f"State updates:\n{state_updates.model_dump_json(indent=2)}"
            )
        )
        # Emit state updates as JSON for debugging
        yield AgentTextEvent(
            content=f"State updates: {state_updates.model_dump_json(indent=2)}",
            is_thought=True,
        )

        # Generate image if appearance or environment changed
        if state_updates.appearance or state_updates.environment:
            # Build image description from updated state
            image_description = _build_image_description(
                chloe_state.current_appearance, chloe_state.current_environment
            )

            # Generate image using the existing tool with streaming events
            if tools.has_tool("generate_image"):
                try:
                    # Emit tool started event
                    yield ToolStartedEvent(
                        tool_name="generate_image",
                        tool_id="auto_image_gen",
                        parameters={"description": image_description},
                    )

                    # Execute image generation via tool registry
                    def progress_callback(progress):
                        pass  # Could emit progress events here if needed

                    image_result = tools.execute(
                        "generate_image",
                        "auto_image_gen",
                        {"description": image_description},
                        progress_callback,
                    )

                    # Add tool call to current message for UI display
                    from agent.types import ToolCallContent

                    tool_call = ToolCallContent(
                        tool_name="generate_image",
                        call_id="auto_image_gen",
                        parameters={"description": image_description},
                        result=image_result,
                    )
                    current_message.content.append(tool_call)
                    current_message.tool_calls.append(
                        ToolCallFinished(
                            tool_name="generate_image",
                            tool_id="auto_image_gen",
                            parameters={"description": image_description},
                            result=image_result,
                        )
                    )

                    # Emit tool finished event
                    yield ToolFinishedEvent(
                        tool_id="auto_image_gen", result=image_result
                    )

                except Exception as e:
                    logger.warning(f"Auto image generation failed: {e}")

    except Exception as e:
        logger.warning(f"State analysis failed: {e}")
        # Continue without state updates

    # Generate response based on thoughts and updated state
    response = ""
    for text in _generate_response(
        history.get_summarized_history(),
        current_message,
        llm,
        model,
        chloe_state,
    ):
        response += text
        yield AgentTextEvent(content=text, is_thought=False)

    # Add response text to the message
    current_message.content.append(TextContent(text=response.strip()))

    # Add complete message to history
    history.add_message(current_message)


def _populate_legacy_tool_calls(message: AgentMessage):
    """Populate the legacy tool_calls field from inline ToolCallContent for backward compatibility"""
    from agent.types import ToolCallFinished

    tool_calls = []
    for content in message.content:
        if isinstance(content, ToolCallContent):
            # Convert ToolCallContent to legacy ToolCallFinished format
            # Use the actual result if available, otherwise create a placeholder
            result = (
                content.result
                if content.result is not None
                else ToolCallError(type="error", error="No result")
            )
            tool_call = ToolCallFinished(
                tool_name=content.tool_name,
                tool_id=content.call_id,
                parameters=content.parameters,
                result=result,
            )
            tool_calls.append(tool_call)

    message.tool_calls = tool_calls


def _generate_response(
    conversation: List[Message],
    current_message: AgentMessage,
    llm: LLM,
    model: SupportedModel,
    chloe_state: ChloeState,
) -> Iterator[str]:
    # Serialize conversation history for context
    from .analyze import _serialize_conversation_context

    context_str = _serialize_conversation_context(conversation, include_thoughts=True)

    # Extract reasoning and tool results from current message
    reasoning_parts = []
    tool_results = []

    for content in current_message.content:
        if isinstance(content, ThoughtContent):
            # Use unstructured thoughts text directly
            reasoning_parts.append(content.text)
        elif isinstance(content, ToolCallContent) and content.result:
            tool_results.append(f"{content.tool_name}: {content.result}")

    reasoning_context = (
        "\n".join(reasoning_parts) if reasoning_parts else "No thoughts available"
    )
    tools_context = "\n".join(tool_results) if tool_results else "No tools executed"

    # Build Chloe's state description for response context
    state_description = build_chloe_state_description(chloe_state)

    # Use Chloe-specific response generation prompt (first-person direct)
    direct_prompt = build_chloe_response_prompt(
        context_str, reasoning_context, tools_context, state_description
    )

    # Use direct streaming generation instead of chat
    for content in llm.generate_streaming(model=model, prompt=direct_prompt):
        yield content.response
