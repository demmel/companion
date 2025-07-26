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
from agent.reasoning.chloe_prompts import build_chloe_response_prompt
from agent.chloe_state import ChloeState, build_chloe_state_description
from agent.types import (
    AgentMessage,
    Message,
    ThoughtContent,
    ToolCallContent,
    TextContent,
    ToolCallError,
    UserMessage,
)
from agent.reasoning.types import AnalysisType
from agent.tools import ToolExecutionError, ToolRegistry

logger = logging.getLogger(__name__)


def run_reasoning_loop(
    history: ConversationHistory,
    user_input: str,
    tools: ToolRegistry,
    llm: LLM,
    model: SupportedModel,
    chloe_state: ChloeState,
) -> Iterator[AgentEvent]:

    current_text = user_input
    current_analysis_type = AnalysisType.USER_INPUT
    current_message = AgentMessage(
        role="assistant",
        content=[],
        tool_calls=[],
    )

    while True:
        # Build Chloe's state description for reasoning context
        state_description = build_chloe_state_description(chloe_state)

        reasoning_result = analyze_conversation_turn(
            current_text,
            current_analysis_type,
            history.get_summarized_history(),
            tools,
            llm,
            model,
            False,
            state_description,
        )

        if current_analysis_type == AnalysisType.USER_INPUT:
            user_message = UserMessage(content=[TextContent(text=current_text)])
            history.add_message(user_message)
        else:
            # After reasoning, we can append the text content to the current message which is in history
            current_message.content.append(TextContent(text=current_text))
            current_message = AgentMessage(
                role="assistant",
                content=[],
                tool_calls=[],
            )

        current_message.content.append(
            ThoughtContent(
                text=reasoning_result.model_dump_json(indent=2),
                reasoning=reasoning_result,
            )
        )

        # Emit reasoning as thought event
        yield AgentTextEvent(
            content=reasoning_result.model_dump_json(indent=2),
            is_thought=True,
        )

        for tool_event in reasoning_result.proposed_tools:
            # Check if tool exists before emitting any events
            if not tools.has_tool(tool_event.tool_name):
                # Agent made a mistake - inform it via tool_results for next iteration
                error_msg = f"Tool '{tool_event.tool_name}' not found"

                # Create error tool call
                current_message.content.append(
                    ToolCallContent(
                        tool_name=tool_event.tool_name,
                        call_id=tool_event.call_id,
                        parameters=tool_event.parameters,
                        reasoning=tool_event.reasoning,
                        result=ToolCallError(type="error", error=error_msg),
                    )
                )

                yield AgentErrorEvent(
                    message=error_msg,
                    tool_name=tool_event.tool_name,
                    tool_id=tool_event.call_id,
                )
                continue  # No tool events for nonexistent tools

            # Tool exists - proceed with normal execution flow
            yield ToolStartedEvent(
                tool_name=tool_event.tool_name,
                tool_id=tool_event.call_id,
                parameters=tool_event.parameters,
            )

        for tool_event in reasoning_result.proposed_tools:
            try:
                # Check if tool exists again (in case of previous errors)
                if not tools.has_tool(tool_event.tool_name):
                    # If tool was not found, skip execution
                    continue

                # Execute tool with progress callback
                def progress_callback(data):
                    # Send progress events for streaming
                    pass  # For now, we can ignore progress in the core

                logger.info(
                    f"Executing tool {tool_event.tool_name} with ID {tool_event.call_id}\n{json.dumps(tool_event.parameters, indent=2)}"
                )
                result = tools.execute(
                    tool_event.tool_name,
                    tool_event.call_id,
                    tool_event.parameters,
                    progress_callback,
                )

                # Create successful tool call
                current_message.content.append(
                    ToolCallContent(
                        tool_name=tool_event.tool_name,
                        call_id=tool_event.call_id,
                        parameters=tool_event.parameters,
                        reasoning=tool_event.reasoning,
                        result=result,
                    )
                )

                # Signal successful tool execution
                yield ToolFinishedEvent(
                    tool_id=tool_event.call_id,
                    result=result,
                )
            except ToolExecutionError as e:
                logger.error(
                    f"Error executing tool {tool_event.tool_name} ({tool_event.call_id}): {str(e)}"
                )

                # Create error tool call
                current_message.content.append(
                    ToolCallContent(
                        tool_name=tool_event.tool_name,
                        call_id=tool_event.call_id,
                        parameters=tool_event.parameters,
                        reasoning=tool_event.reasoning,
                        result=ToolCallError(type="error", error=str(e)),
                    )
                )

                yield ToolFinishedEvent(
                    tool_id=tool_event.call_id, result=ToolCallError(error=str(e))
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error executing tool {tool_event.tool_name} ({tool_event.call_id}): {str(e)}"
                )

                # Unexpected system error
                error_msg = f"System error executing {tool_event.tool_name}: {str(e)}"

                # Create system error tool call
                current_message.content.append(
                    ToolCallContent(
                        tool_name=tool_event.tool_name,
                        call_id=tool_event.call_id,
                        parameters=tool_event.parameters,
                        reasoning=tool_event.reasoning,
                        result=ToolCallError(type="error", error=error_msg),
                    )
                )

                yield AgentErrorEvent(
                    message=str(e),
                    tool_name=tool_event.tool_name,
                    tool_id=tool_event.call_id,
                )

        if reasoning_result.should_end_turn:
            _populate_legacy_tool_calls(current_message)
            history.add_message(current_message)
            break

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

        response = response.strip()
        current_text = response
        current_analysis_type = AnalysisType.AGENT_RESPONSE

        _populate_legacy_tool_calls(current_message)
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

    context_str = _serialize_conversation_context(conversation, include_thoughts=False)

    # Extract reasoning and tool results from current message
    reasoning_parts = []
    tool_results = []

    for content in current_message.content:
        if isinstance(content, ThoughtContent):
            reasoning_parts.append(f"Understanding: {content.reasoning.understanding}")
            reasoning_parts.append(
                f"Situational awareness: {content.reasoning.situational_awareness}"
            )
            reasoning_parts.append(
                f"Emotional context: {content.reasoning.emotional_context}"
            )
        elif isinstance(content, ToolCallContent) and content.result:
            tool_results.append(f"{content.tool_name}: {content.result}")

    reasoning_context = (
        "\n".join(reasoning_parts) if reasoning_parts else "No reasoning available"
    )
    tools_context = "\n".join(tool_results) if tool_results else "No tools executed"

    # Build Chloe's state description for response context
    state_description = build_chloe_state_description(chloe_state)

    # Use Chloe-specific response generation prompt
    system_prompt, user_prompt = build_chloe_response_prompt(
        context_str, reasoning_context, tools_context, state_description
    )

    for content in llm.chat_streaming(
        model=model,
        messages=[
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ],
    ):
        yield content["message"]["content"]
