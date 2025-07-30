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
from agent.reasoning.prompts import build_response_prompt
from agent.state import State, build_agent_state_description
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


def _resolve_removal_ids_for_streaming(state_updates, state):
    """Resolve removal IDs to readable content for streaming display"""
    import json

    # Start with the original state updates (JSON-safe serialization)
    updates_dict = state_updates.model_dump(mode='json')

    # Resolve memory IDs to content
    if updates_dict.get("memory_ids_to_forget"):
        resolved_memories = []
        for memory_id in updates_dict["memory_ids_to_forget"]:
            memory = next((m for m in state.memories if m.id == memory_id), None)
            if memory:
                resolved_memories.append(f"{memory.content} (ID: {memory_id})")
            else:
                resolved_memories.append(f"Unknown memory (ID: {memory_id})")
        updates_dict["memory_ids_to_forget"] = resolved_memories

    # Resolve goal IDs to content
    if updates_dict.get("goal_ids_to_remove"):
        resolved_goals = []
        for goal_id in updates_dict["goal_ids_to_remove"]:
            goal = next((g for g in state.current_goals if g.id == goal_id), None)
            if goal:
                resolved_goals.append(f"{goal.content} (ID: {goal_id})")
            else:
                resolved_goals.append(f"Unknown goal (ID: {goal_id})")
        updates_dict["goal_ids_to_remove"] = resolved_goals

    # Resolve desire IDs to content
    if updates_dict.get("desire_ids_to_remove"):
        resolved_desires = []
        for desire_id in updates_dict["desire_ids_to_remove"]:
            desire = next(
                (d for d in state.immediate_desires if d.id == desire_id), None
            )
            if desire:
                resolved_desires.append(f"{desire.content} (ID: {desire_id})")
            else:
                resolved_desires.append(f"Unknown desire (ID: {desire_id})")
        updates_dict["desire_ids_to_remove"] = resolved_desires

    # Resolve value IDs to content
    if updates_dict.get("value_ids_to_remove"):
        resolved_values = []
        for value_id in updates_dict["value_ids_to_remove"]:
            value = next((v for v in state.core_values if v.id == value_id), None)
            if value:
                resolved_values.append(f"{value.content} (ID: {value_id})")
            else:
                resolved_values.append(f"Unknown value (ID: {value_id})")
        updates_dict["value_ids_to_remove"] = resolved_values

    return json.dumps(updates_dict, indent=2)


def _build_image_description(
    appearance: str, environment: str, agent_name: str, llm: LLM, model: SupportedModel
) -> str:
    """Use LLM to convert first-person descriptions and combine into image generation prompt"""

    # Handle empty descriptions
    if not appearance and not environment:
        return f"{agent_name}"
    elif not appearance:
        appearance = f"I am {agent_name}"
    elif not environment:
        environment = "I'm in a simple setting"

    prompt = f"""Create an image generation description by converting these first-person descriptions to third-person and combining them naturally:

Character name: {agent_name}
Appearance: "{appearance}"
Environment: "{environment}"

Convert to third-person and combine into a coherent image description suitable for AI image generation. Focus on visual details that can be rendered. Keep it concise but descriptive.

Image description:"""

    response = llm.generate_complete(model, prompt)
    return response.strip()


def run_reasoning_loop(
    history: ConversationHistory,
    user_input: str,
    tools: ToolRegistry,
    llm: LLM,
    model: SupportedModel,
    state: State,
) -> Iterator[AgentEvent]:

    # Build the agent's state description for reasoning context
    state_description = build_agent_state_description(state)

    # Generate the agent's thoughts about the user input
    thoughts = analyze_conversation_turn(
        user_input,
        AnalysisType.USER_INPUT,
        history.get_summarized_history(),
        tools,
        llm,
        model,
        state,
        True,  # Include thoughts for reasoning continuity
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
        state_updates = analyze_thoughts_for_state_updates(thoughts, state, llm, model)

        # Resolve removal IDs to content for readable streaming
        readable_updates = _resolve_removal_ids_for_streaming(state_updates, state)

        # Emit readable state updates for debugging
        yield AgentTextEvent(
            content=f"State updates: {readable_updates}",
            is_thought=True,
        )

        # Apply updates to the agent's state
        state = apply_state_updates(state, state_updates)
        current_message.content.append(
            ThoughtContent(text=f"State updates:\n{readable_updates}")
        )

        # Generate image if appearance or environment changed
        if state_updates.appearance or state_updates.environment:
            # Build image description from updated state
            image_description = _build_image_description(
                state.current_appearance,
                state.current_environment,
                state.name,
                llm,
                model,
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
        state,
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
    state: State,
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

    # Use agent-specific response generation prompt (first-person direct)
    direct_prompt = build_response_prompt(
        context_str, reasoning_context, tools_context, state
    )

    # Use direct streaming generation instead of chat
    for content in llm.generate_streaming(model=model, prompt=direct_prompt):
        yield content.response
