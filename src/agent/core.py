"""
Core agent implementation
"""

import json
import time
from typing import List, Optional, Iterator
from pydantic import BaseModel

from .llm import LLM, SupportedModel, Message as LLMMessage
from .tools import ToolRegistry, ToolExecutionError
from .streaming import StreamingParser, TextEvent, ToolCallEvent, InvalidToolCallEvent
from .config import AgentConfig
from .types import (
    AgentMessage,
    Message,
    SummarizationContent,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolCallError,
    ToolCallFinished,
    ToolResult,
    UserMessage,
)
from .agent_events import (
    AgentEvent,
    AgentTextEvent,
    ToolStartedEvent,
    ToolFinishedEvent,
    AgentErrorEvent,
    SummarizationStartedEvent,
    SummarizationFinishedEvent,
    ResponseCompleteEvent,
)
import logging

logger = logging.getLogger(__name__)


class ContextInfo(BaseModel):
    """Information about the agent's current context usage"""

    message_count: int
    conversation_messages: int
    estimated_tokens: int
    context_limit: int
    usage_percentage: float
    approaching_limit: bool


class Agent:
    """Main agent class that coordinates between LLM and tools"""

    def __init__(
        self,
        config: AgentConfig,
        model: SupportedModel,
        llm: LLM,
    ):
        self.llm = llm
        self.model = model
        self.context_window = llm.models[model].context_window
        self.auto_summarize_threshold = int(self.context_window * 0.75)  # 75% threshold

        # Use provided configuration
        self.config = config

        # Initialize tools based on configuration
        self.tools = ToolRegistry(self, self.config.tools)

        # Dual conversation histories
        self.conversation_history: List[Message] = []  # Complete history for user view
        self.llm_conversation_history: List[Message] = (
            []
        )  # Optimized history for LLM (with summaries)

        # Generic agent state - configurations can extend this
        self.state = (
            self.config.default_state.copy() if self.config.default_state else {}
        )

        # System prompt will be built dynamically with current state

    def get_state(self, key: Optional[str] = None):
        """Get agent state"""
        parts = key.split(".") if key else []
        current_state = self.state
        for part in parts:
            if isinstance(current_state, dict) and part in current_state:
                current_state = current_state[part]
            else:
                return None
        return current_state

    def set_state(self, key: Optional[str], value):
        """Set a specific state value"""
        if key is None:
            # If no key is provided, set the entire state
            self.state = value
            return

        parts = key.split(".")
        current_state = self.state

        # Navigate to the parent of the target
        for part in parts[:-1]:
            if part not in current_state:
                current_state[part] = {}
            current_state = current_state[part]

        # Set the final value
        final_key = parts[-1]
        current_state[final_key] = value

    def get_conversation_history(self) -> List[Message]:
        """Get the current conversation history"""
        return self.conversation_history.copy()

    def get_llm_conversation_history(
        self,
        include_tools: bool,
        iteration_info: tuple,
    ) -> List[LLMMessage]:
        """Get the conversation history formatted for LLM"""
        messages = [
            LLMMessage(
                role="system",
                content=self._build_system_prompt(
                    include_tools=include_tools,
                    iteration_info=iteration_info,
                ),
            )
        ]
        # Use the optimized LLM history (which may include summaries)
        for msg in self.llm_conversation_history:
            for llm_msg in message_to_llm_messages(msg):
                messages.append(llm_msg)
        return messages

    def get_context_info(self) -> ContextInfo:
        """Get information about current context usage"""
        # Build current message list
        messages = self.get_llm_conversation_history(
            include_tools=True, iteration_info=(1, self.config.max_iterations)
        )

        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        total_chars = sum(len(msg.content) for msg in messages)
        estimated_tokens = total_chars // 4

        return ContextInfo(
            message_count=len(messages),
            conversation_messages=len(
                self.conversation_history
            ),  # User-visible message count
            estimated_tokens=estimated_tokens,
            context_limit=self.context_window,
            usage_percentage=(estimated_tokens / self.context_window) * 100,
            approaching_limit=estimated_tokens > self.auto_summarize_threshold,
        )

    def _build_system_prompt(self, include_tools: bool, iteration_info: tuple) -> str:
        """Build the system prompt using configuration with current state"""
        if include_tools:
            tools_start = time.time()
            tools_desc = self.tools.get_tools_description()
            tools_time = time.time() - tools_start
            logger.debug(f"Tools description generation took: {tools_time:.3f}s")
        else:
            tools_desc = ""

        return self.config.build_prompt(tools_desc, self.state, iteration_info)

    def chat_stream(self, user_input: str) -> Iterator[AgentEvent]:
        """Streaming chat interface that yields typed events"""
        start_time = time.time()

        # Add user message to both histories
        user_message = UserMessage(content=[TextContent(text=user_input)])
        self.conversation_history.append(user_message)
        self.llm_conversation_history.append(user_message)

        # Check if we need auto-summarization before processing
        context_info = self.get_context_info()
        keep_recent = 6  # Default retention size
        if (
            context_info.approaching_limit
            and len(self.llm_conversation_history) > keep_recent
        ):
            # Perform auto-summarization with event emission
            for event in self._auto_summarize_with_events(keep_recent):
                yield event

        # Continue generating responses with iteration limit
        max_iterations = self.config.max_iterations
        for iteration in range(1, max_iterations + 1):
            iteration_start = time.time()

            # Build system prompt based on iteration (no tools on final iteration)
            is_final_iteration = iteration == max_iterations

            prompt_start = time.time()
            messages = self.get_llm_conversation_history(
                include_tools=not is_final_iteration,
                iteration_info=(iteration, max_iterations),
            )
            # logger.info(
            #     f"System prompt for iteration {iteration}/{max_iterations}:\n{messages[0].content}"
            # )
            prompt_time = time.time() - prompt_start

            logger.debug(
                f"Sending {len(messages)} messages to LLM (iteration {iteration}/{max_iterations})"
            )
            logger.debug(f"Prompt building took: {prompt_time:.3f}s")

            # Get streaming LLM response
            llm_start = time.time()
            stream = self.llm.chat_streaming(self.model, messages)
            llm_init_time = time.time() - llm_start

            logger.debug(f"LLM initialization took: {llm_init_time:.3f}s")

            # Initialize streaming parser
            parser = StreamingParser(debug=False)
            collected_response = ""
            tool_events: List[ToolCallEvent] = []

            # Process the stream
            first_chunk = True
            first_chunk_time = None

            for chunk in stream:
                if first_chunk:
                    first_chunk_time = time.time() - llm_start
                    logger.debug(f"Time to first chunk: {first_chunk_time:.3f}s")
                    first_chunk = False
                chunk_content = chunk["message"]["content"]

                # Parse chunk for events
                events = list(parser.parse_chunk(chunk_content))

                for event in events:
                    if isinstance(event, TextEvent):
                        # Only collect actual text content, not tool syntax
                        collected_response += event.delta
                        yield AgentTextEvent(content=event.delta)
                    elif isinstance(event, ToolCallEvent):
                        # Store tool event for later execution (only if not final iteration)
                        if not is_final_iteration:
                            tool_events.append(event)
                    elif isinstance(event, InvalidToolCallEvent):
                        # System-level error (malformed tool call)
                        logger.error(
                            f"Invalid tool call detected: {event.error} in {event.tool_name} ({event.id})"
                        )
                        logger.error(f"Tool call content: {event.raw_content}")
                        yield AgentErrorEvent(
                            message=f"Invalid tool call: {event.error}",
                            tool_name=event.tool_name,
                            tool_id=event.id,
                        )

            # Finalize parser
            final_events = list(parser.finalize())
            for event in final_events:
                if isinstance(event, TextEvent):
                    collected_response += event.delta
                    yield AgentTextEvent(content=event.delta)
                elif isinstance(event, InvalidToolCallEvent):
                    logger.error(
                        f"Invalid tool call detected: {event.error} in {event.tool_name} ({event.id})"
                    )
                    logger.error(f"Tool call content: {event.raw_content}")
                    yield AgentErrorEvent(
                        message=f"Invalid tool call: {event.error}",
                        tool_name=event.tool_name,
                        tool_id=event.id,
                    )

            # If no tools or final iteration, we're done
            if not tool_events or is_final_iteration:
                agent_message = AgentMessage(
                    content=[TextContent(text=collected_response)], tool_calls=[]
                )
                self.conversation_history.append(agent_message)
                self.llm_conversation_history.append(agent_message)
                break

            # Execute tools and continue to next iteration
            tool_results = []
            finished_tool_calls = []

            for tool_event in tool_events:
                # Check if tool exists before emitting any events
                if not self.tools.has_tool(tool_event.tool_name):
                    # Agent made a mistake - inform it via tool_results for next iteration
                    error_msg = f"Tool '{tool_event.tool_name}' not found"
                    tool_results.append(
                        f"{tool_event.tool_name} ({tool_event.id}): {error_msg}"
                    )

                    # Create error tool call
                    finished_tool_calls.append(
                        ToolCallFinished(
                            tool_name=tool_event.tool_name,
                            tool_id=tool_event.id,
                            parameters=tool_event.parameters,
                            result=ToolCallError(type="error", error=error_msg),
                        )
                    )
                    yield AgentErrorEvent(
                        message=error_msg,
                        tool_name=tool_event.tool_name,
                        tool_id=tool_event.id,
                    )
                    continue  # No tool events for nonexistent tools

                # Tool exists - proceed with normal execution flow
                yield ToolStartedEvent(
                    tool_name=tool_event.tool_name,
                    tool_id=tool_event.id,
                    parameters=tool_event.parameters,
                )

            for tool_event in tool_events:
                try:
                    # Check if tool exists again (in case of previous errors)
                    if not self.tools.has_tool(tool_event.tool_name):
                        # If tool was not found, skip execution
                        continue

                    # Execute tool with progress callback
                    def progress_callback(data):
                        # Send progress events for streaming
                        pass  # For now, we can ignore progress in the core

                    logger.info(
                        f"Executing tool {tool_event.tool_name} with ID {tool_event.id}\n{json.dumps(tool_event.parameters, indent=2)}"
                    )
                    result = self.tools.execute(
                        tool_event.tool_name,
                        tool_event.id,
                        tool_event.parameters,
                        progress_callback,
                    )
                    tool_results.append(
                        f"{tool_event.tool_name} ({tool_event.id}): {result}"
                    )

                    # Create successful tool call
                    finished_tool_calls.append(
                        ToolCallFinished(
                            tool_name=tool_event.tool_name,
                            tool_id=tool_event.id,
                            parameters=tool_event.parameters,
                            result=result,
                        )
                    )

                    # Signal successful tool execution
                    yield ToolFinishedEvent(
                        tool_id=tool_event.id,
                        result=result,
                    )
                except ToolExecutionError as e:
                    logger.error(
                        f"Error executing tool {tool_event.tool_name} ({tool_event.id}): {str(e)}"
                    )

                    # Tool exists but failed during execution
                    tool_results.append(
                        f"{tool_event.tool_name} ({tool_event.id}): {str(e)}"
                    )

                    # Create error tool call
                    finished_tool_calls.append(
                        ToolCallFinished(
                            tool_name=tool_event.tool_name,
                            tool_id=tool_event.id,
                            parameters=tool_event.parameters,
                            result=ToolCallError(type="error", error=str(e)),
                        )
                    )

                    yield ToolFinishedEvent(
                        tool_id=tool_event.id, result=ToolCallError(error=str(e))
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error executing tool {tool_event.tool_name} ({tool_event.id}): {str(e)}"
                    )

                    # Unexpected system error
                    error_msg = (
                        f"System error executing {tool_event.tool_name}: {str(e)}"
                    )
                    tool_results.append(
                        f"{tool_event.tool_name} ({tool_event.id}): {error_msg}"
                    )

                    # Create system error tool call
                    finished_tool_calls.append(
                        ToolCallFinished(
                            tool_name=tool_event.tool_name,
                            tool_id=tool_event.id,
                            parameters=tool_event.parameters,
                            result=ToolCallError(type="error", error=error_msg),
                        )
                    )

                    yield AgentErrorEvent(
                        message=str(e),
                        tool_name=tool_event.tool_name,
                        tool_id=tool_event.id,
                    )

            # Store the agent message with tool calls in both histories
            agent_message = AgentMessage(
                content=[TextContent(text=collected_response)],
                tool_calls=finished_tool_calls,
            )
            self.conversation_history.append(agent_message)
            self.llm_conversation_history.append(agent_message)

        # Emit response complete event with context info
        context_info = self.get_context_info()
        yield ResponseCompleteEvent(
            message_count=context_info.message_count,
            conversation_messages=context_info.conversation_messages,
            estimated_tokens=context_info.estimated_tokens,
            context_limit=context_info.context_limit,
            usage_percentage=context_info.usage_percentage,
            approaching_limit=context_info.approaching_limit,
        )

        # Performance logging
        total_time = time.time() - start_time
        logger.debug(f"Total chat_stream time: {total_time:.3f}s")

    def reset_conversation(self):
        """Reset both conversation histories"""
        self.conversation_history = []
        self.llm_conversation_history = []

    def _auto_summarize_with_events(self, keep_recent: int = 6) -> Iterator[AgentEvent]:
        """Auto-summarize with event emission for streaming clients"""
        # Calculate what we're about to do - work on LLM history only
        old_messages = self.llm_conversation_history[:-keep_recent]
        recent_messages = self.llm_conversation_history[-keep_recent:]
        context_before = self.get_context_info()

        # Emit started event
        yield SummarizationStartedEvent(
            messages_to_summarize=len(old_messages),
            recent_messages_kept=len(recent_messages),
            context_usage_before=context_before.usage_percentage,
        )

        # Convert old messages to text format for summarization
        conversation_text = ""
        for msg in old_messages:
            # Convert message to text representation
            content_parts = []
            for item in msg.content:
                if isinstance(item, TextContent):
                    content_parts.append(item.text)
                elif isinstance(item, SummarizationContent):
                    content_parts.append(f"[Summary: {item.summary}]")
            content = "\n".join(content_parts)

            if isinstance(msg, AgentMessage) and msg.tool_calls:
                # Include tool calls in the text representation
                tool_calls_str = format_tool_calls(msg.tool_calls)
                if tool_calls_str:
                    content += "\n\n" + tool_calls_str

                # Include tool results if available
                tool_results_str = format_tool_results(
                    [
                        tool_call
                        for tool_call in msg.tool_calls
                        if isinstance(tool_call, ToolCallFinished)
                    ]
                )
                if tool_results_str:
                    content += "\n\n" + tool_results_str

            conversation_text += f"{msg.role.upper()}: {content}\n\n"

        # Get config-specific summarization prompt with current state context
        summary_system_prompt = self.config.get_summarization_system_prompt()

        # Add current state context to system prompt (critical for roleplay)
        state_context = self.config._build_state_info(self.state)
        if state_context:
            summary_system_prompt += f"\n\nCURRENT STATE CONTEXT:\n{state_context}"

        # Build summarization request as single user message
        summary_task = self.config.get_summarization_prompt()
        user_request = f"{summary_task}\n\n{conversation_text.strip()}"

        summary_request = [
            LLMMessage(role="system", content=summary_system_prompt),
            LLMMessage(role="user", content=user_request),
        ]

        summary_response = self.llm.chat_complete(self.model, summary_request)
        assert summary_response, "LLM response is empty"

        # Create summary as agent message to maintain proper alternation
        summary_message = SystemMessage(
            content=[
                SummarizationContent(
                    type="summarization",
                    title=f"Conversation Summary ({len(old_messages)} messages)",
                    summary=summary_response,
                    messages_summarized=len(old_messages),
                    context_usage_before=context_before.usage_percentage,
                    context_usage_after=0.0,  # Will be updated later
                )
            ]
        )

        self.llm_conversation_history = [summary_message] + recent_messages

        context_after = self.get_context_info()

        # Add structured summarization notification to user history
        # Find the position where summarization occurred in user history
        user_summary_index = len(self.conversation_history) - len(recent_messages)

        # Create structured summarization content that matches frontend expectations
        summarization_content = SummarizationContent(
            type="summarization",
            title=f"✅ Summarized {len(old_messages)} messages. Context usage: {context_before.usage_percentage:.1f}% → {context_after.usage_percentage:.1f}%",
            summary=summary_response,
            messages_summarized=len(old_messages),
            context_usage_before=context_before.usage_percentage,
            context_usage_after=context_after.usage_percentage,
        )

        summary_message = SystemMessage(
            content=[summarization_content],
        )

        # Insert notification at the right position to maintain chronological order
        self.conversation_history.insert(user_summary_index, summary_message)
        self.llm_conversation_history = [summary_message] + recent_messages

        # Emit finished event
        yield SummarizationFinishedEvent(
            summary=summary_response,
            messages_summarized=len(old_messages),
            messages_after=len(self.llm_conversation_history),
            context_usage_after=context_after.usage_percentage,
        )


def message_to_llm_messages(message: Message) -> Iterator[LLMMessage]:
    """Convert internal Message to LLMMessage format"""

    # Extract text from content list
    content_parts = []
    for item in message.content:
        if isinstance(item, TextContent):
            content_parts.append(item.text)
        elif isinstance(item, SummarizationContent):
            content_parts.append(f"[Summary: {item.summary}]")

    content = "\n".join(content_parts)
    tool_results_str = ""

    if isinstance(message, AgentMessage) and message.tool_calls:
        # Add tool call information if available
        tool_calls_str = format_tool_calls(message.tool_calls)
        tool_results_str = format_tool_results(
            [
                tool_call
                for tool_call in message.tool_calls
                if isinstance(tool_call, ToolCallFinished)
            ]
        )

        if tool_calls_str:
            content += "\n\n" + tool_calls_str

    yield LLMMessage(role=message.role, content=content)

    if tool_results_str:
        yield LLMMessage(role="user", content=tool_results_str)


def format_tool_calls(tool_calls: List[ToolCall]) -> str:
    """Format a list of tool calls into a string representation"""
    formatted_calls = []
    for tool_call in tool_calls:
        formatted_calls.append(
            f"TOOL_CALL: {tool_call.tool_name} ({tool_call.tool_id})\n{json.dumps(tool_call.parameters, indent=2)}"
        )
    return "\n\n".join(formatted_calls)


def format_tool_results(tool_results: List[ToolCallFinished]) -> str:
    """Format a list of tool call results into a string representation"""
    formatted_results = []
    for result in tool_results:
        if result.result.type == "success":
            # Use simple llm_feedback instead of full JSON
            feedback = result.result.llm_feedback
        else:
            # For errors, include the error message
            feedback = f"Error: {result.result.error}"

        formatted_results.append(
            f"TOOL_RESULT: {result.tool_name} ({result.tool_id}): {feedback}"
        )
    return "\n\n".join(formatted_results)
