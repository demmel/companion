"""
Core agent implementation
"""

import json
import time
from typing import List, Optional, Iterator
from pydantic import BaseModel

from .llm import LLMClient, Message as LLMMessage
from .tools import ToolRegistry, ToolExecutionError
from .streaming import StreamingParser, TextEvent, ToolCallEvent, InvalidToolCallEvent
from .config import AgentConfig
from .message import (
    AgentMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolCallFinished,
    ToolCallResult,
    ToolCallResultType,
    UserMessage,
)
from .agent_events import (
    AgentEvent,
    AgentTextEvent,
    ToolStartedEvent,
    ToolFinishedEvent,
    ToolResultType,
    AgentErrorEvent,
    SummarizationStartedEvent,
    SummarizationFinishedEvent,
    ResponseCompleteEvent,
)


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
        model: str = "llama3.1:8b",
        verbose: bool = False,
    ):
        self.llm = LLMClient(model=model)
        self.context_window = self.llm.context_window
        self.auto_summarize_threshold = int(self.context_window * 0.75)  # 75% threshold

        # Use provided configuration
        self.config = config

        # Initialize tools based on configuration
        self.tools = ToolRegistry(self, self.config.tools)
        self.verbose = verbose

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
        """Get agent state or specific state value"""
        if key is None:
            return self.state
        return self.state.get(key)

    def set_state(self, key: str, value):
        """Set a specific state value"""
        self.state[key] = value

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
            if self.verbose:
                print(f"[PERF] Tools description generation took: {tools_time:.3f}s")
        else:
            tools_desc = ""

        return self.config.build_prompt(tools_desc, self.state, iteration_info)

    def chat_stream(self, user_input: str) -> Iterator[AgentEvent]:
        """Streaming chat interface that yields typed events"""
        start_time = time.time()

        # Add user message to both histories
        user_message = UserMessage(content=user_input)
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
            prompt_time = time.time() - prompt_start

            if self.verbose:
                print(
                    f"Sending {len(messages)} messages to LLM (iteration {iteration}/{max_iterations})"
                )
                print(f"[PERF] Prompt building took: {prompt_time:.3f}s")

            # Get streaming LLM response
            llm_start = time.time()
            stream = self.llm.chat(messages)
            llm_init_time = time.time() - llm_start

            if self.verbose:
                print(f"[PERF] LLM initialization took: {llm_init_time:.3f}s")

            # Initialize streaming parser
            parser = StreamingParser(debug=self.verbose)
            collected_response = ""
            tool_events: List[ToolCallEvent] = []

            # Process the stream
            first_chunk = True
            first_chunk_time = None

            for chunk in stream:
                if first_chunk:
                    first_chunk_time = time.time() - llm_start
                    if self.verbose:
                        print(f"[PERF] Time to first chunk: {first_chunk_time:.3f}s")
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
                    yield AgentErrorEvent(
                        message=f"Invalid tool call: {event.error}",
                        tool_name=event.tool_name,
                        tool_id=event.id,
                    )

            # If no tools or final iteration, we're done
            if not tool_events or is_final_iteration:
                agent_message = AgentMessage(content=collected_response, tool_calls=[])
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
                            result=ToolCallResult(
                                type=ToolCallResultType.ERROR, content=error_msg
                            ),
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

                try:
                    # Execute tool
                    result = self.tools.execute(
                        tool_event.tool_name, tool_event.parameters
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
                            result=ToolCallResult(
                                type=ToolCallResultType.SUCCESS, content=result
                            ),
                        )
                    )

                    # Signal successful tool execution
                    yield ToolFinishedEvent(
                        tool_id=tool_event.id,
                        result_type=ToolResultType.SUCCESS,
                        result=result,
                    )
                except ToolExecutionError as e:
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
                            result=ToolCallResult(
                                type=ToolCallResultType.ERROR, content=str(e)
                            ),
                        )
                    )

                    yield ToolFinishedEvent(
                        tool_id=tool_event.id,
                        result_type=ToolResultType.ERROR,
                        result=str(e),
                    )
                except Exception as e:
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
                            result=ToolCallResult(
                                type=ToolCallResultType.ERROR, content=error_msg
                            ),
                        )
                    )

                    yield AgentErrorEvent(
                        message=str(e),
                        tool_name=tool_event.tool_name,
                        tool_id=tool_event.id,
                    )

            # Store the agent message with tool calls in both histories
            agent_message = AgentMessage(
                content=collected_response, tool_calls=finished_tool_calls
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
        if self.verbose:
            print(f"[PERF] Total chat_stream time: {total_time:.3f}s")

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
            content = msg.content
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
        summary_task = self.config.get_summarization_prompt(len(old_messages))
        user_request = f"Please summarize the following conversation:\n\n{conversation_text}\n{summary_task}"

        summary_request = [
            LLMMessage(role="system", content=summary_system_prompt),
            LLMMessage(role="user", content=user_request),
        ]

        summary_response = self.llm.chat_complete(summary_request)

        # Create summary as agent message to maintain proper alternation

        summary_message = AgentMessage(
            content=f"[Summary of {len(old_messages)} previous messages: {summary_response}]",
            tool_calls=[],
        )

        # Check recent messages for proper alternation
        # If recent messages start with an agent message, we might break alternation
        if recent_messages and recent_messages[0].role == "assistant":
            # Insert a system message to maintain flow
            separator_message = SystemMessage(
                content="[Previous conversation summarized above]"
            )
            self.llm_conversation_history = [
                summary_message,
                separator_message,
            ] + recent_messages
        else:
            # Safe to concatenate directly
            self.llm_conversation_history = [summary_message] + recent_messages

        # Add summarization notification to user history (they should see what happened)
        # Find the position where summarization occurred in user history
        user_summary_index = len(self.conversation_history) - len(recent_messages)
        summary_notification = AgentMessage(
            content=f"📝 [Summarized {len(old_messages)} older messages to manage context - conversation continues below]",
            tool_calls=[],
        )

        # Insert notification at the right position to maintain chronological order
        self.conversation_history.insert(user_summary_index, summary_notification)

        context_after = self.get_context_info()

        # Emit finished event
        yield SummarizationFinishedEvent(
            summary=summary_response,
            messages_summarized=len(old_messages),
            messages_after=len(self.llm_conversation_history),
            context_usage_after=context_after.usage_percentage,
        )

    def summarize_and_trim_context(self, keep_recent: int = 6) -> dict:
        """Summarize older conversation and keep only recent messages"""
        # Split conversation into old (to summarize) and recent (to keep)
        old_messages = self.conversation_history[:-keep_recent]
        recent_messages = self.conversation_history[-keep_recent:]

        # Create conversation text for summarization
        conversation_text = ""
        for msg in old_messages:
            conversation_text += f"{msg.role.upper()}: {msg.content}\n\n"

        # Request summarization from LLM using config-specific prompt
        summary_prompt = self.config.get_summarization_prompt(conversation_text)

        summary_response = self.llm.chat_complete(
            [
                Message(
                    role="system",
                    content="You are a conversation summarizer. Provide clear, structured summaries.",
                ),
                Message(role="user", content=summary_prompt),
            ]
        )

        # Replace old conversation with summary
        summary_message = Message(
            role="system", content=f"CONVERSATION SUMMARY: {summary_response}"
        )
        self.conversation_history = [summary_message] + recent_messages

        # System prompt will automatically include summary in next build

        return {
            "messages_before": len(old_messages) + len(recent_messages),
            "messages_after": len(self.conversation_history),
            "summarized_count": len(old_messages),
        }


def message_to_llm_messages(message: Message) -> Iterator[LLMMessage]:
    """Convert internal Message to LLMMessage format"""

    content = message.content
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
        formatted_results.append(
            f"TOOL_RESULT: {result.tool_name} ({result.tool_id})\n{result.model_dump_json(indent=2)}"
        )
    return "\n\n".join(formatted_results)
