"""
Core agent implementation
"""

from typing import List, Optional, Iterator
from pydantic import BaseModel

from .llm import LLMClient, Message
from .tools import ToolRegistry, ToolExecutionError
from .streaming import StreamingParser, TextEvent, ToolCallEvent, InvalidToolCallEvent
from .config import AgentConfig
from .agent_events import (
    AgentEvent,
    AgentTextEvent,
    ToolStartedEvent,
    ToolProgressEvent,
    ToolFinishedEvent,
    ToolResultType,
    AgentErrorEvent,
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
        self.conversation_history: List[Message] = []

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

    def get_context_info(self) -> ContextInfo:
        """Get information about current context usage"""
        # Build current message list
        current_system_prompt = self._build_system_prompt()
        messages = [Message(role="system", content=current_system_prompt)]
        messages.extend(self.conversation_history)

        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        total_chars = sum(len(msg.content) for msg in messages)
        estimated_tokens = total_chars // 4

        return ContextInfo(
            message_count=len(messages),
            conversation_messages=len(self.conversation_history),
            estimated_tokens=estimated_tokens,
            context_limit=self.context_window,
            usage_percentage=(estimated_tokens / self.context_window) * 100,
            approaching_limit=estimated_tokens > self.auto_summarize_threshold,
        )

    def _build_system_prompt(
        self, include_tools: bool = True, iteration_info: tuple = None
    ) -> str:
        """Build the system prompt using configuration with current state"""
        tools_desc = self.tools.get_tools_description() if include_tools else ""
        return self.config.build_prompt(tools_desc, self.state, iteration_info)


    def chat_stream(self, user_input: str) -> Iterator[AgentEvent]:
        """Streaming chat interface that yields typed events"""
        # Add user message to history
        self.conversation_history.append(Message(role="user", content=user_input))

        # Continue generating responses with iteration limit
        max_iterations = self.config.max_iterations
        for iteration in range(1, max_iterations + 1):
            # Build system prompt based on iteration (no tools on final iteration)
            is_final_iteration = iteration == max_iterations
            current_system_prompt = self._build_system_prompt(
                include_tools=not is_final_iteration,
                iteration_info=(iteration, max_iterations),
            )
            messages = [Message(role="system", content=current_system_prompt)]
            messages.extend(self.conversation_history)

            if self.verbose:
                print(
                    f"Sending {len(messages)} messages to LLM (iteration {iteration}/{max_iterations})"
                )

            # Get streaming LLM response
            stream = self.llm.chat(messages)

            # Initialize streaming parser
            parser = StreamingParser(debug=self.verbose)
            collected_response = ""
            tool_events = []

            # Process the stream
            for chunk in stream:
                chunk_content = chunk["message"]["content"]
                collected_response += chunk_content

                # Debug: Show raw chunk content
                if self.verbose and chunk_content:
                    print(f"[DEBUG] Raw chunk: {repr(chunk_content)}")

                # Parse chunk for events
                events = list(parser.parse_chunk(chunk_content))

                for event in events:
                    if isinstance(event, TextEvent):
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
                    yield AgentTextEvent(content=event.delta)
                elif isinstance(event, InvalidToolCallEvent):
                    yield AgentErrorEvent(
                        message=f"Invalid tool call: {event.error}",
                        tool_name=event.tool_name,
                        tool_id=event.id,
                    )

            # Add response to history
            self.conversation_history.append(
                Message(role="assistant", content=collected_response)
            )

            # If no tools or final iteration, we're done
            if not tool_events or is_final_iteration:
                break

            # Execute tools and continue to next iteration
            tool_results = []
            for tool_event in tool_events:
                # Check if tool exists before emitting any events
                if not self.tools.has_tool(tool_event.tool_name):
                    # Agent made a mistake - inform it via tool_results for next iteration
                    error_msg = f"Tool '{tool_event.tool_name}' not found"
                    tool_results.append(f"{tool_event.tool_name} ({tool_event.id}): {error_msg}")
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

                    # Signal successful tool execution
                    yield ToolFinishedEvent(
                        tool_id=tool_event.id,
                        result_type=ToolResultType.SUCCESS,
                        result=result
                    )
                except ToolExecutionError as e:
                    # Tool exists but failed during execution
                    tool_results.append(f"{tool_event.tool_name} ({tool_event.id}): {str(e)}")
                    yield ToolFinishedEvent(
                        tool_id=tool_event.id,
                        result_type=ToolResultType.ERROR,
                        result=str(e)
                    )
                except Exception as e:
                    # Unexpected system error
                    error_msg = f"System error executing {tool_event.tool_name}: {str(e)}"
                    tool_results.append(f"{tool_event.tool_name} ({tool_event.id}): {error_msg}")
                    yield AgentErrorEvent(
                        message=str(e),
                        tool_name=tool_event.tool_name,
                        tool_id=tool_event.id,
                    )

            # Add tool results to history for next iteration
            combined_results = "; ".join(tool_results)
            self.conversation_history.append(
                Message(role="user", content=f"Tool results: {combined_results}")
            )




    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []

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
