"""
Core agent implementation
"""

from typing import List, Optional, Iterator

from .llm import LLMClient, Message
from .tools import ToolRegistry
from .streaming import StreamingParser, TextEvent, ToolCallEvent, InvalidToolCallEvent
from .agent_events import (
    AgentEvent,
    AgentTextEvent,
    ToolStartedEvent,
    ToolProgressEvent,
    ToolFinishedEvent,
    AgentErrorEvent,
)


class Agent:
    """Main agent class that coordinates between LLM and tools"""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        verbose: bool = False,
        config_name: str = "roleplay",
    ):
        self.llm = LLMClient(model=model)
        self.context_window = self.llm.context_window
        self.auto_summarize_threshold = int(self.context_window * 0.75)  # 75% threshold

        # Load configuration
        from .config import get_config

        self.config = get_config(config_name)

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

    def get_context_info(self) -> dict:
        """Get information about current context usage"""
        # Build current message list
        current_system_prompt = self._build_system_prompt()
        messages = [Message(role="system", content=current_system_prompt)]
        messages.extend(self.conversation_history)

        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        total_chars = sum(len(msg.content) for msg in messages)
        estimated_tokens = total_chars // 4

        return {
            "message_count": len(messages),
            "conversation_messages": len(self.conversation_history),
            "estimated_tokens": estimated_tokens,
            "context_limit": self.context_window,
            "usage_percentage": (estimated_tokens / self.context_window) * 100,
            "approaching_limit": estimated_tokens > self.auto_summarize_threshold,
        }

    def _build_system_prompt(
        self, include_tools: bool = True, iteration_info: tuple = None
    ) -> str:
        """Build the system prompt using configuration with current state"""
        tools_desc = self.tools.get_tools_description() if include_tools else ""
        return self.config.build_prompt(tools_desc, self.state, iteration_info)

    def chat(self, user_input: str) -> str:
        """Main chat interface"""
        # Add user message to history
        self.conversation_history.append(Message(role="user", content=user_input))

        # Build messages with current system prompt (includes up-to-date state)
        current_system_prompt = self._build_system_prompt()
        messages = [Message(role="system", content=current_system_prompt)]
        messages.extend(self.conversation_history)

        if self.verbose:
            print(f"Sending {len(messages)} messages to LLM")

        # Get LLM response
        response = self.llm.chat(messages)

        # Check if the response contains tool usage
        tool_result = self._handle_tool_usage(response)

        if tool_result:
            # If tool was used, get follow-up response
            self.conversation_history.append(
                Message(role="assistant", content=response)
            )
            self.conversation_history.append(
                Message(role="system", content=f"Tool result: {tool_result}")
            )

            # Remove tool calls from the original response and get clean dialogue
            clean_response = self._remove_tool_calls_from_response(response)

            # If there's still meaningful content after removing tool calls, use it
            if clean_response.strip():
                self.conversation_history.append(
                    Message(role="assistant", content=clean_response)
                )
                return self._format_response(clean_response)
            else:
                # If no meaningful content, generate a follow-up response
                follow_up_message = Message(
                    role="user",
                    content="Please provide a natural character response. Do not mention tools or their results.",
                )

                messages = [Message(role="system", content=current_system_prompt)]
                messages.extend(self.conversation_history)
                messages.append(follow_up_message)

                final_response = self.llm.chat(messages)
                self.conversation_history.append(
                    Message(role="assistant", content=final_response)
                )

                return self._format_response(final_response)
        else:
            # No tool usage, just add response to history
            self.conversation_history.append(
                Message(role="assistant", content=response)
            )
            return self._format_response(response)

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
            stream = self.llm.chat(messages, stream=True)

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
                # Signal tool execution starting
                yield ToolStartedEvent(
                    tool_name=tool_event.tool_name,
                    tool_id=tool_event.id,
                    parameters=tool_event.parameters,
                )

                try:
                    # Execute tool - result could be success or failure message
                    result = self.tools.execute(
                        tool_event.tool_name, tool_event.parameters
                    )
                    tool_results.append(
                        f"{tool_event.tool_name} ({tool_event.id}): {result}"
                    )

                    # Signal tool execution completed
                    yield ToolFinishedEvent(tool_id=tool_event.id, result=result)
                except Exception as e:
                    # System-level error executing tool
                    error_msg = (
                        f"System error executing {tool_event.tool_name}: {str(e)}"
                    )
                    tool_results.append(error_msg)

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

    def _handle_tool_usage(self, response: str) -> Optional[str]:
        """Parse response for JSON tool usage and execute if found"""
        import json
        import re

        # Look for tool calls in multiple formats (handle mixed formats)
        # Format 1: TOOL_CALL: toolname { "param": "value" }
        # Format 2: TOOL_CALL: toolname\n{ "param": "value" }
        # Format 3: TOOL_CALL: toolname {"param": "value", "param2": "value2"}

        # Use a single comprehensive pattern to avoid duplicates
        tool_call_pattern = r"TOOL_CALL:\s*(\w+)\s*(\{[^}]*\})"
        tool_matches = re.findall(tool_call_pattern, response, re.DOTALL)

        if tool_matches:
            results = []
            for tool_name, param_str in tool_matches:
                param_str = param_str.strip()
                parameters = {}

                # Try to parse as JSON first
                try:
                    if param_str.startswith("{") and param_str.endswith("}"):
                        parameters = json.loads(param_str)
                    else:
                        # Clean up the parameter string for JSON parsing
                        if not param_str.startswith("{"):
                            param_str = "{" + param_str
                        if not param_str.endswith("}"):
                            param_str = param_str + "}"
                        parameters = json.loads(param_str)
                except json.JSONDecodeError:
                    # Fall back to manual parsing
                    for line in param_str.split("\n"):
                        line = line.strip()
                        if (
                            ":" in line
                            and not line.startswith("{")
                            and not line.endswith("}")
                        ):
                            key, value = line.split(":", 1)
                            key = key.strip().strip('"')
                            value = value.strip().strip(",").strip('"')
                            parameters[key] = value

                if self.verbose:
                    print(f"Found tool call: {tool_name} with params: {parameters}")

                # Execute the tool
                result = self.tools.execute(tool_name, parameters)
                if self.verbose:
                    print(f"Tool execution result: {result}")
                results.append(f"{tool_name}: {result}")

            return "; ".join(results)

        # Fallback to JSON patterns
        patterns = [
            r"```json\s*(\{.*?\})\s*```",  # Standard markdown JSON
            r"```\s*(\{.*?\})\s*```",  # JSON without language spec
            r'\{\s*"tool_name"[^}]*\}',  # Bare JSON objects
            r'\{\s*"tool_name"[^}]*?\n[^}]*\}',  # Multi-line JSON
        ]

        json_match = None
        for pattern in patterns:
            json_match = re.search(pattern, response, re.DOTALL)
            if json_match:
                break

        if not json_match:
            return None

        try:
            # Get the JSON string - handle both group(1) and group(0)
            json_str = (
                json_match.group(1) if json_match.groups() else json_match.group(0)
            )

            # Clean up the JSON string
            json_str = json_str.strip()
            if not json_str.startswith("{"):
                # Find the first { and last }
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]

            # Parse the JSON
            tool_call = json.loads(json_str)

            tool_name = tool_call.get("tool_name")
            parameters = tool_call.get("parameters", {})
            reason = tool_call.get("reason", "")

            if not tool_name:
                return "Error: No tool_name specified in JSON"

            if self.verbose:
                print(f"Executing tool: {tool_name} with params: {parameters}")
                print(f"Reason: {reason}")

            # Execute the tool with validated parameters
            result = self.tools.execute(tool_name, parameters)

            if self.verbose:
                print(f"Tool result: {result}")

            return result

        except json.JSONDecodeError as e:
            return f"Error parsing tool JSON: {str(e)}"
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def _format_response(self, response: str) -> str:
        """Format response using configuration-specific formatter"""
        return self.config.format_response(response, self.state)

    def _remove_tool_calls_from_response(self, response: str) -> str:
        """Remove tool call syntax from response text to get clean dialogue"""
        import re

        # Remove TOOL_CALL: lines completely
        tool_call_pattern = r"TOOL_CALL:\s*\w+\s*\{[^}]*\}"
        clean_response = re.sub(tool_call_pattern, "", response, flags=re.DOTALL)

        # Remove any lines that mention tool calls
        lines = clean_response.split("\n")
        filtered_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            if not any(
                phrase in line_lower
                for phrase in [
                    "tool_call:",
                    "remember your name:",
                    "i'll remember",
                    "using the",
                    "tool",
                    "storing detail",
                    "adding to memory",
                ]
            ):
                filtered_lines.append(line)

        # Join lines and clean up extra whitespace
        clean_response = "\n".join(filtered_lines)
        clean_response = re.sub(
            r"\n\s*\n\s*\n", "\n\n", clean_response
        )  # Remove triple+ newlines
        clean_response = clean_response.strip()

        return clean_response

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

        summary_response = self.llm.chat(
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
