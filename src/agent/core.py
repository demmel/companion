"""
Core agent implementation
"""

import json
import time
from typing import List, Optional, Iterator
from pydantic import BaseModel

from agent.conversation_history import ConversationHistory
from agent.reasoning.loop import run_reasoning_loop

from .llm import LLM, SupportedModel, Message as LLMMessage
from .tools import ToolRegistry
from .config import AgentConfig
from .types import (
    AgentMessage,
    ConversationData,
    Message,
    SummarizationContent,
    SystemMessage,
    TextContent,
    ThoughtContent,
    ToolCall,
    ToolCallFinished,
)
from .agent_events import (
    AgentEvent,
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

        self.conversation_history = ConversationHistory()

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
        return self.conversation_history.get_full_history().copy()

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
        for msg in self.conversation_history.get_summarized_history():
            for llm_msg in message_to_llm_messages(msg, include_thoughts=False):
                messages.append(llm_msg)
        return messages

    def get_context_info(self) -> ContextInfo:
        """Get information about current context usage"""
        # Build current message list
        messages = self.get_llm_conversation_history(
            include_tools=True, iteration_info=(1, self.config.max_iterations)
        )

        # Estimate token count (conservative approximation: 1 token ≈ 3 characters)
        total_chars = sum(len(msg.content) for msg in messages)
        estimated_tokens = int(total_chars / 3.4)

        return ContextInfo(
            message_count=len(messages),
            conversation_messages=len(
                self.conversation_history.get_full_history()
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
        """Streaming chat interface that yields typed events using reasoning loop"""
        start_time = time.time()

        # Check if we need auto-summarization before processing
        context_info = self.get_context_info()
        keep_recent = 10  # Conservative retention size
        if (
            context_info.approaching_limit
            and len(self.conversation_history.get_summarized_history()) > keep_recent
        ):
            # Perform auto-summarization with event emission
            for event in self._auto_summarize_with_events(keep_recent):
                yield event

        # Run reasoning loop
        for event in run_reasoning_loop(
            history=self.conversation_history,
            user_input=user_input,
            tools=self.tools,
            llm=self.llm,
            model=self.model,
        ):
            yield event

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
        """Reset conversation history"""
        self.conversation_history = ConversationHistory()

    def replay_conversation(
        self, conversation_data: ConversationData, up_to_index: Optional[int] = None
    ):
        """Replay a conversation from ConversationData up to a specific point

        Args:
            conversation_data: ConversationData loaded from JSON
            up_to_index: Stop replaying at this message index (exclusive). If None, replay all messages.
        """
        # Reset to clean state
        self.reset_conversation()
        self.state = (
            self.config.default_state.copy() if self.config.default_state else {}
        )

        # Determine how many messages to replay
        end_index = (
            up_to_index if up_to_index is not None else len(conversation_data.messages)
        )

        for i, message in enumerate(conversation_data.messages[:end_index]):
            self.conversation_history.append(message)

            # Summarization messages require us to fix the llm_conversation_history
            if isinstance(message, SystemMessage):
                has_summary = any(
                    isinstance(content, SummarizationContent)
                    for content in message.content
                )
                if has_summary:
                    logger.info(
                        f"Replaying summarization message at index {i}: {message.content}"
                    )
                    self.llm_conversation_history = [
                        message
                    ] + self.llm_conversation_history[-10:]
                else:
                    self.llm_conversation_history.append(message)
            else:
                self.llm_conversation_history.append(message)

            # Execute state-altering tools to rebuild agent state
            if isinstance(message, AgentMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, ToolCallFinished):
                        tool_name = tool_call.tool_name
                        parameters = tool_call.parameters

                        # Only execute state-altering tools (skip image generation, etc.)
                        state_altering_tools = {
                            "create_character",
                            "switch_character",
                            "set_mood",
                            "remember_detail",
                            "scene_setting",
                            "correct_detail",
                            "internal_thought",
                            "character_action",
                        }

                        if tool_name not in state_altering_tools:
                            continue

                        # Find and execute the tool to update agent state
                        for tool in self.config.tools:
                            if tool.name == tool_name:
                                try:
                                    # Create tool input and execute
                                    tool_input = tool.input_schema(**parameters)
                                    tool.run(
                                        self,
                                        tool_input,
                                        tool_call.tool_id,
                                        lambda x: None,
                                    )
                                    logger.debug(
                                        f"Replayed state-altering tool: {tool_name}"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to replay tool {tool_name}: {e}"
                                    )
                                break

        logger.info(
            f"Replayed {len(self.conversation_history)} messages, agent state restored"
        )

    @classmethod
    def load_conversation_data(cls, file_path: str) -> ConversationData:
        """Load conversation data from JSON file"""
        import json

        with open(file_path, "r") as f:
            data = json.load(f)
        return ConversationData.model_validate(data)

    def _auto_summarize_with_events(
        self, keep_recent: int = 10
    ) -> Iterator[AgentEvent]:
        """Auto-summarize with event emission for streaming clients"""
        # Calculate what we're about to do - work on LLM history only
        llm_history = self.conversation_history.get_summarized_history()
        old_messages = llm_history[:-keep_recent]
        recent_messages = llm_history[-keep_recent:]
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

        context_after = self.get_context_info()

        # Add structured summarization notification to user history
        # Find the position where summarization occurred in user history
        user_summary_index = len(self.conversation_history.get_full_history()) - len(
            recent_messages
        )

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
        self.conversation_history.insert_summary_notification(
            user_summary_index, summary_message, recent_messages
        )

        # Emit finished event
        yield SummarizationFinishedEvent(
            summary=summary_response,
            messages_summarized=len(old_messages),
            messages_after=len(self.conversation_history.get_summarized_history()),
            context_usage_after=context_after.usage_percentage,
        )


def message_to_llm_messages(
    message: Message, include_thoughts: bool = False
) -> Iterator[LLMMessage]:
    """Convert internal Message to LLMMessage format

    Args:
        message: Message to convert
        include_thoughts: Whether to include ThoughtContent in the LLM message
    """

    # Extract text from content list
    content_parts = []
    for item in message.content:
        if isinstance(item, TextContent):
            content_parts.append(item.text)
        elif isinstance(item, ThoughtContent):
            if include_thoughts:
                content_parts.append(f"<think>\n{item.text}\n</think>")
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
