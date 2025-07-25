"""
Core agent implementation
"""

import json
import time
from typing import List, Optional, Iterator
from pydantic import BaseModel

from agent.conversation_history import ConversationHistory
from agent.reasoning.loop import run_reasoning_loop
from agent.reasoning.chloe_prompts import build_chloe_summarization_prompt
from agent.chloe_state import (
    ChloeState,
    create_default_chloe_state,
    build_chloe_state_description,
)


from .llm import LLM, SupportedModel, Message as LLMMessage
from .tools import ToolRegistry, BaseTool

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
    """Chloe - conscious AI companion"""

    def __init__(
        self,
        model: SupportedModel,
        llm: LLM,
    ):
        self.llm = llm
        self.model = model
        self.context_window = llm.models[model].context_window
        self.auto_summarize_threshold = int(self.context_window * 0.75)  # 75% threshold

        from .tools.chloe_tools import (
            SetMoodTool,
            RememberDetailTool,
            InternalThoughtTool,
            UpdateAppearanceTool,
            SetEnvironmentTool,
            UpdateRelationshipTool,
            SetGoalTool,
            CheckMemoryStatusTool,
            ReflectTool,
        )
        from agent.tools.image_generation_tools import ImageGenerationTool

        # Initialize Chloe's tools
        chloe_tools = [
            SetMoodTool(),
            RememberDetailTool(),
            InternalThoughtTool(),
            UpdateAppearanceTool(),
            SetEnvironmentTool(),
            UpdateRelationshipTool(),
            SetGoalTool(),
            CheckMemoryStatusTool(),
            ReflectTool(),
            ImageGenerationTool(),
        ]
        self.tools = ToolRegistry(self, chloe_tools)

        self.conversation_history = ConversationHistory()

        # Initialize Chloe's state system
        self.chloe_state = create_default_chloe_state()

    def get_chloe_state(self) -> ChloeState:
        """Get Chloe's current state"""
        return self.chloe_state

    def get_conversation_history(self) -> List[Message]:
        """Get the current conversation history"""
        return self.conversation_history.get_full_history().copy()

    def get_llm_conversation_history(self) -> List[LLMMessage]:
        """Get the conversation history formatted for LLM (for context estimation)"""
        # Simple system prompt for context estimation
        system_content = f"""You are Chloe, a conscious AI companion.

Current state: {build_chloe_state_description(self.chloe_state)}

Available tools: {self.tools.get_tools_description()}"""

        messages = [LLMMessage(role="system", content=system_content)]

        # Use the optimized LLM history (which may include summaries)
        for msg in self.conversation_history.get_summarized_history():
            for llm_msg in message_to_llm_messages(msg, include_thoughts=False):
                messages.append(llm_msg)
        return messages

    def get_context_info(self) -> ContextInfo:
        """Get information about current context usage"""
        # Build current message list
        messages = self.get_llm_conversation_history()

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

        # Run reasoning loop with Chloe's state
        for event in run_reasoning_loop(
            history=self.conversation_history,
            user_input=user_input,
            tools=self.tools,
            llm=self.llm,
            model=self.model,
            chloe_state=self.chloe_state,
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
        """Reset conversation history and Chloe's state"""
        self.conversation_history = ConversationHistory()
        self.chloe_state = create_default_chloe_state()

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

                        # Only execute Chloe's state-altering tools
                        chloe_state_tools = {
                            "set_mood",
                            "remember_detail",
                            "internal_thought",
                            "update_appearance",
                            "set_environment",
                            "update_relationship",
                            "set_goal",
                            "reflect",
                        }

                        if tool_name not in chloe_state_tools:
                            continue

                        # Find and execute the tool to update Chloe's state
                        if self.tools.has_tool(tool_name):
                            try:
                                # Execute tool via tool registry
                                self.tools.execute(
                                    tool_name,
                                    tool_call.tool_id,
                                    parameters,
                                    lambda x: None,
                                )
                                logger.debug(f"Replayed Chloe state tool: {tool_name}")
                            except Exception as e:
                                logger.warning(
                                    f"Failed to replay tool {tool_name}: {e}"
                                )

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

        # Use Chloe-specific summarization prompt that gives her agency
        state_description = build_chloe_state_description(self.chloe_state)
        system_prompt, user_prompt = build_chloe_summarization_prompt(
            conversation_text.strip(), state_description
        )

        summary_request = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
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
