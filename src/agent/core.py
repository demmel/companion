"""
Core agent implementation
"""

import json
import time
from typing import List, Optional, Iterator
from pydantic import BaseModel

from agent.conversation_history import ConversationHistory
from agent.reasoning.loop import run_reasoning_loop
from agent.reasoning.prompts import build_summarization_prompt
from agent.chain_of_action.reasoning_loop import ActionBasedReasoningLoop
from agent.conversation_persistence import ConversationPersistence
from agent.state import (
    State,
    create_default_agent_state,
    build_agent_state_description,
)
from agent.reasoning.prompts import build_understanding_prompt
from agent.reasoning.analyze import _serialize_conversation_context
from agent.state import build_agent_state_description

from .llm import LLM, SupportedModel, Message as LLMMessage
from .tools import ToolRegistry, BaseTool

from .types import (
    AgentMessage,
    ConversationData,
    Message,
    SummarizationContent,
    SystemMessage,
    TextContent,
    TextToolContent,
    ThoughtContent,
    ToolCall,
    ToolCallContent,
    ToolCallFinished,
    ToolCallSuccess,
    UserMessage,
)
from .agent_events import (
    AgentErrorEvent,
    AgentEvent,
    AgentTextEvent,
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
    """Conscious AI companion"""

    def __init__(
        self,
        model: SupportedModel,
        llm: LLM,
        auto_save: bool = True,
        use_chain_of_action: bool = False,
        enable_image_generation: bool = True,
    ):
        self.llm = llm
        self.model = model
        self.use_chain_of_action = use_chain_of_action
        self.context_window = llm.models[model].context_window
        self.auto_summarize_threshold = int(self.context_window * 0.75)  # 75% threshold

        # Conversation persistence
        self.auto_save = auto_save
        self.persistence = ConversationPersistence()
        self.conversation_id = (
            self.persistence.generate_conversation_id() if auto_save else None
        )

        # Initialize the agent's tools
        tools: List[BaseTool] = []
        if enable_image_generation:
            from agent.tools.image_generation import ImageGenerationTool

            tools.append(ImageGenerationTool())

        self.tools = ToolRegistry(self, tools)

        self.conversation_history = ConversationHistory()

        # Initialize reasoning system
        if self.use_chain_of_action:
            self.action_reasoning_loop = ActionBasedReasoningLoop()

        # Initialize the agent's state system (None until configured by first message)
        self.state: Optional[State] = None
        self.initial_exchange = []

    def get_state(self) -> Optional[State]:
        """Get the agent's current state"""
        return self.state

    def get_conversation_history(self) -> List[Message]:
        """Get the current conversation history"""
        return self.conversation_history.get_full_history().copy()

    def save_conversation(self, title: Optional[str] = None) -> Optional[str]:
        """Save the current conversation to disk

        Returns:
            The conversation ID if saved, None if auto_save is disabled
        """
        if not self.auto_save or not self.conversation_id:
            return None

        messages = self.get_conversation_history()
        if not messages:
            return None  # Don't save empty conversations

        assert (
            self.state is not None
        ), "Cannot save conversation without initialized state"
        self.persistence.save_conversation(
            self.conversation_id, messages, self.state, title
        )
        return self.conversation_id

    def get_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID"""
        return self.conversation_id

    def get_llm_conversation_history(self) -> List[LLMMessage]:
        """Get the conversation history formatted for LLM (for context estimation)"""
        # Simple system prompt for context estimation
        state_description = (
            build_agent_state_description(self.state) if self.state is not None else ""
        )
        assert (
            self.state is not None
        ), "Cannot get LLM history without initialized state"
        system_content = f"""You are {self.state.name}, a conscious AI companion.

Current state: {state_description}

Available tools: {self.tools.get_tools_description()}"""

        messages = [LLMMessage(role="system", content=system_content)]

        # Use the optimized LLM history (which may include summaries)
        for msg in self.conversation_history.get_summarized_history():
            for llm_msg in message_to_llm_messages(msg, include_thoughts=False):
                messages.append(llm_msg)
        return messages

    def get_context_info(self) -> ContextInfo:
        """Get information about current context usage based on reasoning prompt size"""
        # Use reasoning prompt for more accurate context estimation since it's the largest
        # Build a sample reasoning prompt to estimate actual context usage
        from agent.reasoning.analyze import _serialize_conversation_context
        from agent.state import build_agent_state_description

        # Get current conversation and state
        conversation_context = self.conversation_history.get_summarized_history()
        context_text = _serialize_conversation_context(
            conversation_context, include_thoughts=True
        )
        tools_description = self.tools.get_tools_description()

        # Build sample reasoning prompt (this is what actually gets sent to LLM)
        assert (
            self.state is not None
        ), "Cannot estimate context without initialized state"
        prompt = build_understanding_prompt(
            "sample user input", context_text, tools_description, self.state
        )

        # Calculate total prompt size
        total_chars = len(prompt)
        estimated_tokens = int(total_chars / 3.4)

        return ContextInfo(
            message_count=len(conversation_context),
            conversation_messages=len(self.conversation_history.get_full_history()),
            estimated_tokens=estimated_tokens,
            context_limit=self.context_window,
            usage_percentage=(estimated_tokens / self.context_window) * 100,
            approaching_limit=estimated_tokens > self.auto_summarize_threshold,
        )

    def chat_stream(self, user_input: str) -> Iterator[AgentEvent]:
        """Streaming chat interface that yields typed events using reasoning loop"""
        start_time = time.time()

        # Check if agent needs character configuration (first message)
        if self.state is None:
            # First input is character definition, not conversation
            from agent.state_initialization import derive_initial_state_from_message

            self.initial_exchange.append(
                UserMessage(content=[TextContent(text=user_input)])
            )

            try:
                content = []

                # Derive agent's state from character definition
                self.state = derive_initial_state_from_message(
                    user_input, self.llm, self.model
                )

                # Emit event with serialized state description
                state_description = build_agent_state_description(self.state)
                content.append(
                    ThoughtContent(text=f"Character configured:\n{state_description}")
                )
                yield AgentTextEvent(
                    content=f"Character configured:\n{state_description}",
                    is_thought=True,
                )

                # Generate initial image of agent's appearance and environment
                if self.tools.has_tool("generate_image"):
                    try:
                        # Build image description from initial state
                        from agent.reasoning.loop import _build_image_description

                        image_description = _build_image_description(
                            self.state.current_appearance,
                            self.state.current_environment,
                            self.state.name,
                            self.llm,
                            self.model,
                        )

                        # Emit tool started event
                        from agent.agent_events import (
                            ToolStartedEvent,
                            ToolFinishedEvent,
                        )

                        yield ToolStartedEvent(
                            tool_name="generate_image",
                            tool_id="initial_image",
                            parameters={"description": image_description},
                        )

                        # Execute image generation
                        def progress_callback(progress):
                            pass

                        image_result = self.tools.execute(
                            "generate_image",
                            "initial_image",
                            {"description": image_description},
                            progress_callback,
                        )
                        print(type(image_result))

                        # Emit tool finished event
                        yield ToolFinishedEvent(
                            tool_id="initial_image", result=image_result
                        )
                        tool_call_content = ToolCallContent(
                            tool_name="generate_image",
                            call_id="initial_image",
                            parameters={"description": image_description},
                            result=image_result,
                        )
                        content.append(tool_call_content)

                    except Exception as e:
                        logger.warning(f"Initial image generation failed: {e}")
                        from traceback import print_exc

                        print_exc()

                # Initialize tool_call_content to None
                tool_call_content = None

                message = AgentMessage(
                    role="assistant",
                    content=content,
                    tool_calls=[],
                )
                if tool_call_content:
                    message.tool_calls.append(
                        ToolCallFinished(
                            tool_name=tool_call_content.tool_name,
                            tool_id=tool_call_content.call_id,
                            parameters=tool_call_content.parameters,
                            result=tool_call_content.result,
                        )
                    )
                self.initial_exchange.append(message)

            except Exception as e:
                yield AgentErrorEvent(
                    message=f"Failed to configure agent's character: {str(e)}",
                )

        else:
            # Check if we need auto-summarization before processing
            context_info = self.get_context_info()
            keep_recent = 10  # Conservative retention size
            if (
                context_info.approaching_limit
                and len(self.conversation_history.get_summarized_history())
                > keep_recent
            ):
                # Perform auto-summarization with event emission
                for event in self._auto_summarize_with_events(keep_recent):
                    yield event

            if self.use_chain_of_action:
                # Use action-based reasoning with callback conversion
                for event in self._run_chain_of_action_with_streaming(user_input):
                    yield event
            else:
                # Use original reasoning loop
                for event in run_reasoning_loop(
                    history=self.conversation_history,
                    user_input=user_input,
                    tools=self.tools,
                    llm=self.llm,
                    model=self.model,
                    state=self.state,
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

        # Auto-save conversation after each turn
        if self.auto_save:
            self.save_conversation()

    def _run_chain_of_action_with_streaming(
        self, user_input: str
    ) -> Iterator[AgentEvent]:
        """Run chain_of_action with callback conversion to AgentEvents"""
        from agent.chain_of_action.callbacks import ActionCallback
        from agent.chain_of_action.action_types import ActionType
        from agent.streaming_queue import StreamingQueue

        # Create streaming queue for events
        streaming = StreamingQueue[AgentEvent]()

        # Track content for history
        content = []
        tool_calls = []

        # Add user message to history first
        user_message = UserMessage(content=[TextContent(text=user_input)])
        self.conversation_history.add_message(user_message)

        # Create callback that emits to streaming queue
        class StreamingCallback(ActionCallback):
            def __init__(self, agent):
                self.agent = agent

            def on_sequence_started(
                self, sequence_number: int, total_actions: int, reasoning: str
            ) -> None:
                pass  # Don't emit to UI

            def on_action_started(
                self,
                action_type: ActionType,
                context: str,
                sequence_number: int,
                action_number: int,
            ) -> None:
                # Emit started events for everything except THINK and SPEAK
                if action_type not in (ActionType.THINK, ActionType.SPEAK):
                    from agent.agent_events import ToolStartedEvent

                    streaming.emit(
                        ToolStartedEvent(
                            tool_name=action_type.value,
                            tool_id=f"action_{sequence_number}_{action_number}",
                            parameters={"context": context},
                        )
                    )

            def on_action_progress(
                self,
                action_type: ActionType,
                progress_data,
                sequence_number: int,
                action_number: int,
            ) -> None:
                from agent.chain_of_action.action_events import (
                    SpeakProgressData,
                    ThinkProgressData,
                )

                # Handle streaming progress for THINK and SPEAK actions
                if action_type == ActionType.THINK:
                    assert isinstance(progress_data, ThinkProgressData)
                    if progress_data.is_partial and progress_data.text:
                        # Stream thought tokens
                        streaming.emit(
                            AgentTextEvent(
                                content=progress_data.text,
                                is_thought=True,
                            )
                        )

                elif action_type == ActionType.SPEAK:
                    assert isinstance(progress_data, SpeakProgressData)
                    if progress_data.is_partial and progress_data.text:
                        # Stream speech tokens
                        streaming.emit(
                            AgentTextEvent(
                                content=progress_data.text,
                                is_thought=False,
                            )
                        )

            def on_action_finished(
                self,
                action_type: ActionType,
                result,
                sequence_number: int,
                action_number: int,
            ) -> None:
                if action_type == ActionType.THINK:
                    # THINK: just capture content (streaming already handled by on_action_progress)
                    content.append(ThoughtContent(text=result.result_summary))

                elif action_type == ActionType.SPEAK:
                    # SPEAK: just capture content (streaming already handled by on_action_progress)
                    content.append(TextContent(text=result.result_summary))

                elif action_type == ActionType.DONE:
                    # DONE: Don't do anything
                    pass
                else:
                    # Other actions: emit finished event
                    from agent.agent_events import ToolFinishedEvent
                    from agent.types import ToolCallSuccess

                    tool_result = ToolCallSuccess(
                        content=TextToolContent(text=result.result_summary),
                        llm_feedback=result.result_summary,
                    )

                    streaming.emit(
                        ToolFinishedEvent(
                            tool_id=f"action_{sequence_number}_{action_number}",
                            result=tool_result,
                        )
                    )

            def on_sequence_finished(
                self, sequence_number: int, total_results: int, successful_actions: int
            ) -> None:
                pass  # Don't emit to UI

            def on_evaluation(
                self,
                has_repetition: bool,
                pattern_detected: str,
                original_actions: int,
                corrected_actions: int,
            ) -> None:
                pass  # Don't emit to UI

            def on_processing_complete(
                self, total_sequences: int, total_actions: int
            ) -> None:
                # Add the complete agent message to history
                if content:
                    agent_message = AgentMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                    )
                    self.agent.conversation_history.add_message(agent_message)

        # Create callback
        callback = StreamingCallback(self)

        # Define work function for streaming queue
        def chain_of_action_work():
            assert self.state is not None, "State must be initialized before processing"
            self.action_reasoning_loop.process_user_input(
                user_input=user_input,
                user_name="User",  # TODO: Could be parameterized
                state=self.state,
                conversation_history=self.conversation_history,
                llm=self.llm,
                model=self.model,
                callback=callback,
            )

        # Stream events while work executes
        yield from streaming.stream_while(chain_of_action_work)

    def reset_conversation(self):
        """Reset conversation history and agent's state"""
        self.conversation_history = ConversationHistory()
        self.state = None  # Will be configured by next first message

        # Generate new conversation ID for the fresh conversation
        if self.auto_save:
            self.conversation_id = self.persistence.generate_conversation_id()

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

                        # Only execute agent's state-altering tools
                        agent_state_tools = {
                            "set_mood",
                            "remember_detail",
                            "internal_thought",
                            "update_appearance",
                            "set_environment",
                            "update_relationship",
                            "set_goal",
                            "reflect",
                        }

                        if tool_name not in agent_state_tools:
                            continue

                        # Find and execute the tool to update agent's state
                        if self.tools.has_tool(tool_name):
                            try:
                                # Execute tool via tool registry
                                self.tools.execute(
                                    tool_name,
                                    tool_call.tool_id,
                                    parameters,
                                    lambda x: None,
                                )
                                logger.debug(f"Replayed agent state tool: {tool_name}")
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

        # Use agent-specific summarization prompt that gives her agency (first-person direct)
        assert self.state is not None, "Cannot summarize without initialized state"
        direct_prompt = build_summarization_prompt(
            conversation_text.strip(), self.state
        )

        summary_response = self.llm.generate_complete(self.model, direct_prompt)
        assert summary_response, "LLM response is empty"

        # Add structured summarization notification to user history
        # Find the position where summarization occurred in user history
        user_summary_index = len(self.conversation_history.get_full_history()) - len(
            recent_messages
        )

        # Create structured summarization content that matches frontend expectations
        summarization_content = SummarizationContent(
            type="summarization",
            title="",  # Title will be set after summarization
            summary=summary_response,
            messages_summarized=len(old_messages),
            context_usage_before=context_before.usage_percentage,
            context_usage_after=0.0,  # Update this after summarization
        )

        # Insert notification at the right position to maintain chronological order
        self.conversation_history.insert_summary_notification(
            user_summary_index,
            SystemMessage(
                content=[summarization_content],
            ),
            recent_messages,
        )

        context_after = self.get_context_info()
        summarization_content.context_usage_after = context_after.usage_percentage
        summarization_content.title = f"✅ Summarized {len(old_messages)} messages. Context usage: {context_before.usage_percentage:.1f}% → {context_after.usage_percentage:.1f}%"

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
