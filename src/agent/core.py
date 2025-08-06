"""
Core agent implementation
"""

import json
import time
from typing import List, Optional, Iterator
from pydantic import BaseModel

from agent.chain_of_action.trigger_history import TriggerHistory
from agent.llm import LLM, SupportedModel
from agent.state import State

from agent.chain_of_action.reasoning_loop import ActionBasedReasoningLoop
from agent.chain_of_action.trigger_history import TriggerHistory
from agent.conversation_persistence import ConversationPersistence
from agent.state import (
    State,
    build_agent_state_description,
)
from agent.state import build_agent_state_description

from .llm import LLM, SupportedModel, Message as LLMMessage
from .tools import ToolRegistry, BaseTool

from .types import (
    AgentMessage,
    ConversationData,
    Message,
    SummarizationContent,
    TextContent,
    TextToolContent,
    ThoughtContent,
    ToolCall,
    ToolCallContent,
    ToolCallFinished,
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
        enable_image_generation: bool = True,
    ):
        self.llm = llm
        self.model = model
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
            from agent.tools.image_generation import get_shared_image_generator

            tools.append(get_shared_image_generator())

        self.tools = ToolRegistry(self, tools)

        self.trigger_history = TriggerHistory()

        # Initialize reasoning system
        self.action_reasoning_loop = ActionBasedReasoningLoop()

        # Initialize the agent's state system (None until configured by first message)
        self.state: Optional[State] = None
        self.initial_exchange = []

    def get_state(self) -> Optional[State]:
        """Get the agent's current state"""
        return self.state

    def get_trigger_history(self) -> TriggerHistory:
        """Get the current trigger history"""
        return self.trigger_history

    def get_conversation_history(self) -> List[Message]:
        """Get the current conversation history (converted from trigger history)"""
        return self.trigger_history.to_conversation_history()

    def save_conversation(self, title: Optional[str] = None) -> Optional[str]:
        """Save the current conversation to disk

        Returns:
            The conversation ID if saved, None if auto_save is disabled
        """
        if not self.auto_save or not self.conversation_id:
            return None

        assert (
            self.state is not None
        ), "Cannot save conversation without initialized state"
        self.persistence.save_conversation(
            self.conversation_id,
            self.state,
            self.trigger_history,
            title,
        )
        return self.conversation_id

    def get_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID"""
        return self.conversation_id

    def get_context_info(self) -> ContextInfo:
        """Get information about current context usage based on action planning prompt size"""
        if self.state is not None:
            # Use action planning prompt for accurate estimation
            from agent.chain_of_action.prompts import build_action_planning_prompt
            from agent.chain_of_action.trigger import UserInputTrigger

            # Create a sample trigger for estimation
            sample_trigger = UserInputTrigger(content="sample user input")

            prompt = build_action_planning_prompt(
                state=self.state,
                trigger=sample_trigger,
                completed_actions=[],  # Empty for estimation
                trigger_history=self.trigger_history,
                registry=self.action_reasoning_loop.registry,
            )

            # Calculate total prompt size
            total_chars = len(prompt)
            estimated_tokens = int(total_chars / 3.4)
        else:
            # Fallback estimation when state is not initialized yet
            estimated_tokens = 1000  # Base prompt overhead estimate

        return ContextInfo(
            message_count=len(self.trigger_history.get_recent_entries()),
            conversation_messages=len(self.trigger_history.to_conversation_history()),
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
                        from agent.chain_of_action.actions.update_appearance_action import (
                            _build_image_description,
                        )

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
            if context_info.approaching_limit:
                # Perform auto-summarization with event emission
                for event in self._auto_summarize_with_events(keep_recent):
                    yield event

            # Use action-based reasoning with callback conversion
            for event in self._run_chain_of_action_with_streaming(user_input):
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
                if action_type == ActionType.UPDATE_APPEARANCE:
                    # For UPDATE_APPEARANCE, emit image generation tool started
                    from agent.agent_events import ToolStartedEvent

                    streaming.emit(
                        ToolStartedEvent(
                            tool_name="generate_image",
                            tool_id=f"appearance_image_{sequence_number}_{action_number}",
                            parameters={"description": context},
                        )
                    )
                elif action_type not in (ActionType.THINK, ActionType.SPEAK):
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

                elif action_type == ActionType.UPDATE_APPEARANCE:
                    # UPDATE_APPEARANCE: Handle image generation tool events and add to tool_calls
                    if result.metadata and "image_result" in result.metadata:
                        image_result = result.metadata["image_result"]
                        tool_id = f"appearance_image_{sequence_number}_{action_number}"

                        # Emit tool started and finished events for image generation
                        from agent.agent_events import (
                            ToolStartedEvent,
                            ToolFinishedEvent,
                        )

                        streaming.emit(
                            ToolStartedEvent(
                                tool_name="generate_image",
                                tool_id=tool_id,
                                parameters={
                                    "description": result.metadata["image_description"]
                                },
                            )
                        )

                        streaming.emit(
                            ToolFinishedEvent(
                                tool_id=tool_id,
                                result=image_result,
                            )
                        )

                        # Add to tool_calls for conversation history
                        from agent.types import ToolCallFinished

                        tool_calls.append(
                            ToolCallFinished(
                                tool_name="generate_image",
                                tool_id=tool_id,
                                parameters={
                                    "description": result.metadata["image_description"]
                                },
                                result=image_result,
                            )
                        )

                elif action_type == ActionType.WAIT:
                    # WAIT: Don't do anything special for UI
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
                pass

        # Create callback
        callback = StreamingCallback(self)

        # Define work function for streaming queue
        def chain_of_action_work():
            assert self.state is not None, "State must be initialized before processing"

            # Process with trigger history integration
            self.action_reasoning_loop.process_user_input(
                user_input=user_input,
                user_name="User",  # TODO: Could be parameterized
                state=self.state,
                llm=self.llm,
                model=self.model,
                callback=callback,
                trigger_history=self.trigger_history,
            )

        # Stream events while work executes
        yield from streaming.stream_while(chain_of_action_work)

    def reset_conversation(self):
        """Reset conversation history and agent's state"""
        self.trigger_history = TriggerHistory()
        self.state = None  # Will be configured by next first message

        # Generate new conversation ID for the fresh conversation
        if self.auto_save:
            self.conversation_id = self.persistence.generate_conversation_id()

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
        llm_history = self.trigger_history.get_recent_entries()
        old_triggers = llm_history[:-keep_recent]
        recent_triggers = llm_history[-keep_recent:]
        context_before = self.get_context_info()

        # Emit started event
        yield SummarizationStartedEvent(
            messages_to_summarize=len(old_triggers),
            recent_messages_kept=len(recent_triggers),
            context_usage_before=context_before.usage_percentage,
        )

        # Update trigger history with summary
        assert self.state is not None, "State must be initialized for summarization"
        summary_response = summarize_trigger_history(
            self.trigger_history, keep_recent, self.llm, self.model, self.state
        )

        # Create structured summarization content that matches frontend expectations
        summarization_content = SummarizationContent(
            type="summarization",
            title="",  # Title will be set after summarization
            summary=summary_response,
            messages_summarized=len(old_triggers),
            context_usage_before=context_before.usage_percentage,
            context_usage_after=0.0,  # Update this after summarization
        )

        context_after = self.get_context_info()
        summarization_content.context_usage_after = context_after.usage_percentage
        summarization_content.title = f"✅ Summarized {len(old_triggers)} messages. Context usage: {context_before.usage_percentage:.1f}% → {context_after.usage_percentage:.1f}%"

        # Emit finished event
        yield SummarizationFinishedEvent(
            summary=summary_response,
            messages_summarized=len(old_triggers),
            messages_after=len(self.trigger_history.get_recent_entries()),
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


def summarize_trigger_history(
    trigger_history: TriggerHistory,
    keep_recent: int,
    llm: LLM,
    model: SupportedModel,
    state: State,
) -> str:
    """Pure function to summarize trigger history entries"""
    from agent.chain_of_action.prompts import (
        build_summarization_prompt,
        format_single_trigger_entry,
    )

    total_entries = len(trigger_history.get_all_entries())
    entries_to_summarize_end = total_entries - keep_recent

    if entries_to_summarize_end <= 0:
        return ""

    # Get old summary if exists
    old_summary = (
        trigger_history.summaries[-1].summary_text if trigger_history.summaries else ""
    )
    summary_start = (
        trigger_history.summaries[-1].insert_at_index
        if trigger_history.summaries
        else 0
    )

    # Format entries to summarize
    entries_to_format = trigger_history.entries[summary_start:entries_to_summarize_end]
    if not entries_to_format:
        return ""

    formatted_entries = [
        format_single_trigger_entry(entry) for entry in entries_to_format
    ]
    recent_entries = "\n\n".join(formatted_entries)

    # Use structured prompt with separated sections
    summary = llm.generate_complete(
        model, build_summarization_prompt(old_summary, recent_entries, state)
    )
    trigger_history.add_summary(summary, entries_to_summarize_end)
    return summary
