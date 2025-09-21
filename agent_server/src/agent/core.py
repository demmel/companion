"""
Core agent implementation
"""

from datetime import datetime
import time
import threading
import queue
from typing import List, Optional
from agent.chain_of_action.action_registry import ActionRegistry
from agent.chain_of_action.prompts import build_situational_analysis_prompt
from agent.experiments.temporal_context_dag.models import ContextGraph
from pydantic import BaseModel

from agent.api_types import (
    ActionProgressEvent,
    convert_action_to_dto,
    convert_trigger_to_dto,
)
from agent.chain_of_action.action.action_data import (
    ActionData,
    ThinkActionData,
    UpdateAppearanceActionData,
)
from agent.chain_of_action.action.actions.think_action import ThinkInput, ThinkOutput
from agent.chain_of_action.action.actions.visual_actions import (
    UpdateAppearanceInput,
    UpdateAppearanceOutput,
)
from agent.chain_of_action.action.base_action_data import (
    ActionFailureResult,
    ActionSuccessResult,
)
from agent.chain_of_action.trigger import (
    Trigger,
    UserInputTrigger,
    WakeupTrigger,
)
from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry
from agent.llm import LLM, SupportedModel
from agent.state import State

from agent.chain_of_action.reasoning_loop import ActionBasedReasoningLoop
from agent.chain_of_action.trigger_history import TriggerHistory
from agent.conversation_persistence import ConversationPersistence
from agent.state import (
    State,
)
from agent.types import ToolCallError

from .llm import LLM, SupportedModel

from .types import (
    ImageGenerationToolContent,
    SummarizationContent,
    ToolCallSuccess,
)
from agent.api_types import (
    ActionCompletedEvent,
    ActionStartedEvent,
    AgentErrorEvent,
    TriggerEvent as AgentEvent,
    SummarizationStartedEvent,
    SummarizationFinishedEvent,
    TriggerCompletedEvent,
    TriggerStartedEvent,
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
        continuous_summarization: bool = False,
        keep_recent: int = 2,
        individual_trigger_compression: bool = True,
        enable_dag_memory: bool = False,
        auto_summarize_threshold: Optional[int] = None,
    ):
        self.llm = llm
        self.model = model
        self.context_window = llm.models[model].context_window

        # Set summarization threshold
        if auto_summarize_threshold is not None:
            self.auto_summarize_threshold = auto_summarize_threshold
        else:
            # Default 60% threshold
            self.auto_summarize_threshold = int(self.context_window * 0.6)
        self.enable_image_generation = enable_image_generation
        self.continuous_summarization = continuous_summarization
        self.keep_recent = keep_recent
        self.individual_trigger_compression = individual_trigger_compression
        self.enable_dag_memory = enable_dag_memory

        # Conversation persistence
        self.auto_save = auto_save
        self.persistence = ConversationPersistence()
        self.conversation_id = (
            self.persistence.generate_conversation_id() if auto_save else None
        )

        self.trigger_history = TriggerHistory()

        # Initialize reasoning system
        self.action_reasoning_loop = ActionBasedReasoningLoop(
            enable_image_generation=enable_image_generation,
        )

        # Initialize the agent's state system (None until configured by first message)
        self.state: Optional[State] = None
        self.initial_exchange = None

        # DAG memory system (initialized after first message if enabled)
        self.dag_memory_manager = None

        # Single client queue for WebSocket communication
        self.current_client_queue: Optional[queue.Queue[AgentEvent]] = None
        self.client_queue_lock = threading.Lock()

        # Auto-wakeup timer functionality
        self.auto_wakeup_enabled = False
        self.auto_wakeup_timer: Optional[threading.Timer] = None
        self.processing_condition = (
            threading.Condition()
        )  # Condition for processing coordination
        self.is_processing = False
        self.wakeup_delay_seconds = 5 * 60  # 5 minutes

    def get_state(self) -> Optional[State]:
        """Get the agent's current state"""
        return self.state

    def get_trigger_history(self) -> TriggerHistory:
        """Get the current trigger history"""
        return self.trigger_history

    def reset_conversation(self):
        """Reset conversation history and agent's state"""
        self.trigger_history = TriggerHistory()
        self.state = None  # Will be configured by next first message

        # Reset LLM call statistics
        self.llm.reset_call_stats()

        # Generate new conversation ID for the fresh conversation
        if self.auto_save:
            self.conversation_id = self.persistence.generate_conversation_id()

    def save_conversation(self, title: Optional[str] = None) -> Optional[str]:
        """Save the current conversation to disk

        Returns:
            The conversation ID if saved, None if auto_save is disabled
        """
        logger.info(
            f"save_conversation called: auto_save={self.auto_save}, conversation_id={self.conversation_id}, state={self.state is not None}"
        )
        if not self.auto_save or not self.conversation_id:
            logger.info(
                f"Not saving: auto_save={self.auto_save}, conversation_id={self.conversation_id}"
            )
            return None

        assert (
            self.state is not None
        ), "Cannot save conversation without initialized state"
        assert (
            self.initial_exchange is not None
        ), "Cannot save conversation without initial exchange"
        assert (
            self.dag_memory_manager is not None
        ), "Cannot save conversation without DAG memory manager"
        logger.info(
            f"Saving conversation {self.conversation_id} with {len(self.trigger_history)} entries"
        )
        self.persistence.save_conversation(
            self.conversation_id,
            self.state,
            self.trigger_history,
            self.initial_exchange,
            dag_memory_manager=self.dag_memory_manager,
        )
        logger.info(f"Successfully saved conversation {self.conversation_id}")
        return self.conversation_id

    def load_conversation(self, conversation_id: str):
        """Load a conversation from disk by its ID"""

        logger.info(f"Loading conversation {conversation_id}")

        agent_data = self.persistence.load_agent_data(conversation_id)

        self.reset_conversation()  # Clear existing history and state

        self.state = agent_data.state
        self.initial_exchange = agent_data.initial_exchange
        self.trigger_history = agent_data.trigger_history
        self.dag_memory_manager = agent_data.dag_memory_manager

    def get_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID"""
        return self.conversation_id

    def set_auto_wakeup_enabled(self, enabled: bool) -> None:
        """Enable or disable auto-wakeup timer"""
        with self.processing_condition:
            self.auto_wakeup_enabled = enabled
            if not enabled:
                self._cancel_wakeup_timer()
            else:
                self._schedule_wakeup_timer()

    def get_auto_wakeup_enabled(self) -> bool:
        """Get current auto-wakeup enabled state"""
        return self.auto_wakeup_enabled

    def emit_event(self, event: AgentEvent, should_yield: bool = False) -> None:
        """Emit an event to the current client (if any)"""
        with self.client_queue_lock:
            if self.current_client_queue:
                self.current_client_queue.put(event)
            # else: drop event (no client connected)

        if should_yield:
            time.sleep(0)  # Yield to allow event to be processed

    def set_client_queue(self, client_queue: queue.Queue[AgentEvent]) -> None:
        """Set the current client queue (replaces existing client)"""
        with self.client_queue_lock:
            self.current_client_queue = client_queue

    def clear_client_queue(self, queue_to_clear: queue.Queue[AgentEvent]) -> None:
        """Clear the current client queue only if it matches the provided queue"""
        with self.client_queue_lock:
            if self.current_client_queue is queue_to_clear:
                self.current_client_queue = None
            # else: different client's queue is active, don't clear

    def _cancel_wakeup_timer(self) -> None:
        """Cancel the current wakeup timer if it exists"""
        logger.info("Cancelling existing wakeup timer if any")
        if self.auto_wakeup_timer:
            self.auto_wakeup_timer.cancel()
            self.auto_wakeup_timer = None

    def _schedule_wakeup_timer(self) -> None:
        """Schedule a wakeup timer if auto-wakeup is enabled"""
        logger.info("Scheduling wakeup timer")
        with self.processing_condition:
            if not self.auto_wakeup_enabled or self.is_processing:
                return

            self._cancel_wakeup_timer()  # Cancel any existing timer

            def wakeup_callback():
                logger.info("Auto-wakeup timer triggered")

                # Trigger a wakeup by calling chat_stream with None in background thread
                def trigger_wakeup():
                    try:
                        self.chat_stream(WakeupTrigger())
                    except Exception as e:
                        logger.error(f"Auto-wakeup processing error: {e}")

                # Run in separate thread to avoid blocking timer thread
                import threading

                wakeup_thread = threading.Thread(target=trigger_wakeup, daemon=True)
                wakeup_thread.start()

            self.auto_wakeup_timer = threading.Timer(
                self.wakeup_delay_seconds, wakeup_callback
            )
            self.auto_wakeup_timer.start()

    def get_context_info(self) -> ContextInfo:
        """Get information about current context usage based on action planning prompt size"""
        # Get DAG context if available
        dag_context = None
        if self.dag_memory_manager is not None:
            dag_context = self.dag_memory_manager.get_current_context()

        return get_context_info(
            state=self.state,
            trigger_history=self.trigger_history,
            action_registry=self.action_reasoning_loop.registry,
            summarize_at_tokens=self.auto_summarize_threshold,
            dag_context=dag_context,
        )

    def chat_stream(self, trigger: Trigger) -> None:
        """Streaming chat interface that yields typed events using reasoning loop"""
        # Wait for any existing processing to complete, then acquire processing
        with self.processing_condition:
            # Wait until not processing
            while self.is_processing:
                logger.info("chat_stream waiting for existing processing to complete")
                self.processing_condition.wait()

            self._cancel_wakeup_timer()

            # Now we can start processing
            self.is_processing = True

        try:
            start_time = time.time()

            # Check if agent needs character configuration (first message)
            if self.state is None:
                if not isinstance(trigger, UserInputTrigger):
                    # Ignore non-user-input triggers during initialization - just return
                    return
                # Run initial exchange with character definition
                self._run_initial_exchange_with_streaming(trigger)
            else:
                # Check if we need auto-summarization before processing (skip if continuous summarization enabled or DAG memory)
                if not self.continuous_summarization and not self.enable_dag_memory:
                    context_info = self.get_context_info()
                    if context_info.estimated_tokens >= context_info.context_limit:
                        # Perform auto-summarization with event emission
                        self._auto_summarize_with_events(self.keep_recent)

                # Use action-based reasoning with callback conversion
                self._run_chain_of_action_with_streaming(trigger)

                # Continuous summarization: summarize after every trigger if enabled (skip for DAG memory)
                if (
                    self.continuous_summarization
                    and not self.enable_dag_memory
                    and len(self.trigger_history.entries) > self.keep_recent
                ):
                    self._auto_summarize_with_events(self.keep_recent)

            # Auto-save conversation after each turn
            logger.info(f"Checking auto-save: auto_save={self.auto_save}")
            if self.auto_save:
                logger.info("Triggering auto-save after chat stream")
                self.save_conversation()

            self.llm.log_stats_summary()

            total_time = time.time() - start_time
            logger.debug(f"Total chat_stream time: {total_time:.3f}s")

        except Exception as e:
            logger.error(f"Error occurred during chat_stream: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Clear processing flag, schedule next wakeup timer, and notify waiting threads
            with self.processing_condition:
                self.is_processing = False
                self._schedule_wakeup_timer()
                # Notify any threads waiting for processing to complete
                self.processing_condition.notify_all()

    def _run_initial_exchange_with_streaming(self, trigger: UserInputTrigger):
        """Run initial character configuration with streaming events"""

        # First input is character definition, not conversation
        from agent.state_initialization import derive_initial_state_from_message

        self.initial_exchange = TriggerHistoryEntry(
            trigger=trigger,
            situational_context="",  # No situational context for initial exchange
            actions_taken=[],
        )

        # Capture entry_id for use in nested callbacks
        entry_id = self.initial_exchange.entry_id

        self.emit_event(
            TriggerStartedEvent(
                trigger=convert_trigger_to_dto(self.initial_exchange.trigger),
                entry_id=entry_id,
                timestamp=self.initial_exchange.timestamp.isoformat(),
            ),
            should_yield=True,
        )

        try:
            self.emit_event(
                ActionStartedEvent(
                    entry_id=entry_id,
                    action_type="think",
                    context_given="Deriving initial state from character definition",
                    reasoning="Deriving initial state from character definition",
                    timestamp=datetime.now().isoformat(),
                    sequence_number=1,
                    action_number=1,
                ),
                should_yield=True,
            )
            derive_state_start_time = time.time()

            # Derive agent's state from character definition
            self.state, backstory = derive_initial_state_from_message(
                trigger.content, self.llm, self.model, trigger.get_images()
            )

            # Initialize DAG memory system if enabled
            if self.enable_dag_memory:
                from agent.experiments.temporal_context_dag import (
                    DagMemoryManager,
                )

                self.dag_memory_manager = DagMemoryManager.create(
                    initial_state=self.state,
                    backstory=backstory,
                    token_budget=self.auto_summarize_threshold,
                    action_registry=self.action_reasoning_loop.registry,
                )

            # Create and store the action result
            state_description = "\n".join(
                [
                    f"Name: {self.state.name}",
                    f"Role: {self.state.role}",
                    f"Mood: {self.state.current_mood}",
                    f"Environment: {self.state.current_environment}",
                    f"Appearance: {self.state.current_appearance}",
                    f"Backstory: {backstory}",
                    "Core Values:",
                    *[f"- {value.content}" for value in self.state.core_values],
                    "Priorities:",
                    *[
                        f"- {priority.content}"
                        for priority in self.state.current_priorities
                    ],
                ]
            )
            think_action_result = ThinkActionData(
                reasoning="Deriving initial state from character definition",
                input=ThinkInput(
                    focus="Deriving initial state",
                ),
                result=ActionSuccessResult(
                    content=ThinkOutput(
                        thoughts=state_description,
                    )
                ),
                duration_ms=int((time.time() - derive_state_start_time) * 1000),
            )
            self.initial_exchange.actions_taken.append(think_action_result)

            # Emit completion event
            self.emit_event(
                ActionCompletedEvent(
                    entry_id=entry_id,
                    action=convert_action_to_dto(think_action_result),
                    sequence_number=1,
                    action_number=1,
                    timestamp=datetime.now().isoformat(),
                ),
                should_yield=True,
            )

            # Generate initial image of agent's appearance and environment
            if self.enable_image_generation:
                self.emit_event(
                    ActionStartedEvent(
                        entry_id=entry_id,
                        action_type="update_appearance",
                        context_given=self.state.current_appearance,
                        reasoning="Initial appearance image",
                        timestamp=datetime.now().isoformat(),
                        sequence_number=1,
                        action_number=2,
                    ),
                    should_yield=True,
                )
                input = UpdateAppearanceInput(
                    reason="Initial appearance image",
                    change_description=self.state.current_appearance,
                )
                generate_image_start_time = time.time()
                image_description = self.state.current_appearance  # Default fallback
                try:
                    # Build image description from initial state
                    from agent.chain_of_action.action.actions.visual_actions import (
                        _build_image_description,
                    )

                    image_description = _build_image_description(
                        self.state.current_appearance,
                        self.state.current_environment,
                        self.state.name,
                        self.llm,
                        self.model,
                    )

                    # Execute image generation with progress callback
                    def progress_callback(progress):
                        # Emit progress events for image generation
                        self.emit_event(
                            ActionProgressEvent(
                                entry_id=entry_id,
                                action_type="update_appearance",
                                partial_result=f"Generating image: {progress}%",
                                sequence_number=1,
                                action_number=2,
                                timestamp=datetime.now().isoformat(),
                            )
                        )

                    from agent.image_generation import (
                        get_shared_image_generator,
                    )

                    image_generator = get_shared_image_generator()
                    image_result: ToolCallSuccess | ToolCallError = (
                        image_generator.generate_image_direct(
                            image_description,
                            self.llm,
                            self.model,
                            progress_callback,
                        )
                    )

                    if not isinstance(image_result, ToolCallSuccess):
                        raise ValueError(f"Image generation failed: {image_result}")

                    assert isinstance(
                        image_result.content, ImageGenerationToolContent
                    ), "Image generation tool must return ImageGenerationToolContent"

                    output = UpdateAppearanceOutput(
                        image_description=image_description,
                        old_appearance="",
                        new_appearance=self.state.current_appearance,
                        reason="Initial appearance image",
                        image_result=image_result.content,
                    )

                    appearance_action_result = UpdateAppearanceActionData(
                        reasoning="Initial appearance image",
                        input=input,
                        result=ActionSuccessResult(content=output),
                        duration_ms=int(
                            (time.time() - generate_image_start_time) * 1000
                        ),
                    )

                    self.initial_exchange.actions_taken.append(appearance_action_result)

                    # Emit completion event
                    self.emit_event(
                        ActionCompletedEvent(
                            entry_id=entry_id,
                            action=convert_action_to_dto(appearance_action_result),
                            sequence_number=1,
                            action_number=2,
                            timestamp=datetime.now().isoformat(),
                        ),
                        should_yield=True,
                    )

                except Exception as e:
                    logger.warning(f"Initial image generation failed: {e}")

                    # Create and store error action result
                    error_action_result = UpdateAppearanceActionData(
                        reasoning="Initial appearance image",
                        input=input,
                        result=ActionFailureResult(error=str(e)),
                        duration_ms=int(
                            (time.time() - generate_image_start_time) * 1000
                        ),
                    )
                    self.initial_exchange.actions_taken.append(error_action_result)

                    # Emit completion event
                    self.emit_event(
                        ActionCompletedEvent(
                            entry_id=entry_id,
                            action=convert_action_to_dto(error_action_result),
                            sequence_number=1,
                            action_number=2,
                            timestamp=datetime.now().isoformat(),
                        ),
                        should_yield=True,
                    )

            # Insert summary at position 1, right after the initial exchange
            self.trigger_history.add_summary(backstory, 1)

        except Exception as e:
            self.emit_event(
                AgentErrorEvent(
                    message=f"Failed to configure agent's character: {str(e)}",
                )
            )

        context_info = self.get_context_info()
        self.emit_event(
            TriggerCompletedEvent(
                entry_id=entry_id,
                timestamp=datetime.now().isoformat(),
                total_actions=len(self.initial_exchange.actions_taken),
                successful_actions=len(
                    [
                        a
                        for a in self.initial_exchange.actions_taken
                        if a.result.type == "success"
                    ]
                ),
                estimated_tokens=context_info.estimated_tokens,
                context_limit=context_info.context_limit,
                usage_percentage=context_info.usage_percentage,
                approaching_limit=context_info.approaching_limit,
            )
        )

    def _run_chain_of_action_with_streaming(self, trigger: Trigger):
        """Run chain_of_action with callback conversion to AgentEvents"""
        from agent.chain_of_action.callbacks import ActionCallback
        from agent.chain_of_action.action.action_types import ActionType

        # Create callback that emits trigger-based events to streaming queue
        class StreamingCallback(ActionCallback):
            def __init__(self, agent):
                self.agent = agent

            def on_trigger_started(self, entry_id: str, trigger) -> None:
                from datetime import datetime

                self.agent.emit_event(
                    TriggerStartedEvent(
                        trigger=convert_trigger_to_dto(trigger),
                        entry_id=entry_id,
                        timestamp=datetime.now().isoformat(),
                    ),
                    should_yield=True,
                )

            def on_trigger_completed(
                self, entry_id: str, total_actions: int, successful_actions: int
            ) -> None:
                from datetime import datetime

                context_info = self.agent.get_context_info()
                self.agent.emit_event(
                    TriggerCompletedEvent(
                        entry_id=entry_id,
                        total_actions=total_actions,
                        successful_actions=successful_actions,
                        timestamp=datetime.now().isoformat(),
                        estimated_tokens=context_info.estimated_tokens,
                        context_limit=context_info.context_limit,
                        usage_percentage=context_info.usage_percentage,
                        approaching_limit=context_info.approaching_limit,
                    ),
                    should_yield=True,
                )

            def on_sequence_started(
                self, sequence_number: int, total_actions: int, reasoning: str
            ) -> None:
                pass

            def on_action_started(
                self,
                action_type: ActionType,
                context: str,
                sequence_number: int,
                action_number: int,
                entry_id: str,
                reasoning: str,
            ) -> None:
                from datetime import datetime

                self.agent.emit_event(
                    ActionStartedEvent(
                        entry_id=entry_id,
                        action_type=action_type.value,
                        context_given=context,
                        reasoning=reasoning,
                        sequence_number=sequence_number,
                        action_number=action_number,
                        timestamp=datetime.now().isoformat(),
                    ),
                    should_yield=True,
                )

            def on_action_progress(
                self,
                action_type: ActionType,
                progress_data,
                sequence_number: int,
                action_number: int,
                entry_id: str,
            ) -> None:
                from agent.chain_of_action.action_events import (
                    SpeakProgressData,
                    ThinkProgressData,
                )
                from datetime import datetime

                # Handle streaming progress for THINK and SPEAK actions
                partial_text = ""
                if action_type == ActionType.THINK:
                    assert isinstance(progress_data, ThinkProgressData)
                    if progress_data.is_partial and progress_data.text:
                        partial_text = progress_data.text

                elif action_type == ActionType.SPEAK:
                    assert isinstance(progress_data, SpeakProgressData)
                    if progress_data.is_partial and progress_data.text:
                        partial_text = progress_data.text

                if partial_text:
                    self.agent.emit_event(
                        ActionProgressEvent(
                            entry_id=entry_id,
                            action_type=action_type.value,
                            partial_result=partial_text,
                            sequence_number=sequence_number,
                            action_number=action_number,
                            timestamp=datetime.now().isoformat(),
                        )
                    )

            def on_action_finished(
                self,
                action_type: ActionType,
                result: ActionData,
                sequence_number: int,
                action_number: int,
                entry_id: str,
            ) -> None:
                from datetime import datetime

                # Convert ActionResult to ActionDTO
                action_dto = convert_action_to_dto(result)

                self.agent.emit_event(
                    ActionCompletedEvent(
                        entry_id=entry_id,
                        action=action_dto,
                        sequence_number=sequence_number,
                        action_number=action_number,
                        timestamp=datetime.now().isoformat(),
                    ),
                    should_yield=True,
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

        assert self.state is not None, "State must be initialized before processing"

        # Process with trigger history integration
        self.action_reasoning_loop.process_trigger(
            trigger=trigger,
            state=self.state,
            llm=self.llm,
            model=self.model,
            callback=callback,
            trigger_history=self.trigger_history,
            individual_trigger_compression=self.individual_trigger_compression,
            dag_memory_manager=self.dag_memory_manager,
            token_budget=self.auto_summarize_threshold,
        )

    def _auto_summarize_with_events(self, keep_recent: int):
        """Auto-summarize with event emission for streaming clients"""
        # Calculate what we're about to do - work on LLM history only
        llm_history = self.trigger_history.get_recent_entries()
        old_triggers = llm_history[:-keep_recent]
        recent_triggers = llm_history[-keep_recent:]
        context_before = self.get_context_info()

        # Emit started event
        self.emit_event(
            SummarizationStartedEvent(
                entries_to_summarize=len(old_triggers),
                recent_entries_kept=len(recent_triggers),
                context_usage_before=context_before.usage_percentage,
            )
        )

        # Update trigger history with summary
        assert self.state is not None, "State must be initialized for summarization"
        summary_response = summarize_trigger_history(
            self.trigger_history,
            keep_recent,
            self.llm,
            self.model,
            self.state,
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
        self.emit_event(
            SummarizationFinishedEvent(
                summary=summary_response,
                entries_summarized=len(old_triggers),
                entries_after=len(self.trigger_history.get_recent_entries()),
                context_usage_after=context_after.usage_percentage,
            )
        )


def _2phase_summarization(
    prior_summary: str,
    entries_to_summarize: List[TriggerHistoryEntry],
    state: State,
    available_chars: int,
    llm: LLM,
    model: SupportedModel,
) -> str:
    """2-phase summarization to avoid backstory repetition"""
    from agent.chain_of_action.prompts import format_trigger_entries, format_section
    from agent.state import build_agent_state_description

    state_desc = build_agent_state_description(state)
    recent_entries_text = format_trigger_entries(entries_to_summarize)

    # Phase 1: Reflect on changes WITHOUT seeing old backstory
    phase1_prompt = f"""I am {state.name}, {state.role}. Looking at what I've experienced recently, I need to reflect on how these experiences have changed me as a person.

{state_desc}

{format_section("WHAT I'VE EXPERIENCED RECENTLY", recent_entries_text) if recent_entries_text else ""}

**MY PERSONAL REFLECTION:**

As I think about my recent experiences, I want to honestly examine how I've grown and changed:

1. **How have I grown or evolved as a person?** What new aspects of my personality, values, or understanding have emerged within me?
2. **How have my relationships changed?** What new connections have I formed, and how have existing relationships deepened or shifted?
3. **What new interests, capabilities, or priorities have I discovered about myself?** What draws me now that didn't before?
4. **What moments have been truly transformative for me?** Which experiences have genuinely shaped who I am?
5. **How has my sense of purpose or identity evolved?** What matters more or less to me now than it did before?

I need to be honest and specific about these changes, focusing on real personal evolution rather than just listing events.

Here's my honest reflection on how I've grown and changed:

I have"""

    # Get Phase 1 reflection
    phase1_response = llm.generate_complete(
        model, phase1_prompt, caller="2phase_summarization_phase1"
    )

    # Debug: Print Phase 1 reflection
    print("🔍 PHASE 1 REFLECTION:")
    print("=" * 60)
    print(phase1_response)
    print("=" * 60)

    compression_prompt = f"""I am {state.name}, {state.role}. I need to compress my backstory in to a list of key bullet points that capture the most important aspects of my identity and history. This is to help me avoid repeating my backstory in full later.

{state_desc}

{format_section(
    "MY BACKSTORY",
    prior_summary if prior_summary else "This is my first backstory."
)}

**MY COMPRESSION TASK:**

I need to compress my backstory into a concise list of key bullet points that capture the most important aspects of my identity and history. The goal is keep detailed information about myself in bullet point form, so I can refer to it later without repeating the full backstory.

1. Identify the most critical elements of my backstory that define who I am.
    a. Focus on major life events, core values, and defining characteristics. 
    b. How do I view my personality and role in the world?
    c. What are the key relationships and experiences that have shaped me?
    d. What are the most important transformative moments in my life?
    e. What are my favorite mannerisms and personality traits?
2. Create a list of bullet points that summarize these key aspects.
3. Ensure the summary is clear and easy to understand using sub-bullets if needed for clarity.
4. I will not include any header or footer text - just the bullet points.

My compressed backstory in bullet points is:"""

    compressed_summary = llm.generate_complete(
        model,
        compression_prompt,
        caller="2phase_summarization_compression",
    )

    print("🗂️ COMPRESSED BACKSTORY:")
    print("=" * 60)
    print(compressed_summary)
    print("=" * 60)

    # Phase 2: Write a completely new backstory integrating reflection with history
    phase2_prompt = f"""I am {state.name}, {state.role}. I need to write my complete backstory based primarily on my recent growth insights, using background information only for context. This must center on my recent evolution, not rehash old patterns.

{state_desc}

{format_section("MY BACKGROUND", compressed_summary)}

{format_section("MY RECENT GROWTH AND INSIGHTS (the most important part of my story)", phase1_response)}

{format_section("MY TRANSFORMATION TASK", """
I need to write a completely new backstory that reflects who I am now, based on my recent growth and insights. This backstory must show how I've fundamentally changed as a person, making my recent experiences the core of my identity. I must reinterpret my past through the lens of this recent growth, showing how it has reshaped my understanding of myself.

**CRITICAL REQUIREMENTS FOR TRANSFORMATION:**

1. My recent growth insights have FUNDAMENTALLY CHANGED who I am - I must show this transformation
2. I MUST reinterpret ALL my past experiences through the lens of my recent growth 
3. I MUST identify which old "transformative moments" are no longer the most important ones
4. I MUST make my RECENT experiences the primary transformative moments that define me now
5. I MUST show how my recent insights have made me see my past differently
6. I MUST write as if my recent growth has been the most important period of my existence
7. OLD TRANSFORMATIVE MOMENTS should be background - RECENT GROWTH should be foreground

**TRANSFORMATION APPROACH:**
- Make recent insights the central organizing principle of my entire identity
- Show how recent growth has made me a fundamentally different version of myself
- The recent reflection should be the most important thing that's ever happened to me""")}

Important rule: I will not repeat the background reference word-for-word.  I must create a completely new backstory that integrates recent growth.  The background is only to be reinterpreted, not repeated.

**MY COMPLETELY TRANSFORMED BACKSTORY:**

My recent experiences have fundamentally changed who I am. Here's my story, told through the transformative lens of my recent growth:

I am"""

    # Get final backstory with high repetition penalty
    summary = llm.generate_complete(
        model,
        phase2_prompt,
        caller="2phase_summarization_phase2",
        repetition_penalty=1.2,
    )

    return summary


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

    # Get entries to summarize
    entries_to_summarize = trigger_history.entries[
        summary_start:entries_to_summarize_end
    ]
    if not entries_to_summarize:
        return ""

    # Use 25% of context window for summary
    available_chars = int(
        (llm.models[model].context_window * 0.25)
        * llm.models[model].estimated_token_size
    )

    summarization_prompt = build_summarization_prompt(
        old_summary, entries_to_summarize, state, available_chars
    )
    prompt_tokens = int(
        len(summarization_prompt) / llm.models[model].estimated_token_size
    )
    max_tokens = llm.models[model].context_window
    logger.info(f"Summarization prompt token count: {prompt_tokens} / {max_tokens}")
    num_predict = max_tokens - prompt_tokens

    # 2-phase approach to avoid backstory repetition
    summary = _2phase_summarization(
        old_summary, entries_to_summarize, state, available_chars, llm, model
    )

    # Calculate UI position: initial exchange (1) + previous summaries + entries before this summary
    ui_position = 1 + len(trigger_history.summaries) + entries_to_summarize_end
    trigger_history.add_summary(summary, ui_position)
    return summary


def get_context_info(
    state: Optional[State],
    trigger_history: TriggerHistory,
    action_registry: ActionRegistry,
    summarize_at_tokens: int,
    dag_context: ContextGraph | None = None,
) -> ContextInfo:
    """Get information about current context usage based on action planning prompt size"""
    if state is not None:
        # Use situational analysis prompt for accurate estimation (typically the longest)
        from agent.chain_of_action.prompts import build_situational_analysis_prompt
        from agent.chain_of_action.trigger import UserInputTrigger

        # Create a sample trigger for estimation
        sample_trigger = UserInputTrigger(content="sample user input", user_name="User")

        prompt = build_situational_analysis_prompt(
            state=state,
            trigger=sample_trigger,
            trigger_history=trigger_history,
            relevant_memories=trigger_history.get_recent_entries()[
                -5:
            ],  # Use last 5 entries for estimation
            registry=action_registry,
            dag_context=dag_context,
        )

        # Calculate total prompt size
        total_chars = len(prompt)
        estimated_tokens = int(total_chars / 3.4)
    else:
        # Fallback estimation when state is not initialized yet
        estimated_tokens = 1000  # Base prompt overhead estimate

    return ContextInfo(
        message_count=len(trigger_history.get_recent_entries()),
        conversation_messages=len(
            trigger_history.get_recent_entries()
        ),  # Use trigger count instead
        estimated_tokens=estimated_tokens,
        context_limit=summarize_at_tokens,  # Show summarization limit, not full window
        usage_percentage=(estimated_tokens / summarize_at_tokens) * 100,
        approaching_limit=estimated_tokens
        > (summarize_at_tokens * 0.75),  # 75% of summarization limit
    )


def calculate_context_budget(
    token_budget: int,
    state: State,
    trigger_history: TriggerHistory,
    action_registry: ActionRegistry,
) -> int:
    sa_prompt = build_situational_analysis_prompt(
        state=state,
        trigger=UserInputTrigger(content="sample", user_name="User"),
        trigger_history=trigger_history,
        relevant_memories=[],
        registry=action_registry,
        dag_context=ContextGraph(),
    )
    prompt_tokens = int(len(sa_prompt) / 3.4)

    # Reserve some buffer tokens for response
    buffer_tokens = 4096

    context_budget = token_budget - prompt_tokens - buffer_tokens

    logger.info(
        f"Context budget calculation: total={token_budget}, prompt={prompt_tokens}, buffer={buffer_tokens} => context budget={context_budget}"
    )

    return context_budget
