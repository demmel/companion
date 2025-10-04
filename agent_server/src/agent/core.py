"""
Core agent implementation
"""

from datetime import datetime
import time
import threading
import queue
from typing import Optional
from agent.chain_of_action.action_registry import ActionRegistry
from agent.memory.dag_memory_manager import DagMemoryManager
from pydantic import BaseModel

from agent.api_types import (
    ActionProgressEvent,
    SpeakProgressData,
    ThinkProgressData,
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
    BirthTrigger,
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
    ToolCallSuccess,
)
from agent.api_types import (
    ActionCompletedEvent,
    ActionStartedEvent,
    AgentErrorEvent,
    TriggerEvent as AgentEvent,
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
        individual_trigger_compression: bool = True,
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
            self.auto_summarize_threshold = int(self.context_window * 0.75)
        self.enable_image_generation = enable_image_generation
        self.individual_trigger_compression = individual_trigger_compression

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
            self.dag_memory_manager is not None
        ), "Cannot save conversation without DAG memory manager"
        logger.info(
            f"Saving conversation {self.conversation_id} with {len(self.trigger_history)} entries"
        )
        self.persistence.save_conversation(
            self.conversation_id,
            self.state,
            self.trigger_history,
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
        self.trigger_history = agent_data.trigger_history
        self.dag_memory_manager = agent_data.dag_memory_manager

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

    def clear_client_queue(self, client_queue: queue.Queue[AgentEvent]) -> None:
        """Clear the current client queue if it matches the given queue"""
        with self.client_queue_lock:
            if self.current_client_queue == client_queue:
                self.current_client_queue = None
            # else: do nothing (different client)

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
        return get_context_info(
            state=self.state,
            trigger_history=self.trigger_history,
            action_registry=self.action_reasoning_loop.registry,
            summarize_at_tokens=self.auto_summarize_threshold,
            dag_memory_manager=self.dag_memory_manager,
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
                # Use action-based reasoning with callback conversion
                self._run_chain_of_action_with_streaming(trigger)

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

        birth_trigger = BirthTrigger(
            content=trigger.content,
            user_name=trigger.user_name,
            timestamp=trigger.timestamp,
            image_paths=trigger.image_paths,
        )

        self.initial_exchange = TriggerHistoryEntry(
            trigger=birth_trigger,
            situational_context="",  # No situational context for initial exchange
            actions_taken=[],
        )
        self.trigger_history.entries.append(self.initial_exchange)

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

            from agent.memory import (
                DagMemoryManager,
            )

            self.dag_memory_manager = DagMemoryManager.create(
                initial_state=self.state,
                token_budget=self.auto_summarize_threshold,
                action_registry=self.action_reasoning_loop.registry,
                trigger_history=self.trigger_history,
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
                start_timestamp=datetime.now(),
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
                        start_timestamp=datetime.now(),
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
                        start_timestamp=datetime.now(),
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

        except Exception as e:
            self.emit_event(
                AgentErrorEvent(
                    message=f"Failed to configure agent's character: {str(e)}",
                )
            )

        # Compress the initial exchange before calculating context info
        from agent.chain_of_action.reasoning_loop import _compress_trigger_entry

        assert self.state is not None, "State must be initialized before compression"
        _compress_trigger_entry(
            self.initial_exchange,
            self.state,
            self.llm,
            self.model,
        )

        # Process initial exchange memories if DAG enabled
        if self.dag_memory_manager:
            self.dag_memory_manager.postprocess_trigger(
                trigger=self.initial_exchange,
                state=self.state,
                llm=self.llm,
                model=self.model,
                token_budget=self.auto_summarize_threshold,
                action_registry=self.action_reasoning_loop.registry,
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
        assert self.dag_memory_manager is not None, "DAG memory must be initialized"

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


def get_context_info(
    state: Optional[State],
    trigger_history: TriggerHistory,
    action_registry: ActionRegistry,
    summarize_at_tokens: int,
    dag_memory_manager: Optional[DagMemoryManager] = None,
) -> ContextInfo:
    """Get information about current context usage based on action planning prompt size"""
    if state is not None and dag_memory_manager is not None:
        # Use situational analysis prompt for accurate estimation (typically the longest)
        from agent.chain_of_action.prompts import build_situational_analysis_prompt
        from agent.chain_of_action.trigger import UserInputTrigger

        # Create a sample trigger for estimation
        sample_trigger = UserInputTrigger(content="sample user input", user_name="User")

        prompt = build_situational_analysis_prompt(
            state=state,
            trigger=sample_trigger,
            trigger_history=trigger_history,
            registry=action_registry,
            dag_memory_manager=dag_memory_manager,
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
