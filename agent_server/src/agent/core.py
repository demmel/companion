"""
Core agent implementation
"""

from datetime import datetime
import time
import threading
from typing import Optional
from agent.chain_of_action.action.actions.speak_action import SpeakProgressData
from agent.event_emitter import EventEmitter
from agent.chain_of_action.action_registry import ActionRegistry
from agent.llm.models import ModelConfig
from agent.memory.memory import IMemory
from pydantic import BaseModel

from agent.api_types.actions import (
    convert_action_to_dto,
)
from agent.api_types.triggers import (
    convert_trigger_to_dto,
)
from agent.api_types.timeline import (
    convert_trigger_history_entry_to_dto,
)
from agent.chain_of_action.action.action_data import (
    ActionData,
    ThinkActionData,
    UpdateAppearanceActionData,
)
from agent.chain_of_action.action.actions.think_action import (
    ThinkInput,
    ThinkOutput,
    ThinkProgressData,
)
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
from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.storage import ITriggerHistory
from agent.llm import LLM
from agent.state import State

from agent.chain_of_action.reasoning_loop import ActionBasedReasoningLoop
from agent.conversation_persistence import ConversationPersistence, ConversationContext
from agent.state import (
    State,
)
from agent.types import ToolCallError

from .llm import LLM

from .types import (
    ImageGenerationToolContent,
    ToolCallSuccess,
)
from agent.api_types.events import (
    ActionProgressEvent,
    ActionCompletedEvent,
    ActionStartedEvent,
    AgentErrorEvent,
    AgentEvent,
    TriggerCompletedEvent,
    TriggerStartedEvent,
)
from agent.tts import TTSService
from agent.tts.providers import ChatterboxProvider
from agent.paths import agent_paths
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
        llm: LLM,
        model_config: ModelConfig,
        event_emitter: EventEmitter,
        conversation_context: ConversationContext,
        enable_image_generation: bool = True,
        enable_tts: bool = True,
        individual_trigger_compression: bool = True,
        use_individual_memory_formatting: bool = True,
        auto_summarize_threshold: Optional[int] = None,
    ):
        self.llm = llm
        self.event_emitter = event_emitter
        self.model_config = model_config

        # Set summarization threshold
        if auto_summarize_threshold is not None:
            self.auto_summarize_threshold = auto_summarize_threshold
        else:
            # Default to 70% of context window of situational analysis model
            self.auto_summarize_threshold = int(
                self.llm.models()[
                    model_config.situational_analysis_model
                ].context_window
                * 0.7
            )
        self.enable_image_generation = enable_image_generation
        self.enable_tts = enable_tts
        self.individual_trigger_compression = individual_trigger_compression
        self.use_individual_memory_formatting = use_individual_memory_formatting

        # Conversation context - provided by factory methods
        self.persistence = conversation_context.persistence
        self.conversation_id = conversation_context.conversation_id
        self.trigger_history: ITriggerHistory = conversation_context.trigger_history

        # Initialize reasoning system
        self.action_reasoning_loop = ActionBasedReasoningLoop(
            enable_image_generation=enable_image_generation,
        )

        # Initialize state and memory from context (may be None for new conversations)
        self.state: Optional[State] = conversation_context.state
        self.memory: Optional[IMemory] = conversation_context.memory
        self.memory_context: str = ""

        # TTS service (initialized lazily if enabled)
        self.tts_service: Optional[TTSService] = None
        if enable_tts:
            self._init_tts_service()

        # Auto-wakeup timer functionality
        self.auto_wakeup_enabled = False
        self.auto_wakeup_timer: Optional[threading.Timer] = None
        self.processing_condition = (
            threading.Condition()
        )  # Condition for processing coordination
        self.is_processing = False
        self.wakeup_delay_seconds = 5 * 60  # 5 minutes

    @classmethod
    def new(
        cls,
        llm: LLM,
        model_config: ModelConfig,
        event_emitter: EventEmitter,
        enable_image_generation: bool = True,
        enable_tts: bool = True,
        individual_trigger_compression: bool = True,
        use_individual_memory_formatting: bool = True,
        auto_summarize_threshold: Optional[int] = None,
    ) -> "Agent":
        """Create agent with fresh conversation."""
        persistence = ConversationPersistence()
        context = persistence.new_conversation()
        return cls(
            llm=llm,
            model_config=model_config,
            event_emitter=event_emitter,
            conversation_context=context,
            enable_image_generation=enable_image_generation,
            enable_tts=enable_tts,
            individual_trigger_compression=individual_trigger_compression,
            use_individual_memory_formatting=use_individual_memory_formatting,
            auto_summarize_threshold=auto_summarize_threshold,
        )

    @classmethod
    def load(
        cls,
        conversation_id: str,
        llm: LLM,
        model_config: ModelConfig,
        event_emitter: EventEmitter,
        enable_image_generation: bool = True,
        enable_tts: bool = True,
        individual_trigger_compression: bool = True,
        use_individual_memory_formatting: bool = True,
        auto_summarize_threshold: Optional[int] = None,
    ) -> "Agent":
        """Create agent by loading existing conversation."""
        persistence = ConversationPersistence()
        context = persistence.load_conversation(
            conversation_id, use_individual_memory_formatting
        )
        return cls(
            llm=llm,
            model_config=model_config,
            event_emitter=event_emitter,
            conversation_context=context,
            enable_image_generation=enable_image_generation,
            enable_tts=enable_tts,
            individual_trigger_compression=individual_trigger_compression,
            use_individual_memory_formatting=use_individual_memory_formatting,
            auto_summarize_threshold=auto_summarize_threshold,
        )

    def get_trigger_history(self) -> ITriggerHistory:
        """Get the current trigger history"""
        return self.trigger_history

    def close(self) -> None:
        """Close agent resources to release file handles."""
        # Stop TTS service if running
        if self.tts_service is not None:
            self.tts_service.stop()
            self.tts_service = None

        # Close trigger history (releases SQLite and ChromaDB handles)
        self.trigger_history.close()

    def _init_tts_service(self) -> None:
        """Initialize the TTS service for audio generation."""
        reference_audio = agent_paths.get_tts_reference_audio()
        output_dir = agent_paths.get_generated_audio_dir()

        if reference_audio is None:
            logger.info("TTS not configured (set TTS_REFERENCE_AUDIO in .env)")
            return

        if not reference_audio.exists():
            logger.warning(f"TTS reference audio not found: {reference_audio}")
            logger.warning("TTS service will not be available")
            return

        try:
            provider = ChatterboxProvider(device="cuda")
            self.tts_service = TTSService(
                provider=provider,
                reference_audio=reference_audio,
                output_dir=output_dir,
                llm=self.llm,
                tts_rewrite_model=self.model_config.tts_rewrite_model,
            )
            self.tts_service.start()
            logger.info("TTS service initialized and started")
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            self.tts_service = None

    def save_conversation(self, title: Optional[str] = None) -> str:
        """Save the current conversation state to disk.

        Note: Trigger entries are persisted immediately on add_entry().
        This method saves state and memory.

        Returns:
            The conversation ID
        """
        logger.info(
            f"save_conversation called: conversation_id={self.conversation_id}, state={self.state is not None}"
        )

        assert (
            self.state is not None
        ), "Cannot save conversation without initialized state"
        assert self.memory is not None, "Cannot save conversation without memory"

        logger.info(
            f"Saving conversation {self.conversation_id} with {len(self.trigger_history)} entries"
        )
        self.persistence.save_conversation(
            self.conversation_id,
            self.state,
            self.trigger_history,
            self.memory,
        )
        logger.info(f"Successfully saved conversation {self.conversation_id}")
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
        """Emit an event via the event emitter"""
        self.event_emitter.emit(event, should_yield)

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
            memory_context=self.memory_context,
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

            # Save conversation state after each turn
            logger.info("Saving conversation state after chat stream")
            self.save_conversation()

            self.llm.log_stats_summary()

            total_time = time.time() - start_time
            logger.debug(f"Total chat_stream time: {total_time:.3f}s")

        except Exception as e:
            logger.error(f"Error occurred during chat_stream: {e}")
            import traceback
            from agent.llm.interface import (
                LLMAuthenticationError,
                LLMInsufficientCreditsError,
                LLMRateLimitError,
                LLMAPIError,
                LLMError,
            )

            traceback.print_exc()

            # Determine error type and create appropriate message
            if isinstance(e, LLMAuthenticationError):
                message = (
                    "Authentication failed. Please check your API key configuration."
                )
            elif isinstance(e, LLMInsufficientCreditsError):
                message = "Insufficient credits or quota exceeded. Please add credits to your account."
            elif isinstance(e, LLMRateLimitError):
                message = (
                    "Rate limit exceeded. Please wait a few moments and try again."
                )
            elif isinstance(e, LLMAPIError):
                message = f"LLM API error: {str(e)}"
            elif isinstance(e, LLMError):
                message = f"LLM error: {str(e)}"
            else:
                message = f"Internal error: {str(e)}"

            # Emit error event to trigger buffer clearing
            self.emit_event(
                AgentErrorEvent(
                    message=message,
                )
            )
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
        self.trigger_history.add_entry(self.initial_exchange)

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
            # Load model config for state initialization
            from agent.config import Config

            model_config = Config.get_model_config()
            self.state, backstory = derive_initial_state_from_message(
                trigger.content,
                self.llm,
                model_config.state_initialization_model,
                trigger.get_images(),
            )

            # Store initial state on birth trigger for replay support
            birth_trigger.initial_state = self.state

            from agent.memory.dag import (
                DagMemoryManager,
            )

            self.memory = DagMemoryManager.create(
                trigger_history=self.trigger_history,
                use_individual_formatting=self.use_individual_memory_formatting,
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
                        model_config.visual_action_model,
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
                            model_config.visual_action_model,
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
        from agent.config import Config

        assert self.state is not None, "State must be initialized before compression"
        model_config = Config.get_model_config()
        _compress_trigger_entry(
            self.initial_exchange,
            self.state,
            self.llm,
            model_config.trigger_compression_model,
        )

        # Process initial exchange memories if memory enabled
        if self.memory:
            self.memory.store(
                self.initial_exchange,
                self.state,
                self.llm,
                model_config.memory_formation_model,
            )

        # Persist all modifications to the initial exchange entry
        self.trigger_history.update_entry(self.initial_exchange)

        context_info = self.get_context_info()
        self.emit_event(
            TriggerCompletedEvent(
                entry=convert_trigger_history_entry_to_dto(self.initial_exchange),
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

            def on_trigger_completed(self, entry: TriggerHistoryEntry) -> None:
                context_info = self.agent.get_context_info()
                self.agent.emit_event(
                    TriggerCompletedEvent(
                        entry=convert_trigger_history_entry_to_dto(entry),
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
                from agent.chain_of_action.action.action_data import SpeakActionData

                # Convert ActionResult to ActionDTO
                try:
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

                    # Queue TTS rendering for successful speak actions
                    # Use 0-indexed action_index to match frontend and API conventions
                    if (
                        action_type == ActionType.SPEAK
                        and result.result.type == "success"
                        and self.agent.tts_service is not None
                        and isinstance(result, SpeakActionData)
                    ):
                        action_index = (
                            action_number - 1
                        )  # Convert 1-indexed to 0-indexed
                        action_id = f"{entry_id}_{action_index}"
                        text = result.result.content.response
                        tone = result.input.tone
                        self.agent.tts_service.queue_render(action_id, text, tone)
                        logger.debug(f"Queued TTS render for speak action {action_id}")

                    # Queue TTS rendering for successful think actions
                    # Use 0-indexed action_index to match frontend and API conventions
                    if (
                        action_type == ActionType.THINK
                        and result.result.type == "success"
                        and self.agent.tts_service is not None
                        and isinstance(result, ThinkActionData)
                    ):
                        action_index = (
                            action_number - 1
                        )  # Convert 1-indexed to 0-indexed
                        action_id = f"{entry_id}_{action_index}"
                        text = result.result.content.thoughts
                        self.agent.tts_service.queue_render(action_id, text, None)
                        logger.debug(f"Queued TTS render for think action {action_id}")

                except Exception as e:
                    # If DTO conversion fails (e.g., for failed visual actions),
                    # emit an error event instead to ensure buffer clearing
                    logger.error(
                        f"Failed to convert action to DTO: {e}. Action type: {action_type}, Result type: {result.result.type}"
                    )
                    self.agent.emit_event(
                        AgentErrorEvent(
                            message=f"Failed to process action result: {str(e)}",
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

        assert self.state is not None, "State must be initialized before processing"
        assert self.memory is not None, "Memory must be initialized"

        # Process with trigger history integration
        _, memory_context = self.action_reasoning_loop.process_trigger(
            trigger=trigger,
            state=self.state,
            llm=self.llm,
            callback=callback,
            trigger_history=self.trigger_history,
            individual_trigger_compression=self.individual_trigger_compression,
            token_budget=self.auto_summarize_threshold,
            memory=self.memory,
            previous_memory_context=self.memory_context,
        )

        # Update agent's memory context
        self.memory_context = memory_context


def get_context_info(
    state: Optional[State],
    trigger_history: ITriggerHistory,
    action_registry: ActionRegistry,
    summarize_at_tokens: int,
    memory_context: str,
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
            registry=action_registry,
            formatted_memory_context=memory_context,
        )

        # Calculate total prompt size
        total_chars = len(prompt)
        estimated_tokens = int(total_chars / 3.4)
    else:
        # Fallback estimation when state is not initialized yet
        estimated_tokens = 1000  # Base prompt overhead estimate

    return ContextInfo(
        message_count=trigger_history.get_entry_count(),
        conversation_messages=trigger_history.get_entry_count(),
        estimated_tokens=estimated_tokens,
        context_limit=summarize_at_tokens,  # Show summarization limit, not full window
        usage_percentage=(estimated_tokens / summarize_at_tokens) * 100,
        approaching_limit=estimated_tokens
        > (summarize_at_tokens * 0.75),  # 75% of summarization limit
    )
