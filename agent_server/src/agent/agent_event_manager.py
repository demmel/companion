"""
Agent Event Manager - wraps Agent and manages event streaming
"""

import queue
import threading
from typing import Optional, List, Tuple
from agent.core import Agent
from agent.api_types.events import AgentEvent, EventEnvelope
from agent.chain_of_action.trigger import Trigger
from agent.chain_of_action.trigger_history import TriggerHistory


class AgentEventManager:
    """
    Manages event streaming for an Agent.
    Implements EventEmitter protocol and provides proxy access to Agent methods.
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self.current_client_queue: Optional[queue.Queue[EventEnvelope]] = None
        self.client_queue_lock = threading.Lock()

        # Event buffering for current trigger
        self.current_trigger_id: Optional[str] = None
        self.event_sequence_counter: int = 0
        self.event_buffer: List[EventEnvelope] = []  # Buffered envelopes
        self.buffer_lock = threading.Lock()

    # EventEmitter protocol implementation with buffering and sequencing
    def emit(self, event: AgentEvent, should_yield: bool = False) -> None:
        """Emit an event with sequence tracking and buffering"""
        import time

        with self.buffer_lock:
            # Track trigger lifecycle
            if event.type == "trigger_started":
                # New trigger starting - reset sequence counter and buffer
                self.current_trigger_id = event.entry_id
                self.event_sequence_counter = 0
                self.event_buffer = []

            # Assign current sequence number to this event
            event_sequence = self.event_sequence_counter

            # Increment for next event
            self.event_sequence_counter += 1

            # Create envelope with sequence and trigger_id
            envelope = EventEnvelope(
                event_sequence=event_sequence,
                trigger_id=self.current_trigger_id or "",
                event=event,
            )

            # Buffer the envelope
            self.event_buffer.append(envelope)

            # Check if trigger completed
            if event.type == "trigger_completed":
                # Trigger done - clear buffer (events already sent to client)
                self.current_trigger_id = None
                self.event_buffer = []
                self.event_sequence_counter = 0

        # TODO Phase 4: Remove queue entirely, send envelopes via hydration/streaming
        # For now, keep backwards compatibility with queue
        with self.client_queue_lock:
            if self.current_client_queue:
                self.current_client_queue.put(envelope)

        if should_yield:
            time.sleep(0)  # Yield to allow event to be processed

    # Client queue management
    def set_client_queue(self, client_queue: queue.Queue[EventEnvelope]) -> None:
        """Set the current client queue (replaces existing client)"""
        with self.client_queue_lock:
            self.current_client_queue = client_queue

    def clear_client_queue(self, client_queue: queue.Queue[EventEnvelope]) -> None:
        """Clear the current client queue if it matches the given queue"""
        with self.client_queue_lock:
            if self.current_client_queue == client_queue:
                self.current_client_queue = None
            # else: do nothing (different client)

    # Proxy methods to Agent's public interface
    def chat_stream(self, trigger: Trigger) -> None:
        """Process a trigger and stream events"""
        self.agent.chat_stream(trigger)

    def get_trigger_history(self) -> TriggerHistory:
        """Get the current trigger history"""
        return self.agent.get_trigger_history()

    def get_context_info(self):
        """Get information about current context usage"""
        return self.agent.get_context_info()

    def set_auto_wakeup_enabled(self, enabled: bool) -> None:
        """Enable or disable auto-wakeup timer"""
        self.agent.set_auto_wakeup_enabled(enabled)

    def get_auto_wakeup_enabled(self) -> bool:
        """Get current auto-wakeup enabled state"""
        return self.agent.get_auto_wakeup_enabled()

    def save_conversation(self, title: Optional[str] = None) -> Optional[str]:
        """Save the current conversation to disk"""
        return self.agent.save_conversation(title)

    def load_conversation(self, conversation_id: str):
        """Load a conversation from disk by its ID"""
        self.agent.load_conversation(conversation_id)

    # Expose agent state properties
    @property
    def state(self):
        return self.agent.state

    @property
    def auto_save(self):
        return self.agent.auto_save

    @auto_save.setter
    def auto_save(self, value: bool):
        self.agent.auto_save = value

    @property
    def wakeup_delay_seconds(self):
        return self.agent.wakeup_delay_seconds
