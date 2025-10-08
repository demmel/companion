"""
Agent Event Manager - wraps Agent and manages event streaming
"""

import queue
import threading
from typing import Optional, List
from agent.core import Agent
from agent.api_types.events import AgentEvent, EventEnvelope, AgentServerEvent
from agent.chain_of_action.trigger import Trigger
from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry


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

    def get_hydration_events(
        self,
        last_trigger_id: Optional[str] = None,
        last_event_sequence: Optional[int] = None,
    ) -> List[AgentServerEvent]:
        """
        Get events for hydration based on client's last known state.

        Cases:
        1. trigger_id matches current + event_sequence: Return buffered events after that sequence
        2. No trigger_id provided: Return last 3 complete triggers from history + current buffer
        3. trigger_id doesn't match current: Return all triggers since (and including) that one from history + current buffer

        Returns a list of AgentServerEvent (HydrationResponse | EventEnvelope).
        """
        from agent.api_types.events import HydrationResponse
        from agent.api_types.timeline import build_timeline_page

        with self.buffer_lock:
            # Determine what to send based on client state
            trigger_history = self.agent.get_trigger_history()
            all_entries = trigger_history.get_all_entries()

            # Params: pagination params for build_timeline_page and which buffer events to send
            page_size: int
            after_index: Optional[int]
            buffer_filter_sequence: int  # only send events > this sequence

            # Determine case and set params
            if last_trigger_id is None:
                # Case 2: No trigger_id - send last 3 complete triggers + full buffer
                page_size = 3
                after_index = None  # Default to last page
                buffer_filter_sequence = -1  # Send all buffer
            elif last_trigger_id == self.current_trigger_id:
                # Case 1: Client caught up to current trigger - no history, filtered buffer
                page_size = 0  # No history
                after_index = None  # No history
                buffer_filter_sequence = last_event_sequence or -1
            else:
                # Case 3: trigger_id provided - find it in history
                start_index = None
                for i, entry in enumerate(all_entries):
                    if entry.entry_id == last_trigger_id:
                        start_index = i
                        break

                if start_index is None:
                    raise ValueError(f"Invalid last_trigger_id: {last_trigger_id}")

                # Found it - send all entries from there onwards (including matched) + full buffer
                page_size = len(all_entries) - start_index
                after_index = start_index
                buffer_filter_sequence = -1

            # Execute: build response based on params
            result: List[AgentServerEvent] = []

            # Add HydrationResponse if we have historical entries to send
            if page_size > 0:
                timeline_entries, pagination = build_timeline_page(
                    all_entries, page_size=page_size, after_index=after_index
                )
                result.append(
                    HydrationResponse(entries=timeline_entries, pagination=pagination)
                )

            # Add filtered buffer events
            result.extend(
                env
                for env in self.event_buffer
                if env.event_sequence > buffer_filter_sequence
            )

            return result

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
