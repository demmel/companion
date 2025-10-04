"""
Agent Event Manager - wraps Agent and manages event streaming
"""

import queue
import threading
from typing import Optional
from agent.core import Agent
from agent.api_types.events import AgentEvent
from agent.chain_of_action.trigger import Trigger
from agent.chain_of_action.trigger_history import TriggerHistory


class AgentEventManager:
    """
    Manages event streaming for an Agent.
    Implements EventEmitter protocol and provides proxy access to Agent methods.
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self.current_client_queue: Optional[queue.Queue[AgentEvent]] = None
        self.client_queue_lock = threading.Lock()

    # EventEmitter protocol implementation (pass-through for now)
    def emit(self, event: AgentEvent, should_yield: bool = False) -> None:
        """Emit an event to the current client queue (if any)"""
        import time

        with self.client_queue_lock:
            if self.current_client_queue:
                self.current_client_queue.put(event)
            # else: drop event (no client connected)

        if should_yield:
            time.sleep(0)  # Yield to allow event to be processed

    # Client queue management
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
