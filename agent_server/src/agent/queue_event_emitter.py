"""
Queue-based event emitter implementation
"""

import time
import queue
import threading
from typing import Optional
from agent.api_types.events import AgentEvent


class QueueEventEmitter:
    """Event emitter that sends events to a queue (for WebSocket communication)"""

    def __init__(self):
        self.current_client_queue: Optional[queue.Queue[AgentEvent]] = None
        self.client_queue_lock = threading.Lock()

    def emit(self, event: AgentEvent, should_yield: bool = False) -> None:
        """Emit an event to the current client queue (if any)"""
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
