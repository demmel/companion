"""
Event emitter abstraction for Agent events
"""

from typing import Protocol
from agent.api_types.events import AgentEvent


class EventEmitter(Protocol):
    """Protocol for emitting agent events to external consumers"""

    def emit(self, event: AgentEvent, should_yield: bool = False) -> None:
        """
        Emit an event to external consumers

        Args:
            event: The event to emit
            should_yield: If True, yield execution to allow event processing
        """
        ...
