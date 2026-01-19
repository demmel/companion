"""
Storage module for trigger history storage.

- SQLite + SQLModel for structured data (triggers, actions)
- ChromaDB for embedding similarity search
"""

from agent.storage.models import TriggerEntryTable, ActionTable
from agent.storage.trigger_history import TriggerHistory, create_trigger_history
from agent.storage.interface import ITriggerHistory
from agent.storage.mirrored_trigger_history import MirroredTriggerHistory
from agent.storage.serializers import (
    TriggerSerializer,
    ActionSerializer,
    compress_json,
    decompress_json,
)

__all__ = [
    # Models
    "TriggerEntryTable",
    "ActionTable",
    # Main implementation
    "ITriggerHistory",
    "TriggerHistory",
    "MirroredTriggerHistory",
    "create_trigger_history",
    # Serializers
    "TriggerSerializer",
    "ActionSerializer",
    "compress_json",
    "decompress_json",
]
