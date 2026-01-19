"""
Storage module for trigger history storage.

- SQLite + SQLModel for structured data (triggers, actions)
- ChromaDB for embedding similarity search
"""

from agent.storage.models import TriggerEntryTable, ActionTable
from agent.storage.trigger_history_sqlite import TriggerHistorySQLite, create_trigger_history_sqlite
from agent.storage.serializers import (
    TriggerSerializer,
    ActionSerializer,
    compress_json,
    decompress_json,
)
from agent.storage.migration import (
    TriggerHistoryMigrator,
    migrate_conversation_directory,
    load_trigger_history,
    export_sqlite_to_json,
)

__all__ = [
    # Models
    "TriggerEntryTable",
    "ActionTable",
    # Main implementation
    "TriggerHistorySQLite",
    "create_trigger_history_sqlite",
    # Serializers
    "TriggerSerializer",
    "ActionSerializer",
    "compress_json",
    "decompress_json",
    # Migration
    "TriggerHistoryMigrator",
    "migrate_conversation_directory",
    "load_trigger_history",
    "export_sqlite_to_json",
]
