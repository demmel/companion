"""
SQLModel table definitions for trigger history storage.

These tables store trigger history data with compressed JSON blobs for
efficient storage and fast incremental saves.
"""

from datetime import datetime
from sqlmodel import SQLModel, Field


class TriggerEntryTable(SQLModel, table=True):
    """
    SQLite table for trigger history entries.

    Stores the core trigger entry data with polymorphic trigger as compressed blob.
    """

    __tablename__ = "trigger_entries"  # type: ignore

    id: str = Field(primary_key=True)  # entry_id (timestamp-based)
    timestamp: datetime = Field(index=True)
    end_timestamp: datetime | None = None
    situational_context: str  # Usually short, keep as TEXT

    # Nullable - only populated after summarization
    compressed_summary: str | None = None

    # Polymorphic trigger - type for queries, blob for data
    trigger_type: str = Field(index=True)  # "birth", "user_input", "wakeup"
    trigger_blob: bytes  # zlib-compressed JSON

    created_at: datetime = Field(default_factory=datetime.now)


class ActionTable(SQLModel, table=True):
    """
    SQLite table for actions within trigger entries.

    Each action is stored with indexed fields for queries and compressed blobs
    for variable-length JSON data.
    """

    __tablename__ = "actions"  # type: ignore

    id: int | None = Field(primary_key=True, default=None)
    trigger_entry_id: str = Field(foreign_key="trigger_entries.id", index=True)
    sequence_order: int  # Order within entry

    # Indexed fields for queries
    action_type: str = Field(index=True)  # ActionType enum value
    start_timestamp: datetime
    duration_ms: float

    # Variable-length fields as compressed blobs
    reasoning: str  # Usually 1-3 sentences, keep as TEXT
    input_blob: bytes  # zlib-compressed JSON
    result_type: str  # "success" or "failure" (for filtering)
    result_blob: bytes  # zlib-compressed JSON


# Note: Embeddings are stored in ChromaDB, not SQLite.
# This provides efficient vector similarity search without raw SQL.
