"""
SQLite-backed TriggerHistory implementation.

Query-based design - no caching, databases are the source of truth.
- SQLite + SQLModel for structured data (triggers, actions)
- ChromaDB for embedding similarity search
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator

from agent.storage.interface import ITriggerHistory
from sqlmodel import Session, SQLModel, col, create_engine, select
from sqlalchemy import Engine

from agent.chain_of_action.trigger import Trigger
from agent.chain_of_action.action.action_data import ActionData
from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.storage.models import TriggerEntryTable, ActionTable
from agent.storage.serializers import TriggerSerializer, ActionSerializer


logger = logging.getLogger(__name__)


class TriggerHistory(ITriggerHistory):
    """
    SQLite + ChromaDB backed trigger history with query-based access.

    No caching - queries databases directly for all operations.
    Uses ChromaDB for efficient embedding similarity search.

    Lifecycle:
        This class is designed for one-conversation-per-process usage.
        ChromaDB's PersistentClient lacks a close() method (known upstream issue:
        https://github.com/chroma-core/chroma/issues/5868), so file handles are
        only released when the process exits. This is fine for production but
        causes issues in tests that create/destroy instances in temp directories.
        Use TemporaryDirectory(ignore_cleanup_errors=True) in tests.
    """

    def __init__(self, db_path: str, chroma_path: str | None = None):
        """
        Initialize SQLite-backed trigger history.

        Args:
            db_path: Path to the SQLite database file.
            chroma_path: Path to ChromaDB storage directory.
                        Defaults to same directory as db_path.
        """
        self._db_path = Path(db_path)

        # SQLite for structured data
        self._engine: Engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )

        # ChromaDB for embeddings
        if chroma_path is None:
            chroma_path = str(self._db_path.parent / "chroma")

        from agent.storage.chroma import ChromaClient

        self._chroma_client = ChromaClient(path=chroma_path)
        self._embeddings_collection = self._chroma_client.client().get_or_create_collection(
            name="trigger_embeddings",
            metadata={"hnsw:space": "cosine"},
        )

    def initialize_db(self) -> None:
        """Create database tables if they don't exist."""
        SQLModel.metadata.create_all(self._engine)
        logger.info(f"Initialized SQLite database at {self._db_path}")

    def dispose(self) -> None:
        """Dispose of database connections. See class docstring for ChromaDB limitations."""
        self._engine.dispose()

    # ===================
    # Write Operations
    # ===================

    def add_entry(self, entry: TriggerHistoryEntry) -> None:
        """Add a new trigger history entry to the database."""
        with Session(self._engine) as session:
            self._insert_entry(session, entry)
            session.commit()

        # Add embedding to ChromaDB if present
        if entry.embedding_vector:
            self._embeddings_collection.add(
                ids=[entry.entry_id],
                embeddings=[entry.embedding_vector],
            )

    def update_entry(self, entry: TriggerHistoryEntry) -> None:
        """Update an existing entry in the database."""
        with Session(self._engine) as session:
            self._update_entry(session, entry)
            session.commit()

        # Update embedding in ChromaDB
        if entry.embedding_vector:
            self._embeddings_collection.upsert(
                ids=[entry.entry_id],
                embeddings=[entry.embedding_vector],
            )

    def _insert_entry(self, session: Session, entry: TriggerHistoryEntry) -> None:
        """Insert a new entry into SQLite."""
        trigger_row = TriggerEntryTable(
            id=entry.entry_id,
            timestamp=entry.timestamp,
            end_timestamp=entry.end_timestamp,
            situational_context=entry.situational_context,
            compressed_summary=entry.compressed_summary,
            trigger_type=TriggerSerializer.get_type(entry.trigger),
            trigger_blob=TriggerSerializer.to_blob(entry.trigger),
        )
        session.add(trigger_row)

        for seq, action in enumerate(entry.actions_taken):
            action_row = ActionSerializer.to_row(action, entry.entry_id, seq)
            session.add(action_row)

    def _update_entry(self, session: Session, entry: TriggerHistoryEntry) -> None:
        """Update an existing entry in SQLite."""
        trigger_row = session.get(TriggerEntryTable, entry.entry_id)
        if trigger_row:
            trigger_row.end_timestamp = entry.end_timestamp
            trigger_row.situational_context = entry.situational_context
            trigger_row.compressed_summary = entry.compressed_summary
            trigger_row.trigger_type = TriggerSerializer.get_type(entry.trigger)
            trigger_row.trigger_blob = TriggerSerializer.to_blob(entry.trigger)
            session.add(trigger_row)

        # Delete and re-insert actions
        action_stmt = select(ActionTable).where(
            ActionTable.trigger_entry_id == entry.entry_id
        )
        old_actions = session.exec(action_stmt).all()
        for old_action in old_actions:
            session.delete(old_action)

        for seq, action in enumerate(entry.actions_taken):
            action_row = ActionSerializer.to_row(action, entry.entry_id, seq)
            session.add(action_row)

    # ===================
    # Read Operations
    # ===================

    def get_entry_by_id(self, entry_id: str) -> TriggerHistoryEntry:
        """Get a single entry by ID. Raises KeyError if not found."""
        with Session(self._engine) as session:
            row = session.get(TriggerEntryTable, entry_id)
            if not row:
                raise KeyError(f"Entry not found: {entry_id}")
            return self._row_to_entry(session, row)

    def get_entry_count(self) -> int:
        """Get total number of entries using COUNT(*)."""
        from sqlalchemy import func

        with Session(self._engine) as session:
            stmt = select(func.count()).select_from(TriggerEntryTable)
            count = session.exec(stmt).one()
            return count

    def get_first_entry(self) -> TriggerHistoryEntry | None:
        """Get the first (oldest) trigger entry."""
        with Session(self._engine) as session:
            stmt = (
                select(TriggerEntryTable)
                .order_by(col(TriggerEntryTable.timestamp))
                .limit(1)
            )
            row = session.exec(stmt).first()
            return self._row_to_entry(session, row) if row else None

    def get_last_entry(self) -> TriggerHistoryEntry | None:
        """Get the last (most recent) trigger entry."""
        with Session(self._engine) as session:
            stmt = (
                select(TriggerEntryTable)
                .order_by(col(TriggerEntryTable.timestamp).desc())
                .limit(1)
            )
            row = session.exec(stmt).first()
            return self._row_to_entry(session, row) if row else None

    def __len__(self) -> int:
        """Return the total number of entries."""
        return self.get_entry_count()

    def iter_entries(self, reverse: bool, start: int) -> Iterator[TriggerHistoryEntry]:
        """
        Iterate over entries with optional reverse order and start position.

        Args:
            reverse: If True, iterate from newest to oldest
            start: Starting position (0-indexed from beginning if not reverse,
                   or from end if reverse)

        Yields:
            TriggerHistoryEntry objects in batches to avoid loading all into memory
        """

        BATCH_SIZE = 100
        offset = start

        while True:
            with Session(self._engine) as session:
                if reverse:
                    stmt = (
                        select(TriggerEntryTable)
                        .order_by(col(TriggerEntryTable.timestamp).desc())
                        .offset(offset)
                        .limit(BATCH_SIZE)
                    )
                else:
                    stmt = (
                        select(TriggerEntryTable)
                        .order_by(col(TriggerEntryTable.timestamp))
                        .offset(offset)
                        .limit(BATCH_SIZE)
                    )

                rows = session.exec(stmt).all()
                if not rows:
                    break

                for row in rows:
                    yield self._row_to_entry(session, row)

            offset += BATCH_SIZE

    def get_entry_index(self, entry_id: str) -> int:
        """
        Get the 0-based index position of an entry in chronological order.

        Args:
            entry_id: The entry ID to find

        Returns:
            The index position (0-indexed)

        Raises:
            KeyError: If entry not found
        """
        from sqlalchemy import func

        with Session(self._engine) as session:
            # First get the entry to find its timestamp
            entry_row = session.get(TriggerEntryTable, entry_id)
            if not entry_row:
                raise KeyError(f"Entry not found: {entry_id}")

            # Count entries with timestamp less than this entry's timestamp
            stmt = (
                select(func.count())
                .select_from(TriggerEntryTable)
                .where(col(TriggerEntryTable.timestamp) < entry_row.timestamp)
            )
            count = session.exec(stmt).one()
            return count

    def get_last_entry_by_trigger_type(
        self, trigger_type: str
    ) -> TriggerHistoryEntry | None:
        """
        Get the last (most recent) entry with the specified trigger type.

        Args:
            trigger_type: The trigger type to filter by (e.g., "user_input", "wakeup")

        Returns:
            The most recent entry of that type, or None if not found
        """
        with Session(self._engine) as session:
            stmt = (
                select(TriggerEntryTable)
                .where(col(TriggerEntryTable.trigger_type) == trigger_type)
                .order_by(col(TriggerEntryTable.timestamp).desc())
                .limit(1)
            )
            row = session.exec(stmt).first()
            return self._row_to_entry(session, row) if row else None

    def close(self) -> None:
        """Close database connections and release file handles."""
        self._engine.dispose()
        self._chroma_client.close()

    def _row_to_entry(
        self, session: Session, row: TriggerEntryTable
    ) -> TriggerHistoryEntry:
        """Convert database row to TriggerHistoryEntry."""
        trigger = TriggerSerializer.from_blob(row.trigger_blob)

        action_stmt = (
            select(ActionTable)
            .where(ActionTable.trigger_entry_id == row.id)
            .order_by(col(ActionTable.sequence_order))
        )
        action_rows = session.exec(action_stmt).all()
        actions = [ActionSerializer.from_row(ar) for ar in action_rows]

        # Get embedding from ChromaDB if it exists
        embedding_vector: list[float] | None = None
        try:
            result = self._embeddings_collection.get(
                ids=[row.id], include=["embeddings"]
            )
            if result["embeddings"] and len(result["embeddings"]) > 0:
                embedding_vector = [float(x) for x in result["embeddings"][0]]
        except Exception:
            pass  # Entry may not have embedding

        return TriggerHistoryEntry(
            trigger=trigger,
            actions_taken=actions,
            timestamp=row.timestamp,
            end_timestamp=row.end_timestamp,
            entry_id=row.id,
            situational_context=row.situational_context,
            compressed_summary=row.compressed_summary,
            embedding_vector=embedding_vector,
        )


def create_trigger_history(db_path: str | Path) -> TriggerHistory:
    """
    Factory function to create and initialize a TriggerHistory instance.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Initialized TriggerHistory with database tables created.
    """
    history = TriggerHistory(str(db_path))
    history.initialize_db()
    return history
